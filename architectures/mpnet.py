'''
Created on January 26, 2017

@author: Yinglong Miao
@reference: optas

'''
import warnings
import os.path as osp
import time
import tensorflow as tf
import os.path as osp

from tflearn import is_training
from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from . AE.in_out import create_dir, pickle_data, unpickle_data
from . neural_net import Neural_Net

import numpy as np

class Configuration():
    def __init__(self, experiment_name, n_o_input, n_x_input, n_output, encoder, decoder, mlp, pretrain,
                 pretrain_epoch, pretrain_batch_size, fixAE, encoder_args={}, decoder_args={}, mlp_args={},
                 training_epochs=200, batch_size=10, ae_learning_rate=0.001, mlp_learning_rate=0.001, denoising=False,
                 saver_step=None, train_dir=None, z_rotate=False, loss='chamfer', gauss_augment=None,
                 saver_max_to_keep=None, loss_display_step=1, debug=False,
                 latent_vs_recon=1.0, consistent_io=None):

        # Parameters for any AE
        self.experiment_name = experiment_name
        self.is_denoising = denoising
        self.loss = loss.lower()
        self.decoder = decoder
        self.encoder = encoder
        self.mlp = mlp
        self.encoder_args = encoder_args
        self.decoder_args = decoder_args
        self.mlp_args = mlp_args
        # Training related parameters
        self.batch_size = batch_size
        self.ae_learning_rate = ae_learning_rate
        self.mlp_learning_rate = mlp_learning_rate
        self.loss_display_step = loss_display_step
        self.saver_step = saver_step
        self.train_dir = train_dir
        self.gauss_augment = gauss_augment
        self.z_rotate = z_rotate
        self.saver_max_to_keep = saver_max_to_keep
        self.training_epochs = training_epochs
        self.debug = debug
        self.n_o_input = n_o_input
        self.n_x_input = n_x_input
        self.n_output = n_output
        # Used in VAE
        self.latent_vs_recon = np.array([latent_vs_recon], dtype=np.float32)[0]
        self.pretrain = pretrain
        self.fixAE = fixAE


        # Used in AP
        if n_output is None:
            self.n_output = n_input
        else:
            self.n_output = n_output

        self.consistent_io = consistent_io

    def exists_and_is_not_none(self, attribute):
        return hasattr(self, attribute) and getattr(self, attribute) is not None

    def __str__(self):
        keys = self.__dict__.keys()
        vals = self.__dict__.values()
        index = np.argsort(keys)
        res = ''
        for i in index:
            if callable(vals[i]):
                v = vals[i].__name__
            else:
                v = str(vals[i])
            res += '%30s: %s\n' % (str(keys[i]), v)
        return res

    def save(self, file_name):
        pickle_data(file_name + '.pickle', self)
        with open(file_name + '.txt', 'w') as fout:
            fout.write(self.__str__())

    @staticmethod
    def load(file_name):
        return unpickle_data(file_name + '.pickle').next()


class MPNet(Neural_Net):
    '''
    Motion Planner
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c

        Neural_Net.__init__(self, name, graph)
        self.n_o_input = c.n_o_input
        self.n_x_input = c.n_x_input
        #self.n_z_input = c.n_z_input
        self.n_output = c.n_output
        o_shape = [None] + self.n_o_input
        x_shape = [None] + self.n_x_input
        #z_shape = [None] + self.n_z_input
        out_shape = [None] + self.n_output


        with tf.variable_scope(name):
            with tf.variable_scope('AE'):
                self.o = tf.placeholder(tf.float32, o_shape)
                self.z = c.encoder(self.o, **c.encoder_args)
                self.gt = self.o
                if hasattr(c, 'decoder'):
                    self.x_reconstr = c.decoder(self.z, **c.decoder_args)
                    self.x_reconstr = tf.reshape(self.x_reconstr, [-1, o_shape[1], o_shape[2]])

            with tf.variable_scope('mlp'):
                self.x = tf.placeholder(tf.float32, x_shape)
                # concatenate state and obstacle representation
                self.input = tf.concat([self.z, self.x], axis=1)
                self.target = tf.placeholder(tf.float32, out_shape)
                # may also look at decoder if it is defined
                layer = c.mlp(self.input, **c.mlp_args)
                self.output = layer

            self._create_loss()
            self._setup_optimizer()
            self.AE_saver = tf.train.Saver(self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'AE'),\
                                            max_to_keep=c.saver_max_to_keep)
            self.mlp_saver = tf.train.Saver(self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'mlp'),\
                                            max_to_keep=c.saver_max_to_keep)

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def reconstruct(self, pc):
        return self.sess.run((self.x_reconstr), feed_dict={self.o: pc})

    def ae_fit(self, pc):
        is_training(True, session=self.sess)
        _, output, loss = self.sess.run((self.ae_train_step, self.x_reconstr, loss), \
                            feed_dict={self.o: pc})
        is_training(False, session=self.sess)
        return output, loss

    def mlp_fit(self, pc, x, target):
        is_training(True, session=self.sess)
        _, output, loss = self.sess.run((self.mlp_train_step, self.output, loss), \
                            feed_dict={self.o: pc, self.x: x, self.target: target})
        is_training(False, session=self.sess)
        return output, loss


    def _create_loss(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.ae_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.ae_loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.ae_loss += (w_reg_alpha * rl)
        self.mlp_loss = tf.reduce_mean(tf.square_difference(self.output, self.target))

    def _setup_optimizer(self):
        c = self.configuration
        self.ae_lr = c.ae_learning_rate
        self.mlp_lr = c.mlp_learning_rate
        if hasattr(c, 'exponential_decay'):
            self.ae_lr = tf.train.exponential_decay(c.ae_learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.ae_lr = tf.maximum(self.ae_lr, 1e-5)
            tf.summary.scalar('ae_learning_rate', self.ae_lr)
            self.mlp_lr = tf.train.exponential_decay(c.mlp_learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.mlp_lr = tf.maximum(self.mlp_lr, 1e-5)
            tf.summary.scalar('ae_learning_rate', self.mlp_lr)

        self.ae_optimizer = tf.train.AdamOptimizer(learning_rate=self.ae_lr)
        self.ae_train_step = self.ae_optimizer.minimize(self.ae_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ae'))
        # depending on if we are fixing autoencoder or not, set the learnable parameters

        self.mlp_optimizer = tf.train.AdamgradOptimizer(learning_rate=self.mlp_lr)

        if c.fixAE:
            # only update mlp parameters
            self.mlp_train_step = self.mlp_optimizer.minimize(self.mlp_loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'mlp'))
        else:
            self.mlp_train_step = self.mlp_optimizer.minimize(self.mlp_loss)

    def _single_epoch_pretrain(self, train_pc, configuration, only_fw=False):
        '''
        full training the entire mpnet model
        '''
        n_examples = len(train_pc)
        epoch_loss = 0.
        batch_size = configuration.pretrain_batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for i in range(0, n_examples, batch_size):
            pc_i = train_pc[i:i+batch_size]
            _, loss = ae_fit(pc_i)
            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        return epoch_loss, duration

    def _single_epoch_train(self, train_pc, train_pc_inds, train_input, train_targets, configuration, only_fw=False):
        '''
        full training the entire mpnet model
        '''
        n_examples = len(train_input)
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        # Loop over all batches
        for i in range(0, n_examples, batch_size):
            pc_inds = train_pc_inds[i:i+batch_size]
            pc_i = train_pc[pc_inds]
            batch_i = train_input[i:i+batch_size]
            target_i = train_targets[i:i+batch_size]
            _, loss = mlp_fit(pc_i, batch_i, target_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time

        return epoch_loss, duration

    def pretrain(self, train_pc, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []
        print('pretraining...')
        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_pretrain(train_pc, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            # save AutoEncoder and MLP separately
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                ae_checkpoint_path = osp.join(c.train_dir, 'ae.ckpt')
                self.AE_saver.save(self.sess, ae_checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if held_out_data is not None and c.exists_and_is_not_none('held_out_step') and (epoch % c.held_out_step == 0):
                loss, duration = self._single_epoch_train(held_out_data, c, only_fw=True)
                print("Held Out Data :", 'forward time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('On Held_Out: %04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))
        return stats


    def train(self, train_pc, train_pc_inds, train_input, train_targets, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append((epoch, loss, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))

            # Save the models checkpoint periodically.
            # save AutoEncoder and MLP separately
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                if not c.fixAE:
                    # not fixing AutoEncoder, save the autoencoder
                    ae_checkpoint_path = osp.join(c.train_dir, 'ae.ckpt')
                    self.AE_saver.save(self.sess, ae_checkpoint_path, global_step=self.epoch)

                mlp_checkpoint_path = osp.join(c.train_dir, 'mlp.ckpt')
                self.mlp_saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if held_out_data is not None and c.exists_and_is_not_none('held_out_step') and (epoch % c.held_out_step == 0):
                loss, duration = self._single_epoch_train(held_out_data, c, only_fw=True)
                print("Held Out Data :", 'forward time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=", "{:.9f}".format(loss))
                if log_file is not None:
                    log_file.write('On Held_Out: %04d\t%.9f\t%.4f\n' % (epoch, loss, duration / 60.0))
        return stats

    def get_latent_codes(self, pc):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.o: pc})
