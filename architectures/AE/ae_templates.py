'''
Created on September 2, 2017

@author: optas
'''
import numpy as np

from . encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only, linear_encoder
import tensorflow as tf
def wrap_prelu(_x, name=None):
    # ref: https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.25),
                            dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def mlp_architecture_ala_iclr_18(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = encoder_with_convs_and_symmetry
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'n_filters': [64, 128, 128, 256, bneck_size],
                    'filter_sizes': [1],
                    'strides': [1],
                    'b_norm': True,
                    'verbose': True
                    }

    decoder_args = {'layer_sizes': [256, 256, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True
                    }

    if bneck_post_mlp:
        encoder_args['n_filters'].pop()
        decoder_args['layer_sizes'][0] = bneck_size

    return encoder, decoder, encoder_args, decoder_args

def linear_ae(n_pc_points, bneck_size, bneck_post_mlp=False):
    ''' Single class experiments.
    '''
    #if n_pc_points != 2048:
    #    raise ValueError()

    encoder = linear_encoder
    decoder = decoder_with_fc_only

    n_input = [n_pc_points, 3]

    encoder_args = {'layer_sizes': [786, 512, 256, bneck_size],
                    'verbose': True,
                    'weight_decay': 0.,
                    'non_linearity': wrap_prelu
                    }

    decoder_args = {'layer_sizes': [256, 512, 786, np.prod(n_input)],
                    'b_norm': False,
                    'b_norm_finish': False,
                    'verbose': True,
                    'weight_decay': 0.,
                    'non_linearity': wrap_prelu
                    }

    return encoder, decoder, encoder_args, decoder_args



def default_train_params(single_class=True):
    params = {'batch_size': 50,
              'training_epochs': 500,
              'denoising': False,
              'learning_rate': 0.0005,
              'z_rotate': False,
              'saver_step': 10,
              'loss_display_step': 1
              }

    if not single_class:
        params['z_rotate'] = True
        params['training_epochs'] = 1000

    return params
