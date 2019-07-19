import os.path as osp
import os
import sys
sys.path.append('../../')

from architectures.mlp_pipeline import mlp_pipeline

from architectures.AE.point_net_ae import PointNetAutoEncoder

from architectures.AE.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from architectures.AE.tf_utils import reset_tf_graph
from architectures.AE.general_utils import plot_3d_point_cloud

from architectures.mpnet import Configuration as Conf
from architectures.mpnet import MPNet

from architectures.mlp_models import baxter_mpnet_mlp
from architectures.AE.ae_templates import *

from tools.obs_data_loader import load_dataset
from tools.import_tool import fileImport
from tools.path_data_loader import load_dataset_end2end
import numpy as np
import argparse
#%load_ext autoreload
#%autoreload 2


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.device)
    # define basic parameters
    top_out_dir = args.trained_model_path  # Use to save Neural-Net check-points etc.
    n_pc_points = args.enc_input_size // 3 # Number of points per model.
    bneck_size = args.enc_output_size      # Bottleneck-AE size
    ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'

    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path

    envs = importer.environments_import(env_data_path + args.envs_file)
    print("Loading obstacle data...\n")
    #dataset_train, targets_train, pc_inds_train, obstacles = [], [], [], []
    dataset_train, targets_train, pc_inds_train, obstacles = load_dataset_end2end(
        envs, path_data_path, pcd_data_path, args.path_data_file, importer, NP=100)

    print("Loaded dataset, targets, and pontcloud obstacle vectors: ")
    print(str(len(dataset_train)) + " " +
        str(len(targets_train)) + " " + str(len(pc_inds_train)))
    print("\n")

    if not os.path.exists(args.trained_model_path):
        os.makedirs(args.trained_model_path)

    # write encoder and decoder pipeline
    if args.AE_type == 'pointnet':
        encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    elif args.AE_type == 'linear':
        encoder, decoder, enc_args, dec_args = linear_ae(n_pc_points, bneck_size)

    # write mpnet pipeline
    mlp, mlp_args = baxter_mpnet_mlp(args.mlp_input_size, args.mlp_output_size)

    train_dir = create_dir(osp.join(top_out_dir, args.experiment_name))

    conf = Conf(loss = ae_loss,
            experiment_name = args.experiment_name,
            training_epochs = args.num_epochs,
            batch_size = args.batch_size,
            denoising = False,
            ae_learning_rate = args.ae_learning_rate,
            mlp_learning_rate = args.mlp_learning_rate,
            train_dir = train_dir,
            loss_display_step = args.loss_display_step,
            saver_step = args.saver_step,
            z_rotate =args.z_rotate,
            encoder = encoder,
            decoder = decoder,
            mlp = mlp,
            encoder_args = enc_args,
            decoder_args = dec_args,
            mlp_args = mlp_args,
            n_o_input = [n_pc_points, 3],
            n_x_input = [args.mlp_input_size-args.enc_output_size],
            n_output = [args.mlp_output_size],
            pretrain = args.pretrain,
            pretrain_epochs = args.pretrain_epochs,
            pretrain_batch_size = args.pretrain_batch_size,
            fixAE = args.fixAE
           )
    conf.held_out_step = 5   # How often to evaluate/print out loss on
                             # held_out data (if they are provided in ae.train() ).
    conf.AE_type = args.AE_type
    conf.save(osp.join(train_dir, 'configuration'))
    # reload
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    mpnet = MPNet(conf.experiment_name, conf)
    if args.AE_start_epoch > 0:
        if args.AE_restore_pretrain:
            # restoring from pretrained AE
            # filename is pretrain_ae
            mpnet.restore_model(mpnet.AE_saver, conf.train_dir, epoch=args.AE_start_epoch, filename='pretrain_ae')
        else:
            # otherwise, restore from end2end trained AE
            # filename is ae
            mpnet.restore_model(mpnet.AE_saver, conf.train_dir, epoch=args.AE_start_epoch, filename='ae')
    if args.start_epoch > 0:
        mpnet.restore_model(mpnet.mlp_saver, conf.train_dir, epoch=args.start_epoch, filename='mlp')
    # print out mpnet model structure
    vars = tf.trainable_variables()
    print(vars) #some infos about variables...
    vars_vals = mpnet.sess.run(vars)
    for var, val in zip(vars, vars_vals):
        print("var: {}, value: {}".format(var.name, val))

    # training AE
    if args.pretrain:
        # load point cloud data for pretraining
        envs_files = [args.pretrain_envs_file]
        pretrain_obstacles = []
        env_data_path = args.env_data_path
        pcd_data_path = args.pointcloud_data_path
        for envs_file in envs_files:
            envs = importer.environments_import(env_data_path + envs_file)
            print("Loading obstacle data...\n")
            obs = load_dataset(envs, pcd_data_path, importer)
            pretrain_obstacles.append(obs)

        pretrain_obstacles = np.stack(pretrain_obstacles).astype(float)[0].reshape(len(pretrain_obstacles[0]),-1,3)
        buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
        fout = open(osp.join(conf.train_dir, 'pretrain_stats.txt'), 'a', buf_size)
        train_stats = mpnet.pretrain(pretrain_obstacles, conf, log_file=fout)
        fout.close()

    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    if args.AE_type == 'pointnet':
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
    elif args.AE_type == 'linear':
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1)
    train_stats = mpnet.train(obstacles, pc_inds_train, dataset_train, targets_train, conf, log_file=fout)
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='mpnet')

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/train/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/train/pcd/')
    parser.add_argument('--trained_model_path', type=str, default='./models/sample_train/', help='path for saving trained models')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--ae_learning_rate', type=float, default=0.001)
    parser.add_argument('--mlp_learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)

    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainPaths.pkl')

    parser.add_argument('--pretrain_envs_file', type=str, default='trainEnvironmentsLarge.pkl')

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--AE_type', type=str, default='pointnet')
    parser.add_argument('--AE_start_epoch', type=int, default=0)
    parser.add_argument('--AE_restore_pretrain', type=int, default=0, help='indicate if AutoEncoder is going to restore the pretrained model or the end2end model')

    parser.add_argument('--pretrain', type=int, default=0, help='indicate if pretraining AutoEncoder or not')
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--pretrain_batch_size', type=int, default=100)
    parser.add_argument('--fixAE', type=int, default=0, help='fix AutoEncoder or not when training with MLP')

    parser.add_argument('--loss_display_step', type=int, default=1)
    parser.add_argument('--saver_step', type=int, default=10)
    parser.add_argument('--z_rotate', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    main(args)
