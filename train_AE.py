import os.path as osp
import os
import sys
sys.path.append('../../')

from architectures.AE.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from architectures.AE.autoencoder import Configuration as Conf
from architectures.AE.point_net_ae import PointNetAutoEncoder

from architectures.AE.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from architectures.AE.tf_utils import reset_tf_graph
from architectures.AE.general_utils import plot_3d_point_cloud

from tools.obs_data_loader import load_dataset
from tools.import_tool import fileImport
import numpy as np
import argparse
#%load_ext autoreload
#%autoreload 2


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # define basic parameters
    top_out_dir = args.trained_model_path  # Use to save Neural-Net check-points etc.
    n_pc_points = args.enc_input_size // 3 # Number of points per model.
    bneck_size = args.enc_output_size      # Bottleneck-AE size
    ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'


    # load point cloud data
    importer = fileImport()
    envs_files = ['trainEnvironmentsLarge.pkl']
    obstacles = []
    env_data_path = args.env_data_path
    pcd_data_path = args.pointcloud_data_path
    for envs_file in envs_files:
        envs = importer.environments_import(env_data_path + envs_file)
        print("Loading obstacle data...\n")
        obs = load_dataset(envs, pcd_data_path, importer)
        obstacles.append(obs)

    all_pc_data = np.stack(obstacles).astype(float)[0].reshape(len(obs),-1,3)
    all_pc_data = PointCloudDataSet(all_pc_data, init_shuffle=False)

    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))

    conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = args.training_epochs,
            batch_size = args.batch_size,
            denoising = args.denoising,
            learning_rate = args.learning_rate,
            train_dir = train_dir,
            loss_display_step = args.loss_display_step,
            saver_step = args.saver_step,
            z_rotate =args.z_rotate,
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
    conf.experiment_name = args.experiment_name
    conf.held_out_step = 5   # How often to evaluate/print out loss on
                             # held_out data (if they are provided in ae.train() ).
    conf.save(osp.join(train_dir, 'configuration'))
    # reload
    if args.start_epoch > 0:
        conf = Conf.load(train_dir + '/configuration')
        reset_tf_graph()
        ae = PointNetAutoEncoder(conf.experiment_name, conf)
        ae.restore_model(conf.train_dir, epoch=args.start_epoch)

    # build AE model
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)

    # training AE
    buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
    fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
    train_stats = ae.train(all_pc_data, conf, log_file=fout)
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/train/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/train/pcd/')
    parser.add_argument('--trained_model_path', type=str, default='./models/sample_train/', help='path for saving trained models')

    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=1000)

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainPaths.pkl')

    parser.add_argument('--denoising', type=int, default=0)
    parser.add_argument('--z_rotate', type=int, default=0)
    parser.add_argument('--saver_step', type=int, default=10)
    parser.add_argument('--loss_display_step', type=int, default=1)
    parser.add_argument('--environment_name', type=str, default='pointcloud_linear_ae')

    args = parser.parse_args()
    main(args)
