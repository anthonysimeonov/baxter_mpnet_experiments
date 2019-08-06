import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tools.path_data_loader import load_dataset_end2end
from torch.autograd import Variable
import math
from tools.import_tool import fileImport
import time
import sys

###
from torch_architectures import MLP, MLP_Path, Encoder, Encoder_End2End, VoxelEncoder


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_input(i, data, targets, pc_inds, obstacles, bs):
    """
    Input:  i (int) - starting index for the batch
            data/targets/pc_inds (numpy array) - data vectors to obtain batch from
            obstacles (numpy array) - point cloud array
            bs (int) - batch size
    """
    if i+bs < len(data):
        bi = data[i:i+bs]
        bt = targets[i:i+bs]
        bpc = pc_inds[i:i+bs]
        bobs = obstacles[bpc]
    else:
        bi = data[i:]
        bt = targets[i:]
        bpc = pc_inds[i:]
        bobs = obstacles[bpc]

    return torch.from_numpy(bi), torch.from_numpy(bt), torch.from_numpy(bobs).type(torch.FloatTensor)


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.device)

    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path

    envs = importer.environments_import(env_data_path + args.envs_file)

    print("Loading obstacle data...\n")
    dataset_train, targets_train, pc_inds_train, obstacles = load_dataset_end2end(
        envs, path_data_path, pcd_data_path, args.path_data_file, importer, NP=args.NP)

    print("Loaded dataset, targets, and pontcloud obstacle vectors: ")
    print(str(len(dataset_train)) + " " +
        str(len(targets_train)) + " " + str(len(pc_inds_train)))
    print("\n")

    for i in range(len(dataset_train)):
        print(dataset_train[i])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/train/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/train/pcd/')
    parser.add_argument('--trained_model_path', type=str, default='./models/sample_train/', help='path for saving trained models')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)

    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainPaths.pkl')
    parser.add_argument('--AE_type', type=str, default='linear')
    parser.add_argument('--NP', type=int, default=1000)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    main(args)
