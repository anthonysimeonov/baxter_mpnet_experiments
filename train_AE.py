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
from architectures.mlp import MLP, MLP_Path
from architectures.AE.pointnetAE_linear import Encoder, Decoder
from architectures.AE.chamfer_distance import ChamferDistance
#from architectures import MLP, MLP_Path, Encoder, Encoder_End2End
from tools.obs_data_loader import load_dataset

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

    return torch.from_numpy(bi), torch.from_numpy(bt), torch.from_numpy(bobs)


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path
    # append all envs and obstacles
    envs_files = os.listdir(env_data_path)
    obstacles = []
    for envs_file in envs_files:
        envs = importer.environments_import(env_data_path + envs_file)

        print("Loading obstacle data...\n")
        obs = load_dataset(envs, pcd_data_path, importer)
        obstacles.append(obs)
        print(obs.shape)
    obstacles = np.stack(obstacles)
    print("Loaded dataset, targets, and pontcloud obstacle vectors: ")
    print("\n")

    if not os.path.exists(args.trained_model_path):
        os.makedirs(args.trained_model_path)

    # Build the models
    #mlp = MLP(args.mlp_input_size, args.mlp_output_size)
    encoder = Encoder(args.enc_input_size, args.enc_output_size)
    decoder = Decoder(args.enc_output_size, args.enc_input_size)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()


    # Loss and Optimizer
    criterion = ChamferDistance()
    params = list(encoder.parameters())+list(decoder.parameters())
    optimizer = torch.optim.Adagrad(params, lr=args.learning_rate)

    total_loss = []
    epoch = 1

    sm = 1  # start saving models after 100 epochs

    print("Starting epochs...\n")
    # epoch=1
    done = False
    for epoch in range(args.num_epochs):
        # while (not done)
        start = time.time()
        print("epoch" + str(epoch))
        avg_loss = 0
        for i in range(0, int(args.train_ratio*len(obstacles)), args.batch_size):
            # Forward, Backward and Optimize
            # zero gradients
            encoder.zero_grad()
            decoder.zero_grad()
            # convert to pytorch tensors and Varialbes
            bobs = torch.from_numpy(obstacles[i:i+args.batch_size])
            bobs = to_var(bobs).view(len(bobs), 3, -1)

            # forward pass through encoder
            h = encoder(bobs)

            # decoder
            bt = decoder(h)

            # compute overall loss and backprop all the way
            loss1, loss2 = criterion(bobs, bt)
            loss = torch.mean(loss1) + torch.mean(loss2)
            avg_loss = avg_loss+loss.data
            loss.backward()
            optimizer.step()

        print("--average loss:")
        print(avg_loss/(int(args.train_ratio*len(obstacles))/args.batch_size))
        total_loss.append(avg_loss/(int(args.train_ratio*len(obstacles))/args.batch_size))
        # Save the models
        if epoch == sm:
            print("\nSaving model\n")
            print("time: " + str(time.time() - start))
            torch.save(encoder.state_dict(), os.path.join(
                args.trained_model_path, 'pointnet_encoder_'+str(epoch)+'.pkl'))
            torch.save(decoder.state_dict(), os.path.join(
                args.trained_model_path, 'pointnet_decoder_'+str(epoch)+'.pkl'))
            torch.save(total_loss, 'total_loss_'+str(epoch)+'.dat')
            #if (epoch != 1):
            sm = sm+1  # save model after every 50 epochs from 100 epoch ownwards

    torch.save(total_loss, 'total_loss.dat')

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

    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainPaths.pkl')

    args = parser.parse_args()
    main(args)
