import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import os.path
import random
import sys
from tools.import_tool import fileImport
from multiprocessing import Pool

import fnmatch
def pool_pc_length_check(args):
    fname, importer, pcd_data_path = args
    return importer.pointcloud_length_check(pcd_fname=pcd_data_path + fname)

def pool_pc_load(args):
    try:
        fname, min_length, importer, pcd_data_path = args
        data = importer.pointcloud_import(pcd_fname=pcd_data_path + fname)[:min_length]
        return data
    except:
        return None

def load_dataset(env_names,pcd_data_path,importer,min_length=(5351*3)):
    """
    Load point cloud dataset into array of obstacle pointclouds, which will be entered as input to the encoder NN

    Input:     env_names (list) - list of strings with names of environments to import
            pcd_data_path (string) - path to folder with pointcloud files
            importer (fileImport) - object from utility library to help with importing different data
            min_length (int) - if known in advance, number of flattened points in the shortest obstacle point cloud vector

    Return: obstacles (numpy array) - array of obstacle point clouds, with different rows for different environments
                                        and different columns for points
    """
    # get file names based on environment names, just grabbing first one available (sometimes there's multiple)
    fnames = []

    print("Searing for file names...")
    for i, env in enumerate(env_names):
        # hacky reordering so that we don't load the last .pcd file which is always corrupt
        # sort by the time step on the back, hopefully that helps it obtain the earliest possible
        for file in sorted(os.listdir(pcd_data_path), key=lambda x: int(x.split('Env_')[1].split('_')[1][:-4])):
            if (fnmatch.fnmatch(file, env+"*")):
                #print('failed')
                fnames.append(file)
                break
        print('loaded %d envs:' % (len(fnames)))
        if len(fnames) >= 2000:  # break  the loop when there are at least 2000 pcs
            break
    if min_length is None: # compute minimum length for dataset will take twice as long if necessary
        lengths = []
        min_length = 1e6 # large to start
        # multiprocessing loading
        pool = Pool(8)
        args = [(fnames[i], importer, pcd_data_path) for i in range(len(fnames))]
        for i, length in enumerate(pool.imap(pool_pc_length_check, args)):
            if (length < min_length):
                min_length = length
        pool.close()
        pool.join()
    print("Loading files, minimum point cloud obstacle length: ")
    print(min_length)
    N = len(fnames)
    print('env_names:')
    print(len(env_names))
    print("N")
    print(N)
    obstacles=np.zeros((N,min_length),dtype=np.float32)
    # multiprocessing loading
    pool = Pool(8)

    obstacles = []
    args = [(fnames[i], min_length, importer, pcd_data_path) for i in range(len(fnames))]
    for i, data in enumerate(pool.imap(pool_pc_load, args)):
        if data is not None:
            obstacles.append(data)
    pool.close()
    pool.join()
    obstacles = np.stack(obstacles)

    return obstacles

def load_normalized_dataset(env_names,pcd_data_path,importer,min_length=(5351*3)):
	"""
	Load point cloud dataset into array of obstacle pointclouds, which will be entered as input to the encoder NN, but first normalizing all the data based on mean and norm

	Input: 	env_names (list) - list of strings with names of environments to import
			pcd_data_path (string) - filepath to file with environment representation
			importer (fileImport) - object from utility library to help with importing different data
			min_length (int) - if known in advance, number of flattened points in the shortest obstacle point cloud vector

	Return: obstacles (numpy array) - array of obstacle point clouds, with different rows for different environments
										and different columns for points
	"""
	# get file names, just grabbing first one available (sometimes there's multiple)
	fnames = []

	print("Searing for file names...")
	for i, env in enumerate(env_names):
		# hacky reordering so that we don't load the last .pcd file which is always corrupt
		# sort by the time step on the back, which helps us obtain the earliest possible
		for file in sorted(os.listdir(pcd_data_path), key=lambda x: int(x.split('Env_')[1].split('_')[1][:-4])):
			if (fnmatch.fnmatch(file, env+"*")):
				fnames.append(file)
				break

	if min_length is None: # compute minimum length for dataset will take twice as long if necessary
		min_length = 1e6 # large to start
		for i, fname in enumerate(fnames):
			length = importer.pointcloud_length_check(pcd_fname=pcd_data_path + fname)
			if (length < min_length):
				min_length = length

	print("Loading files, minimum point cloud obstacle length: ")
	print(min_length)
	N = len(fnames)

	# make empty array of known length, and use import tool to fill with obstacle pointcloud data
	min_length_array = min_length/3
	obstacles_array = np.zeros((3, min_length_array, N), dtype=np.float32)
	for i, fname in enumerate(fnames):
		data = importer.pointcloud_import_array(pcd_data_path + fname, min_length_array) #using array version of import, and will flatten manually after normalization
		obstacles_array[:, :, i] = data

	# compute mean and std of each environment
	means = np.mean(obstacles_array, axis=1)
	stds = np.std(obstacles_array, axis=1)
	norms = np.linalg.norm(obstacles_array, axis=1)

	# compute mean and std of means and stds
	mean_overall = np.expand_dims(np.mean(means, axis=1), axis=1)
	std_overall = np.expand_dims(np.std(stds, axis=1), axis=1)
	norm_overall = np.expand_dims(np.mean(norms, axis=1), axis=1)

	print("mean: ")
	print(mean_overall)
	print("std: ")
	print(std_overall)
	print("norm: ")
	print(norm_overall)

	# normalize data based on mean and overall norm, and then flatten into vector
	obstacles=np.zeros((N,min_length),dtype=np.float32)
	for i in range(obstacles_array.shape[2]):
		temp_arr = (obstacles_array[:, :, i] - mean_overall)
		temp_arr = np.divide(temp_arr, norm_overall)
		obstacles[i] = temp_arr.flatten('F')

	return obstacles

def main():
	importer = fileImport()
	env_path = '../env/environment_data/'
	data_path = '../data/train/'
	pcd_path = data_path + 'pcd/'

	envs = importer.environments_import(env_path + 'trainEnvironments_GazeboPatch.pkl')
	obstacles = load_normalized_dataset(envs,pcd_path,importer)
	if (obstacles.any()):
		print("loaded obstacles")
		print("obstacle size: ")
		print(obstacles.shape)
	else:
		print("failed to load obstacles")



if __name__ == '__main__':
	main()
