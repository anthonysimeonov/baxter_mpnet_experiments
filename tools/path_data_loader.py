import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
from import_tool import fileImport
import fnmatch
from obs_data_loader import load_normalized_dataset

# Environment Encoder
# class Encoder(nn.Module):
# 	def __init__(self):
# 		super(Encoder, self).__init__()
# 		self.encoder = nn.Sequential(nn.Linear(16053, 786), nn.PReLU(),
# 									 nn.Linear(786, 512), nn.PReLU(),
# 									 nn.Linear(512, 256),nn.PReLU(),
# 									 nn.Linear(256, 60))

# 	def forward(self, x):
# 		x = self.encoder(x)
# 		return x


def load_dataset_end2end(env_names, data_path, pcd_path, path_data_file, importer, NP=940, min_length=5351*3):
	"""
	Load dataset for end to end encoder+planner training, which will return shuffled training data, targets and corresponding obstacle point cloud indices corresponding to shuffle

	Input:	env_names (list) - list of string names of environments to load
			data_path (string) - path to directory with path data files
			pcd_path (string) - path to directory with point cloud data files
			importer (fileImport) - object from lib to help with import functions
			NP=940 (int) - number of paths to import from each file (should by 1000, but some files ended up with less during data generation)
			min_length (int) - known number of points in point cloud with minimum number of points (None if not known)
	
	Return: dataset_shuffle (numpy array) - shuffled dataset with start, goal, obstacle encoding
			targets_shuffle (numpy array) - shuffled array of targets for next state to compare to NN prediction and compute loss
			pointcloud_inds_shuffle (numpy array) - shuffled array of indices corresponding to the environments, so that obstacles can be properly loaded and passed through encoder
			obstacles (numpy array) - array of pointcloud data for each environment
	"""	
	N = len(env_names)
	obstacles = load_normalized_dataset(env_names, pcd_path, importer)

	### obtain path length data ###
	paths_file = path_data_file

	# calculating length of the longest trajectory
	max_length = 0
	path_lengths = np.zeros((N, NP), dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		print("env len: " + str(len(env_paths)))
		print("i: " + str(i))
		for j in range(0, NP):  # for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j]) > max_length:
				max_length = len(env_paths[j])

	print("Obtained max path length: \n")
	print(max_length)

	### obtain path data ###

	# padded paths #7D from 2D originally
	paths = np.zeros((N, NP, max_length, 7), dtype=np.float32)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		for j in range(0, NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	print("Obtained paths,for envs: ")
	print(len(paths))
	print("Path matrix shape: ")
	print(paths.shape)
	print("\n")

	### create dataset and targets ###

	dataset = []
	targets = []
	pointcloud_inds = []
	# obs_rep_sz = obs_rep[0].shape[0] ?
	for i, env in enumerate(env_names):
		for j in range(0, NP):
			if path_lengths[i][j] > 0:
				for m in range(0, path_lengths[0][j]-1):
					data = np.zeros(14, dtype=np.float32)

					for joint in range(7):
						data[joint] = paths[i][j][m][joint]
						data[joint + 7] = paths[i][j][path_lengths[i][j] - 1][joint]

					pointcloud_inds.append(i)
					targets.append(paths[i][j][m+1])
					dataset.append(data)

	# clean up paths by removing first element
	targets_new = targets[1:]
	dataset_new = dataset[1:]
	pointcloud_inds_new = pointcloud_inds[1:]

	data = zip(dataset_new, targets_new, pointcloud_inds_new)
	random.shuffle(data)
	dataset_shuffle, targets_shuffle, pointclouds_inds_shuffle = zip(*data)

	# , dataset_clean, targets_clean, paths_new, path_lengths_new
	return np.asarray(dataset_shuffle), np.asarray(targets_shuffle), np.asarray(pointclouds_inds_shuffle), obstacles


def load_test_dataset_end2end(env_names, data_path, pcd_path, path_data_file, importer, NP=80, min_length=5351*3):
	"""
	Load dataset for end to end encoder+planner testing, which will return obstacle point clouds, paths, and path lengths

	Input:	env_names (list) - list of string names of environments to load
			data_path (string) - path to directory with path data files
			pcd_path (string) - path to directory with point cloud data files
			importer (fileImport) - object from lib to help with import functions
			NP (int) - number of paths to import from each file (should by 1000, but some files ended up with less during data generation)
			min_length (int) - known number of points in point cloud with minimum number of points (None if not known)

	Return: obstacles (numpy array) - array of pointcloud data for each environment
			paths_new (numpy array) - numpy array with test paths
			path_lenghts_new (numpy array) - numpy array with lengths of each path, to know where goal index is (the rest are padded with zeros)
	"""
	N = len(env_names)
	obstacles = load_normalized_dataset(env_names, pcd_path, importer)

	### obtain path length data ###
	# paths_file = 'trainEnvironments_testPaths_GoalsCorrect_RRTSTAR_trainEnv_4.pkl'
	paths_file = path_data_file
	print("LOADING FROM: ")
	print(paths_file)
	# calculating length of the longest trajectory
	max_length = 0
	path_lengths = np.zeros((N, NP), dtype=np.int64)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		print("env len: " + str(len(env_paths)))
		print("i: " + str(i))
		print("env name: " + env)
		for j in range(0, NP):  # for j in num_paths:
			path_lengths[i][j] = len(env_paths[j])
			if len(env_paths[j]) > max_length:
				max_length = len(env_paths[j])

	print("Obtained max path length: \n")
	print(max_length)

	### obtain path data ###

	# padded paths #7D from 2D originally
	paths = np.zeros((N, NP, max_length, 7), dtype=np.float32)
	for i, env in enumerate(env_names):
		env_paths = importer.paths_import_single(
			path_fname=data_path+paths_file, env_name=env, single_env=False)
		for j in range(0, NP):
			paths[i][j][:len(env_paths[j])] = env_paths[j]

	print("Obtained paths,for envs: ")
	print(len(paths))
	print("Path matrix shape: ")
	print(paths.shape)
	print("\n")

	### create dataset and targets ###

	# clean up paths
	paths_new = paths[:, :, 1:, :]
	path_lengths_new = path_lengths - 1

	return obstacles, paths_new, path_lengths_new

def main():
	importer = fileImport()
	env_path = '../env/environment_data/'
	data_path = '../data/train/'
	pcd_path = data_path + 'pcd/'

	envs = importer.environments_import(env_path + 'trainEnvironments_GazeboPatch.pkl')
	dataset, targets, inds, obs = load_dataset_end2end(envs, data_path, pcd_path, 'trainPathsLarge_RRTSTAR_Fix.pkl',importer)

	print("Obtained dataset, length: ")
	print(len(dataset))
	print("Obtained targets, length: ")
	print(len(targets))
	print("Environment indices, length: ")
	print(len(inds))
	print("Obstacle point cloud, size: ")
	print(obs.shape)


if __name__ == '__main__':
	main()
