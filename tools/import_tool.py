import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
# import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
import pypcd

class fileImport():
	def __init__(self):
		#parameters of the baxter
		self.joint_range = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])
		self.joint_names = ['s0', 's1', 'w0', 'w1', 'w2', 'e0', 'e1']

	def moveit_unscramble(self, paths):
		"""
		Function to take in a set of Baxter paths, with joint angles ordered in MoveIt fashion,
		and reorder all the waypoints in the path for the Baxter interface joint angle order,
		and normalize using the joint ranges

		Input: paths (list) - list of numpy arrays, each array is a (Nx7) array of waypoints

		Return: new_paths (list) - list of numpy arrays, same dimension as input
		"""
		new_paths = []
		for path in paths:
			new_path = np.zeros((path.shape[0], 7))
			new_path[:, 0:2] = path[:, 0:2] #replace proper indices in each path
			new_path[:, 2:5] = path[:, 4:]
			new_path[:, 5:] = path[:, 2:4]

			new_path = np.divide(new_path, self.joint_range) #normalize with joint range
			new_paths.append(new_path)

		return new_paths

	def paths_import_all(self, path_fname):
		"""
		Import all paths from a file with paths from all environments, ofand return into a single a dictionary
		keyed by environment name. Unscramle the path data and normalize it in the process

		Input: path_fname (string)

		Return: unscrambled_dict (dictionary)
		"""
		with open (path_fname, "rb") as paths_f:
			paths_dict = pickle.load(paths_f)

		# keyed by environment name
		unscrambled_dict = {}
		for key in paths_dict.keys():
			unscrambled_dict[key] = self.moveit_unscramble(paths_dict[key])

		return unscrambled_dict

	def paths_import_single(self, path_fname, env_name, single_env=False):
		"""
		Import the paths from a single environment. File to load from may contain data from many environments, or
		just a single one, indicated by the single_env (True/False) flag

		Input: 	path_fname (string) - filepath to file with path data
				env_name (string) - name of environment to get path data for
				single_env (bool) - flag of whether path_fname indicates a file with data from multiple environments or a single env

		Return: env_paths (list) - list of numpy arrays
		"""
		if not single_env:
			with open (path_fname, "rb") as paths_f:
				paths_dict = pickle.load(paths_f)

			# for non single environment, need to use the environment name as a dictionary key to get the right path list
			env_paths = self.moveit_unscramble(paths_dict[env_name])
			return env_paths

		else:
			with open (path_fname, "rb") as paths_f:
				paths_list = pickle.load(paths_f)

			env_paths = self.moveit_unscramble(paths_list)
			return env_paths

	def pointcloud_import_array(self, pcd_fname, min_length_array):
		"""
		Import the pointcloud data from a file, and leave it in a (3xN) array

		Input: 	pcd_fname (string) - filepath of file with pcd data
				min_length_array (int) - index to chop all rows of the pointcloud data to, based on environment with minimum number of points

		Return: obs_pc (numpy array) - array of pointcloud data (3XN) [x, y, z]
		"""
		pc = pypcd.PointCloud.from_path(pcd_fname)

		# flatten into vector
		# obs_pc = np.zeros((3, pc.pc_data['x'].shape[0]))
		obs_pc = np.zeros((3, min_length_array))
		obs_pc[0] = pc.pc_data['x'][~np.isnan(pc.pc_data['x'])][:min_length_array]
		obs_pc[1] = pc.pc_data['y'][~np.isnan(pc.pc_data['x'])][:min_length_array]
		obs_pc[2] = pc.pc_data['z'][~np.isnan(pc.pc_data['x'])][:min_length_array]

		return obs_pc

	def pointcloud_import(self, pcd_fname):
		"""Import the pointcloud data from a file, and flatten it into a vector

		Input: 	pcd_fname (string) - filepath of file with pcd data

		Return: obs_pc (numpy array) - array of pointcloud data (1X(3N))
		"""
		print('pointcloud filename:')
		print(pcd_fname)
		pc = pypcd.PointCloud.from_path(pcd_fname)

		# flatten into vector
		temp = []
		temp.append(pc.pc_data['x'][~np.isnan(pc.pc_data['x'])])
		temp.append(pc.pc_data['y'][~np.isnan(pc.pc_data['x'])])
		temp.append(pc.pc_data['z'][~np.isnan(pc.pc_data['x'])])
		temp = np.array(temp)
		print(temp.shape)
		obs_pc = temp.flatten('F') #flattened column wise, [x0, y0, z0, x1, y1, z1, x2, y2, ...]

		return obs_pc

	def pontcloud_length_check(self, pcd_fname):
		"""
		Get number of points in the pointcloud file pcd_fname
		"""
		pc = self.pointcloud_import(pcd_fname)
		return pc.shape[0]

	def environments_import(self, envs_fname):
		"""
		Import environments from files with description of where obstacles reside, dictionary keyed by 'poses' and 'obsData'.
		This function uses the poses key, which has the positions of all the environment obstacles

		Input: envs_fname (string) - filepath of file with environment data
		Return: env_names (list) - list of strings, based on keys of dictionary in envs['poses']
		"""
		with open (envs_fname, "rb") as env_f:
			envs = pickle.load(env_f)
		env_names = envs['poses'].keys() # also has obstacle meta data
		return env_names
