import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tools.path_data_loader import load_test_dataset_end2end
from torch.autograd import Variable
import math
from tools.import_tool import fileImport
import time
import rospy
import sys

from neuralplanner_functions import *
from architectures import *
from tools.planning_scene_editor import *
from tools.get_state_validity import StateValidity


joint_ranges = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])

### collision check

def check_full_path(overall_path):
    invalid = []

    valid = True
    overall_valid = True
    for i, state in enumerate(overall_path):
        filler_robot_state[10:17] = moveit_scrambler(state)
        rs_man.joint_state.position = tuple(filler_robot_state)
        collision_free = sv.getStateValidity(rs_man, group_name="right_arm")

        valid = valid and collision_free
        overall_valid = overall_valid and collision_free

        if not valid:
#             print("invalid: " + str(i))
            invalid.append(i)
            valid = True

    if (len(invalid)==0 and overall_valid):
        print("Full path valid!")
    else:
        print("Not valid")
        
    return overall_valid

def path_to_np(path):
    path_np = []
    for i, state in enumerate(path):
        path_np.append(np.multiply(state.numpy(), joint_ranges)) #unnormalize
    return path_np

def make_overall_path(path_np):
    dists = []
    for i in range(1,len(path_np)):
        dists.append(np.mean(abs(path_np[i] - path_np[i-1]), axis=0))

    overall_dist = sum(dists)
    fractions = [x for x in dists/overall_dist]
    total_pts = 300
    pts = [int(total_pts * x) for x in fractions]
    path_full = []
    for i, n_pts in enumerate(pts):
        vec = np.transpose(np.linspace(path_np[i][0], path_np[i+1][0], n_pts)[np.newaxis])
        for j in range(1,7):
            vec = np.hstack([vec, np.transpose(np.linspace(path_np[i][j], path_np[i+1][j], n_pts)[np.newaxis])])
        path_full.append(vec)

    overall_path = []
    for mini_path in path_full:
        for state in mini_path:
            overall_path.append(state)
            
    return overall_path

def move_to_start(overall_path):
    joint_state = limb.joint_angles()
    for i, name in enumerate(joint_state.keys()):
        joint_state[name] = overall_path[0][i]
    
    limb.move_to_joint_positions(joint_state)  

def play_smooth(overall_path, sleep=False):
    
    joint_state = limb.joint_angles()
    for i, name in enumerate(joint_state.keys()):
        joint_state[name] = overall_path[0][i]
    
    limb.move_to_joint_positions(joint_state)

    k = 1
    
    joint_state = limb.joint_angles()
    if sleep:
        print("sleeping for 5 seconds to setup")
        time.sleep(5)
    
    done = False

    while not done:
        for i, name in enumerate(joint_state.keys()):
            joint_state[name] = overall_path[k][i]
        # limb.move_to_joint_positions(joint_state)
        limb.set_joint_positions(joint_state)
        time.sleep(0.025)
        k += 1
        if k > len(overall_path)-1:
            done = True

def main(args):

    global sv
    global filler_robot_state
    global rs_man
    global limb

    # # Build data loader
    importer = fileImport()
    env_data_path = args.env_data_path

    envs = importer.environments_import(env_data_path + args.envs_file)
    with open(env_data_path+args.envs_file, 'rb') as env_f:
        envDict = pickle.load(env_f)

    rospy.init_node("playback")

    # sometimes takes a few tries to connect to robot arm
    limb_init = False
    while not limb_init:
        try:
            limb = baxter_interface.Limb('right')
            limb_init = True
        except OSError:
            limb_init = False

    scene = PlanningSceneInterface()
    robot = RobotCommander()
    group = MoveGroupCommander("right_arm")
    scene._scene_pub = rospy.Publisher('planning_scene',
                                    PlanningScene,
                                    queue_size=0)

    sv = StateValidity()
    set_environment(robot, scene)

    masterModifier = ShelfSceneModifier()
    sceneModifier = PlanningSceneModifier(envDict['obsData'])
    sceneModifier.setup_scene(scene, robot, group)


    rs_man = RobotState()
    robot_state = robot.get_current_state()
    rs_man.joint_state.name = robot_state.joint_state.name
    filler_robot_state = list(robot_state.joint_state.position)

    env_name = 'trainEnv_' + str(args.env_to_load)
    print(env_name)

    sceneModifier.delete_obstacles()
    new_pose = envDict['poses'][env_name]
    sceneModifier.permute_obstacles(new_pose)

    with open(args.good_path_sample_path + '/' + args.experiment_name + '/' + env_name + '/fp_' + str(args.path_file_to_play) + '.pkl', 'rb') as good_f:
        path = pickle.load(good_f)

    path_np = path_to_np(path)
    overall_path = make_overall_path(path_np)
    valid = check_full_path(overall_path)

    if valid:
        move_to_start(overall_path)
        play_smooth(overall_path)
    else:
        print("Path not valid! Not executing")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--good_path_sample_path', type=str,default='./path_samples/good_path_samples/sample/')
    parser.add_argument('--bad_path_sample_path', type=str, default='./path_samples/bad_path_samples/sample/')
    parser.add_argument('--experiment_name', type=str, default='test_experiment')
    parser.add_argument('--env_to_load', type=int, default=4)
    parser.add_argument('--path_file_to_play', type=int, default = 1)

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')

    args = parser.parse_args()
    main(args)
