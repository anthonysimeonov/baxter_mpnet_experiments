#!/usr/bin/python

import numpy as np
from numpy import matlib
import rospy
import baxter_interface
from moveit_msgs.msg import RobotState, DisplayRobotState, PlanningScene, RobotTrajectory, ObjectColor
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander, MoveItCommanderException
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point
from std_msgs.msg import Header
import random
import sys
import tf
import os

from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates

from util.get_state_validity import StateValidity
from util.moveit_functions import create_scene_obs, moveit_scrambler, moveit_unscrambler, set_environment

import pickle
import subprocess
import time

class GazeboSceneModifier():
    def __init__(self, obstacles, port=1):
        # super(ShelfSceneModifier, self).__init__(files)

        # self._root_path = '/home/anthony/.gazebo/models/'
        self._root_path = './models/'
        self._obstacle_files = {}
        self._obstacles = obstacles

        self.port = port

        self.import_filenames(obstacles)

    def import_filenames(self, obstacles):
        for name in self._obstacles.keys():
            full_path = self._root_path + name + '/model.sdf'
            self._obstacle_files[name] = full_path

    # def cloud_pub(self):
        #todo

    def spawn_service_call(self, model, file, pose, z_offset):
        orientation = list(tf.transformations.quaternion_from_euler(0, 0, 0))
        orient_fixed = Quaternion(
            orientation[0], orientation[1], orientation[2], orientation[3])
        init_pose = Pose(Point(
            x=(pose[0]-1),
            y=pose[1],
            z=pose[2] + z_offset), orient_fixed)  # hacky fix for x value to be correct... TODO

        f = open(file)
        sdf_f = f.read()

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        try:
          # expects model name, model file, robot namespace, initial pose, and reference frame
          resp1 = spawn_model(model, sdf_f, "", init_pose, "world")
        except rospy.ServiceException as exc:
          print("Service did not process request (Failed to spawn model): " + str(exc))

    def permute_obstacles(self, pose_dict):
        for name in pose_dict.keys():
            file = self._obstacle_files[name]
            self.spawn_service_call(
                name, file, pose_dict[name], self._obstacles[name]['z_offset'])

    def delete_obstacles(self):
        for name in self._obstacles.keys():
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model = rospy.ServiceProxy(
                '/gazebo/delete_model', DeleteModel)
            try:
              resp1 = delete_model(name)
            except rospy.ServiceException as exc:
              print(
                  "Service did not process request (Failed to delete model): " + str(exc))


def create_call(ex, root, name):
    input = '/voxel_grid/output'
    # input = '/camera/depth/points'
    call = [str(ex), "input:="+input, "_prefix:=" + str(root) + "/" + str(name) + "_", "_compressed:=true"]
    return call

def main():
    if (len(sys.argv) > 1):
        start = int(sys.argv[1])
        if (len(sys.argv) > 2):
            end = int(sys.argv[2])
        else:
            end = -1
    else:
      start = 0
      end = 10

    print("start ind: " + str(start))
    print("end ind: " + str(end))

    rospy.init_node("testing_gazebo")

    rospy.wait_for_service('/gazebo/delete_model')
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
    try:
      resp1 = delete_model('ground_plane')
    except rospy.ServiceException as exc:
      print("Service did not process request (Failed to delete ground plane): " + str(exc))

    # executable = '/home/anthony/catkin_workspaces/baxter_ws/point_cloud_data/pointcloud_to_pcd'
    executable = './pointcloud_to_pcd'
    rootPath = 'data/train/pcd/'

    # load data from environment files for obstacle locations and collision free goal poses
    envs = os.listdir('env/environment_data')
    for env in envs:
        with open(env, "rb") as env_f:
            masterPoseDict = pickle.load(env_f)

        gazeboMod = GazeboSceneModifier(masterPoseDict['obsData'])
        raw_keys = masterPoseDict['poses'].keys()
        if len(raw_keys) == 0:
            continue
        sorted_keys = sorted(raw_keys, key=lambda x: int(x[9:])) # 9 because 'trainEnv_' is 9 characters

        for i, pose_name in enumerate(sorted_keys):
            print("iter number: " + str(i) + " \n\n\n")
            gazeboMod.delete_obstacles()
            print("POSE: " + str(pose_name))
            new_pose = masterPoseDict['poses'][pose_name]
            gazeboMod.permute_obstacles(new_pose)
            print("Loaded new pose and permuted obstacles")

            call = create_call(executable, rootPath, pose_name)

            ### Printing the executable call and allowing user to manually cycle through environments for demonstration
            print(call)
            #raw_input("press enter to continue\n")

            ### Uncomment below to call pointcloud_to_pcd executable which takes snapshot of the streaming pointcloud data
            ### and saves it to a .pcd file in a desired file location (as specified by prefix in the command call)

            # print("Calling executable... \n\n\n")
            # t = time.time()
            # p = subprocess.Popen(call)
            # rospy.sleep(0.8)
            # p.terminate()
            # p.wait()

            rospy.sleep(0.1)
            gazeboMod.delete_obstacles()
            print("Deleted obstacles \n\n\n")

if __name__ == '__main__':
    main()
