import numpy as np
from numpy import matlib
import rospy
import baxter_interface
from moveit_msgs.msg import RobotState, DisplayRobotState, PlanningScene, RobotTrajectory, ObjectColor
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander, MoveItCommanderException
from sensor_msgs.msg import JointState
from get_state_validity import StateValidity
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point
from std_msgs.msg import Header
import random
import time
# import matplotlib.pyplot as plt
import sys
from gazebo_msgs.srv import SpawnModel
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.msg import ModelState, ModelStates
import tf
import os
import pickle

_colors = {}  # fix this later


def create_scene_box(name, dimension, is_mesh, mesh_file, orientation, z_offset):
    """
    Function to take in the name and dimensions of a box to be added into a PlanningScene object for MoveIt

    Inputs:
        name (string): name of box, example: "box_1"
        dimension (list): dimensions of box in [length, width, height], example: [2, 3, 4]

    Outputs:
        box_dict (dictionary): dictionary representation of the box with fields [name, dim]
    """
    box_dict = {}
    box_dict['name'] = name  # string
    box_dict['dim'] = dimension
    box_dict['is_mesh'] = is_mesh
    box_dict['mesh_file'] = mesh_file
    box_dict['orientation'] = orientation
    box_dict['z_offset'] = z_offset
    return box_dict


def create_scene_mesh(name, size):
    """
    Function to take in the name and scale of a mesh to be added into a PlanningScene object for MoveIt

    Inputs:
        name (string): name of mesh, example: "mesh_1"
        dimension (list): scale of mesh in [x, y, z], example: [2, 3, 4]

    Outputs:
        mesh_dict (dictionary): dictionary representation of the mesh with fields [name, size]
    """
    mesh_dict = {}
    mesh_dict['name'] = name
    mesh_dict['size'] = size
    return mesh_dict

def moveit_scrambler(positions):
    new_positions = list(positions)
    new_positions[2:4] = positions[5:7]
    new_positions[4:7] = positions[2:5]
    return new_positions


def moveit_unscrambler(positions):
    new_positions = list(positions)
    new_positions[2:5] = positions[4:]
    new_positions[5:] = positions[2:4]
    return new_positions

def setColor(name, r, g, b, a):
    # Create our color
    color = ObjectColor()
    color.id = name
    color.color.r = r
    color.color.g = g
    color.color.b = b
    color.color.a = a
    _colors[name] = color

## @brief Actually send the colors to MoveIt!


def sendColors(scene):
    # Need to send a planning scene diff
    # print(_colors['table_center'])
    p = PlanningScene()
    p.is_diff = True
    for color in _colors.values():
        p.object_colors.append(color)
    # print(p)
    scene._scene_pub.publish(p)


def color_norm(col):
    """ takes in a list of colours in 0-255 """

    norm_col = [x/255.0 for x in col]
    return norm_col


def set_environment(bot, scene):
    # centre table
    p1 = PoseStamped()
    p1.header.frame_id = bot.get_planning_frame()
    p1.pose.position.x = 1.  # position of the table in the world frame
    p1.pose.position.y = -0.3
    p1.pose.position.z = 0.

    # left table (from baxter's perspective)
    p1_l = PoseStamped()
    p1_l.header.frame_id = bot.get_planning_frame()
    p1_l.pose.position.x = 0.5  # position of the table in the world frame
    p1_l.pose.position.y = 1
    p1_l.pose.position.z = 0.
    p1_l.pose.orientation.x = 0.707
    p1_l.pose.orientation.y = 0.707

    # right table (from baxter's perspective)
    p1_r = PoseStamped()
    p1_r.header.frame_id = bot.get_planning_frame()
    p1_r.pose.position.x = 0.5  # position of the table in the world frame
    p1_r.pose.position.y = -1
    p1_r.pose.position.z = 0.
    p1_r.pose.orientation.x = 0.707
    p1_r.pose.orientation.y = 0.707

    # open back shelf
    p2 = PoseStamped()
    p2.header.frame_id = bot.get_planning_frame()
    p2.pose.position.x = 1.2  # position of the table in the world frame
    p2.pose.position.y = 0.0
    p2.pose.position.z = 0.75
    p2.pose.orientation.x = 0.5
    p2.pose.orientation.y = -0.5
    p2.pose.orientation.z = -0.5
    p2.pose.orientation.w = 0.5

    pw = PoseStamped()
    pw.header.frame_id = bot.get_planning_frame()

    # add an object to be grasped
    p_ob1 = PoseStamped()
    p_ob1.header.frame_id = bot.get_planning_frame()
    p_ob1.pose.position.x = .92
    p_ob1.pose.position.y = 0.3
    p_ob1.pose.position.z = 0.89

    # the ole duck
    p_ob2 = PoseStamped()
    p_ob2.header.frame_id = bot.get_planning_frame()
    p_ob2.pose.position.x = 0.87
    p_ob2.pose.position.y = -0.4
    p_ob2.pose.position.z = 0.24

    scene.remove_world_object("table_center")
    scene.remove_world_object("table_side_left")
    scene.remove_world_object("table_side_right")
    scene.remove_world_object("shelf")
    scene.remove_world_object("wall")
    scene.remove_world_object("part")
    scene.remove_world_object("duck")

    rospy.sleep(1)

    # dimensions of the table
    scene.add_box("table_center", p1, (.5, 1.1, 0.4))
    # rospy.sleep(.2)
    # scene.add_box("table_side_left", p1_l, (.5, 1.5, 0.4))
    # rospy.sleep(.2)
    scene.add_box("table_side_right", p1_r, (.5, 1.5, 0.4))
    # rospy.sleep(.2)
    # scene.add_mesh("shelf", p2, "./smallshelf.stl", size = (.1, .025, .025))  # closed back shelf

    # scene.add_mesh("shelf", p2, "./bookshelf_light.stl", size = (.022, .01, .01)) #commented out 1/3/19 because switching to just tabletop
    # rospy.sleep(.5)
    scene.add_plane("wall", pw, normal=(0, 1, 0))

    part_size = (0.07, 0.05, 0.12)
    # scene.add_box("part", p_ob1, size=part_size) #commented out 1/3/19 to just use our own randomly generated boxes
    # scene.add_mesh("duck", p_ob2, "./duck.stl", size = (.001, .001, .001))

    rospy.sleep(1)

    print(scene.get_known_object_names())

    ## ---> SET COLOURS

    table_color = color_norm([105, 105, 105])
    shelf_color = color_norm([139, 69, 19])
    duck_color = color_norm([255, 255, 0])
    # print(table_color)
    setColor('table_center', table_color[0], table_color[1], table_color[2], 1)
    setColor('table_side_left',
             table_color[0], table_color[1], table_color[2], 1)
    setColor('table_side_right',
             table_color[0], table_color[1], table_color[2], 1)
    setColor('shelf', shelf_color[0], shelf_color[1], shelf_color[2], 1)
    setColor('duck', duck_color[0], duck_color[1], duck_color[2], 1)
    # rospy.sleep(4)

    sendColors(scene)


class ShelfSceneModifier():
    def __init__(self, gazebo_path='../env/gazebo_models/'):
        #x, y boundary with some padding
        self.valid_pose_boundary_x = [[0.8, 0.95], [-0.2, 1.02]]
        self.valid_pose_boundary_y = [[-0.69, 0.0], [-1.05, -0.79]] #top shelf [xmin, xmax, ymin, ymax, z1 = 0.24, z2 = 0.78
        # self.valid_x_range = self.valid_pose_boundary_x[1] - self.valid_pose_boundary_x[0] #should use min/max
        # self.valid_y_range = self.valid_pose_boundary_y[1] - self.valid_pose_boundary_y[0]

        self.boundary_dict = {}
        # self.z = [0.24, 0.78] #only use for shelf, otherwise always same z
        self.z = 0.24
        self.z_range = 0.12

        self.obstacles = {}
        self.gazebo_path = gazebo_path

        self.setup_boundaries()
        self.setup_obstacles()

    def setup_boundaries(self):
        self.boundary_dict['x_bounds'] = self.valid_pose_boundary_x
        self.boundary_dict['y_bounds'] = self.valid_pose_boundary_y

        self.boundary_dict['x_range'] = []
        self.boundary_dict['y_range'] = []
        for area in range(len(self.valid_pose_boundary_x)):
            self.boundary_dict['x_range'].insert(area, max(self.valid_pose_boundary_x[area]) - min(self.valid_pose_boundary_x[area]))
            self.boundary_dict['y_range'].insert(area, max(self.valid_pose_boundary_y[area]) - min(self.valid_pose_boundary_y[area]))

    def setup_obstacles(self):
        # names = ['mpnet_box1', 'mpnet_box2', 'mpnet_box3', 'mpnet_box4', 'mpnet_box5'] #5 boxes
        # obstacles{}.keys() = ['name', 'dimensions', 'is_mesh', 'mesh_file', 'orientation', 'z_offset']
        obstacles = [
        ('mpnet_box1', [0.14, 0.09, 0.12], 0, None, None, 0.02),
        ('mpnet_bottle', [0.001, 0.001, 0.001], 1, self.gazebo_path+'mpnet_bottle/meshes/bottle.stl', None, -0.02),
        ('mpnet_pepsi', [0.001, 0.001, 0.001], 1, self.gazebo_path+'mpnet_pepsi/meshes/pepsi.STL', [0.70710678, 0, 0, 0.70710678], -0.041),
        ('mpnet_coffee', [0.035, 0.035, 0.035], 1, self.gazebo_path+'mpnet_coffee/meshes/coffee.stl', [0.70710678, 0, 0, 0.70710678], -0.035),
        ('mpnet_book', [0.0017, 0.0017, 0.0017], 1, self.gazebo_path+'mpnet_book/meshes/Dictionary.STL', None, -0.04)
        ] # (name(string), is_mesh(bool), mesh_file(str))
        # dimensions = [[0.14, 0.09, 0.12], [0.001, 0.001, 0.001], [0.001, 0.001, 0.001], [0.05, 0.05, 0.05], [0.001, 0.001, 0.001]]

        for i, obs in enumerate(obstacles):
            self.obstacles[obs[0]] = create_scene_box(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])

    def get_permutation(self):
        area = 0 if (np.random.random() < 0.5) else 1
        object_location_x = np.random.random() * (self.boundary_dict['x_range'][area]) + min(self.boundary_dict['x_bounds'][area])
        object_location_y = np.random.random() * (self.boundary_dict['y_range'][area]) + min(self.boundary_dict['y_bounds'][area])
        # object_location_z = self.z[0] if (np.random.random() < 0.5) else self.z[1] #removed when not using shelf
        object_location_z = self.z

        pose = [object_location_x, object_location_y, object_location_z]

        return pose

    def permute_obstacles(self):
        pose_dict = {}
        for name in self.obstacles.keys():
            # name = obstacle['name']
            pose_dict[name] = self.get_permutation()

        return pose_dict


class PlanningSceneModifier():
    def __init__(self, obstacles, port=0):
        self._obstacles = obstacles

        self.port = port

        self._scene = None
        self._robot = None

    def setup_scene(self, scene, robot, group):
        self._scene = scene
        self._robot = robot
        self._group = group

    def permute_obstacles(self, pose_dict):
        for name in pose_dict.keys():
            pose = PoseStamped()
            pose.header.frame_id = self._robot.get_planning_frame()
            pose.pose.position.x = pose_dict[name][0]
            pose.pose.position.y = pose_dict[name][1]
            pose.pose.position.z = pose_dict[name][2] + self._obstacles[name]['z_offset']

            if self._obstacles[name]['orientation'] is not None:
                pose.pose.orientation.x = self._obstacles[name]['orientation'][0]
                pose.pose.orientation.y = self._obstacles[name]['orientation'][1]
                pose.pose.orientation.z = self._obstacles[name]['orientation'][2]
                pose.pose.orientation.w = self._obstacles[name]['orientation'][3]

            if self._obstacles[name]['is_mesh']:
                print(self._obstacles[name]['mesh_file'])
                self._scene.add_mesh(name, pose, filename=self._obstacles[name]['mesh_file'], size=self._obstacles[name]['dim'])
            else:
                self._scene.add_box(name, pose, size=self._obstacles[name]['dim'])

        rospy.sleep(1)
        print(self._scene.get_known_object_names())

    def delete_obstacles(self):
        #scene.remove_world_object("table_center")
        for name in self._obstacles.keys():
            self._scene.remove_world_object(name)

class GazeboSceneModifier():
    def __init__(self, obstacles, port=1):
        # super(ShelfSceneModifier, self).__init__(files)
        self._root_path = '$HOME/.gazebo/models/'
        self._obstacle_files = {}
        self._obstacles = obstacles

        self.port = port

        self.import_filenames(obstacles)

    def import_filenames(self, obstacles):
        for name in self._obstacles.keys():
            full_path = self._root_path + name + '/model.sdf'
            self._obstacle_files[name] = full_path

    def spawn_service_call(self, model, file, pose, z_offset):
        orientation = list(tf.transformations.quaternion_from_euler(0, 0, 0))
        orient_fixed = Quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        init_pose = Pose(Point(
        x=(pose[0]-1),
        y=pose[1],
        z=pose[2] + z_offset), orient_fixed) #hacky fix for x value to be correct... TODO

        f = open(file)
        sdf_f = f.read()

        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        try:
          resp1 = spawn_model(model, sdf_f, "", init_pose, "world") #expects model name, model file, robot namespace, initial pose, and reference frame
        except rospy.ServiceException as exc:
          print("Service did not process request (Failed to spawn model): " + str(exc))

    def permute_obstacles(self, pose_dict):
        for name in pose_dict.keys():
            file = self._obstacle_files[name]
            self.spawn_service_call(name, file, pose_dict[name], self._obstacles[name]['z_offset'])

    def delete_obstacles(self):
        for name in self._obstacles.keys():
            rospy.wait_for_service('/gazebo/delete_model')
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            try:
              resp1 = delete_model(name)
            except rospy.ServiceException as exc:
              print("Service did not process request (Failed to delete model): " + str(exc))
