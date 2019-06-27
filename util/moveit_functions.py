import numpy as np
from numpy import matlib
import rospy
import baxter_interface
from moveit_msgs.msg import RobotState, DisplayRobotState, PlanningScene, RobotTrajectory, ObjectColor
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from sensor_msgs.msg import JointState
from get_state_validity import StateValidity
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point
from std_msgs.msg import Header
import random
import time
import sys

_colors = {} #fix this later

def create_scene_obs(name, dimension, is_mesh, mesh_file, orientation, z_offset):
    """
    Function to take in the name and dimensions of a box to be added into a PlanningScene object for MoveIt

    Inputs:
        name (string): name of box, example: "box_1"
        dimension (list): dimensions of box in [length, width, height], example: [2, 3, 4]

    Outputs:
        obs_dict (dictionary): dictionary representation of the box with fields [name, dim]
    """
    obs_dict = {}
    obs_dict['name'] = name #string
    obs_dict['dim'] = dimension
    obs_dict['is_mesh'] = is_mesh
    obs_dict['mesh_file'] = mesh_file
    obs_dict['orientation'] = orientation
    obs_dict['z_offset'] = z_offset
    return obs_dict

def sample_loc(sample_points):
    joint = {}
    # ref hardware specs for baxter joint limits
    joint['right_s0'] = float(np.random.uniform(-1.7016, 1.7016, sample_points)) # s0
    joint['right_s1'] = float(np.random.uniform(-2.147, 1.047, sample_points))  # s1
    joint['right_e0'] = float(np.random.uniform(-3.0541, 3.0541, sample_points)) # e0
    joint['right_e1'] = float(np.random.uniform(-0.05, 2.618, sample_points)) # e1
    joint['right_w0'] = float(np.random.uniform(-3.059, 3.059, sample_points)) # w0
    joint['right_w1'] = float(np.random.uniform(-1.5707, 2.094, sample_points)) # w1
    joint['right_w2'] = float(np.random.uniform(-3.059, 3.059, sample_points)) # w2
    return joint

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

def collision_flag(waypoint, rs_man, sv):
    collision = sv.getStateValidity(rs_man, group_name='right_arm')# and sv.getStateValidity(rs_man, group_name='right_hand')
    return collision

###############################################################
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
    p1.pose.position.x = 1. # position of the table in the world frame
    p1.pose.position.y = -0.3
    p1.pose.position.z = 0.

    # left table (from baxter's perspective)
    p1_l = PoseStamped()
    p1_l.header.frame_id = bot.get_planning_frame()
    p1_l.pose.position.x = 0.5 # position of the table in the world frame
    p1_l.pose.position.y = 1
    p1_l.pose.position.z = 0.
    p1_l.pose.orientation.x = 0.707
    p1_l.pose.orientation.y = 0.707

    # right table (from baxter's perspective)
    p1_r = PoseStamped()
    p1_r.header.frame_id = bot.get_planning_frame()
    p1_r.pose.position.x = 0.5 # position of the table in the world frame
    p1_r.pose.position.y = -1
    p1_r.pose.position.z = 0.
    p1_r.pose.orientation.x = 0.707
    p1_r.pose.orientation.y = 0.707

    # open back shelf
    p2 = PoseStamped()
    p2.header.frame_id = bot.get_planning_frame()
    p2.pose.position.x = 1.2 # position of the table in the world frame
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

    print("Adding center table...")
    scene.add_box("table_center", p1, (.5, 1.1, 0.4)) # dimensions of the table
    # rospy.sleep(.2)
    # scene.add_box("table_side_left", p1_l, (.5, 1.5, 0.4))
    # rospy.sleep(.2)
    print("Adding right table...")
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
    setColor('table_side_left', table_color[0], table_color[1], table_color[2], 1)
    setColor('table_side_right', table_color[0], table_color[1], table_color[2], 1)
    setColor('shelf', shelf_color[0], shelf_color[1], shelf_color[2], 1)
    setColor('duck', duck_color[0], duck_color[1], duck_color[2], 1)
    # rospy.sleep(4)

    sendColors(scene)
