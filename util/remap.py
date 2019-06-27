#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState

rospy.init_node("remapper")

joint_states = rospy.Publisher("/joint_states", JointState, queue_size = 10)

def remap(data):
    if not rospy.is_shutdown():
        joint_states.publish(data)

rospy.Subscriber("/robot/joint_states", JointState, remap)

rospy.spin()
