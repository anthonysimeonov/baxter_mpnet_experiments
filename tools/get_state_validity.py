#!/usr/bin/python

import rospy
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse

DEFAULT_SV_SERVICE = "/check_state_validity"

class StateValidity():
    def __init__(self):
        rospy.loginfo("Initializing stateValidity class")
        self.sv_srv = rospy.ServiceProxy(DEFAULT_SV_SERVICE, GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")
        rospy.wait_for_service("check_state_validity")
        rospy.loginfo("Reached this point")

        if rospy.has_param('/play_motion/approach_planner/planning_groups'):
            list_planning_groups = rospy.get_param('/play_motion/approach_planner/planning_groups')
        else:
            rospy.logwarn("Param '/play_motion/approach_planner/planning_groups' not set. We can't guess controllers")
        rospy.loginfo("Ready for making Validity calls")


    def close_SV(self):
        self.sv_srv.close()


    def getStateValidity(self, robot_state, group_name='both_arms_torso', constraints=None, print_depth=False):
        """Given a RobotState and a group name and an optional Constraints
        return the validity of the State"""
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = group_name
        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)

        if (not result.valid):
            contact_depths = []
            for i in range(len(result.contacts)):
                contact_depths.append(result.contacts[i].depth)

            max_depth = max(contact_depths)
            if max_depth < 0.0001:
                return True
            else:
                return False 
    
        return result.valid
