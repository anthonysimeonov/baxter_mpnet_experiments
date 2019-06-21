import numpy as np
import pickle
import rospy
import argparse
import baxter_interface

from tools.planning_scene_editor import *


def compute_cost(path):
    state_dists = []
    for i in range(len(path) - 1):
        dist = 0
        for j in range(7):  # baxter DOF
            diff = path[i][j] - path[i+1][j]
            dist += diff*diff

        state_dists.append(np.sqrt(dist))
    total_cost = sum(state_dists)
    return total_cost


def main(args):
    rospy.init_node('path_data_gen')

    # sometimes takes a few tries to connect to robot arm
    limb_init = False
    while not limb_init:
        try:
            limb = baxter_interface.Limb('right')
            limb_init = True
        except OSError:
            limb_init = False

    neutral_start = limb.joint_angles()
    min_goal_cost_threshold = 2.0

    # Set up planning scene and Move Group objects
    scene = PlanningSceneInterface()
    scene._scene_pub = rospy.Publisher(
        'planning_scene', PlanningScene, queue_size=0)
    robot = RobotCommander()
    group = MoveGroupCommander("right_arm")
    sv = StateValidity()

    set_environment(robot, scene)

    # Setup tables (geometry and location defined in moveit_functions.py, set_environment function)
    # set_environment(robot, scene)

    # Additional Move group setup

    # group.set_goal_joint_tolerance(0.001)
    # group.set_max_velocity_scaling_factor(0.7)
    # group.set_max_acceleration_scaling_factor(0.1)
    max_time = args.max_time
    group.set_planning_time(max_time)

    # Dictionary to save path data, and filename to save the data to in the end
    pathsDict = {}
    # pathsFile = "data/train/path_data_example"
    pathsFile = args.path_data_path+args.path_data_file

    # load data from environment files for obstacle locations and collision free goal poses
    # with open("env/environment_data/trainEnvironments_GazeboPatch.pkl", "rb") as env_f:
    with open(args.env_data_path+args.env_data_file, "rb") as env_f:
        envDict = pickle.load(env_f)

    # with open("env/environment_data/trainEnvironments_testGoals.pkl", "rb") as goal_f:
    with open(args.targets_data_path+args.targets_data_file, "rb") as goal_f:
        goalDict = pickle.load(goal_f)

    # Obstacle data stored in environment dictionary, loaded into scene modifier to apply obstacle scenes to MoveIt
    sceneModifier = PlanningSceneModifier(envDict['obsData'])
    sceneModifier.setup_scene(scene, robot, group)

    robot_state = robot.get_current_state()
    rs_man = RobotState()
    rs_man.joint_state.name = robot_state.joint_state.name

    # Here we go
    # slice this if you want only some environments
    test_envs = envDict['poses'].keys()
    done = False
    iter_num = 0
    print("Testing envs: ")
    print(test_envs)

    while(not rospy.is_shutdown() and not done):
        for i_env, env_name in enumerate(test_envs):
            print("env iteration number: " + str(i_env))
            print("env name: " + str(env_name))

            sceneModifier.delete_obstacles()
            new_pose = envDict['poses'][env_name]
            sceneModifier.permute_obstacles(new_pose)
            print("Loaded new pose and permuted obstacles\n")

            pathsDict[env_name] = {}
            pathsDict[env_name]['paths'] = []
            pathsDict[env_name]['costs'] = []
            pathsDict[env_name]['times'] = []
            pathsDict[env_name]['total'] = 0
            pathsDict[env_name]['feasible'] = 0

            collision_free_goals = goalDict[env_name]['Joints']

            total_paths = 0
            feasible_paths = 0
            i_path = 0

            # run until either desired number of total or feasible paths has been found
            while (total_paths < args.paths_to_save):

                #do planning and save data

                # some invalid goal states found their way into the goal dataset, check to ensure goals are "reaching" poses above the table
                # by only taking goals which have a straight path cost above a threshold
                valid_goal = False
                while not valid_goal:
                    goal = collision_free_goals[np.random.randint(
                        0, len(collision_free_goals))]
                    optimal_path = [neutral_start.values(), goal.values()]
                    optimal_cost = compute_cost(optimal_path)

                    if optimal_cost > min_goal_cost_threshold:
                        valid_goal = True

                print("FP: " + str(feasible_paths))
                print("TP: " + str(total_paths))
                total_paths += 1
                i_path += 1

                # Uncomment below if using a start state different than the robot current state

                # filler_robot_state = list(robot_state.joint_state.position) #not sure if I need this stuff
                # filler_robot_state[10:17] = moveit_scrambler(start.values())
                # rs_man.joint_state.position = tuple(filler_robot_state)
                # group.set_start_state(rs_man)   # set start

                group.set_start_state_to_current_state()

                group.clear_pose_targets()
                try:
                    group.set_joint_value_target(
                        moveit_scrambler(goal.values()))  # set target
                except MoveItCommanderException as e:
                    print(e)
                    continue

                start_t = time.time()
                plan = group.plan()

                pos = [plan.joint_trajectory.points[i].positions for i in range(
                    len(plan.joint_trajectory.points))]
                if pos != []:
                    pos = np.asarray(pos)
                    cost = compute_cost(pos)
                    t = time.time() - start_t
                    print("Time: " + str(t))
                    print("Cost: " + str(cost))

                    # Uncomment below if using max time as criteria for failure
                    if (t > (max_time*0.99)):
                        print("Reached max time...")
                        continue

                    feasible_paths += 1

                    pathsDict[env_name]['paths'].append(pos)
                    pathsDict[env_name]['costs'].append(cost)
                    pathsDict[env_name]['times'].append(t)
                    pathsDict[env_name]['feasible'] = feasible_paths
                    pathsDict[env_name]['total'] = total_paths

                    # Uncomment below if you want to overwrite data on each new feasible path
                    with open(pathsFile + "_" + env_name + ".pkl", "wb") as path_f:
                        pickle.dump(pathsDict[env_name], path_f)

                print("\n")

            sceneModifier.delete_obstacles()
            iter_num += 1

            print("Env: " + str(env_name))
            print("Feasible Paths: " + str(feasible_paths))
            print("Total Paths: " + str(total_paths))
            print("\n")

            pathsDict[env_name]['total'] = total_paths
            pathsDict[env_name]['feasible'] = feasible_paths

            with open(pathsFile + "_" + env_name + ".pkl", "wb") as path_f:
                pickle.dump(pathsDict[env_name], path_f)

        print("Done iterating, saving all data and exiting...\n\n\n")

        with open(pathsFile + ".pkl", "wb") as path_f:
            pickle.dump(pathsDict, path_f)

        done = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--env_data_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--targets_data_path', type=str, default='./data/train/targets/')
    parser.add_argument('--targets_data_file', type=str, default='trainTargets.pkl')
    parser.add_argument('--path_data_path', type=str, default='./data/train/paths/')
    parser.add_argument('--path_data_file', type=str, default='path_data_sample')

    parser.add_argument('--paths_to_save', type=int, default=5)
    parser.add_argument('--max_time', type=int, default=5)

    args = parser.parse_args()
    main(args)
