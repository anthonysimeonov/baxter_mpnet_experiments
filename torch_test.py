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

from torch_neuralplanner_functions import *
from torch_architectures import *
from tools.planning_scene_editor import *
from tools.get_state_validity import StateValidity

joint_ranges = np.array([3.4033, 3.194, 6.117, 3.6647, 6.117, 6.1083, 2.67])


def IsInCollision(state, print_depth=False):
    # returns true if robot state is in collision, false if robot state is collision free
    filler_robot_state[10:17] = moveit_scrambler(np.multiply(state,joint_ranges))
    rs_man.joint_state.position = tuple(filler_robot_state)

    col_start = time.clock()
    collision_free = sv.getStateValidity(rs_man, group_name="right_arm", print_depth=print_depth)
    col_end = time.clock()
    col_time = col_end - col_start

    global counter
    global col_time_env

    counter += 1
    col_time_env.append(col_time)
    return (not collision_free)

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.device)

    importer = fileImport()
    env_data_path = args.env_data_path
    path_data_path = args.path_data_path
    pcd_data_path = args.pointcloud_data_path

    envs = importer.environments_import(env_data_path + args.envs_file)
    envs = envs[:args.N]
    with open (env_data_path+args.envs_file, 'rb') as env_f:
        envDict = pickle.load(env_f)

    obstacles, paths, path_lengths = load_test_dataset_end2end(envs, path_data_path, pcd_data_path, args.path_data_file, importer, NP=args.NP)

    if args.AE_type == 'linear':
        encoder = Encoder(args.enc_input_size, args.enc_output_size)
    elif args.AE_type == 'voxel':
        encoder = VoxelEncoder(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel2':
        encoder = VoxelEncoder2(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel3':
        encoder = VoxelEncoder3(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)
    elif args.AE_type == 'voxel4':
        encoder = VoxelEncoder3(args.enc_input_size, args.enc_output_size)
        # convert obstacles to voxel
        obstacles = obstacles.astype(float).reshape(len(obstacles),-1,3)
        obstacles = importer.pointcloud_to_voxel(obstacles).reshape(len(obstacles),1,args.enc_input_size,args.enc_input_size,args.enc_input_size)

    mlp = MLP(args.mlp_input_size, args.mlp_output_size)

    # device = torch.device('cpu')
    model_path = args.model_path
    mlp.load_state_dict(torch.load(model_path+args.mlp_model_name)) #, map_location=device))
    encoder.load_state_dict(torch.load(model_path+args.enc_model_name)) #, map_location=device))
    # print Parameters
    for name, param in encoder.named_parameters():
        print(name)
        print(param)
    if torch.cuda.is_available():
        encoder.cuda()
        mlp.cuda()

    rospy.init_node("environment_monitor")
    scene = PlanningSceneInterface()
    robot = RobotCommander()
    group = MoveGroupCommander("right_arm")
    scene._scene_pub = rospy.Publisher('planning_scene',
                                        PlanningScene,
                                        queue_size=0)

    global sv
    global filler_robot_state
    global rs_man

    sv = StateValidity()
    set_environment(robot, scene)

    masterModifier = ShelfSceneModifier()
    sceneModifier = PlanningSceneModifier(envDict['obsData'])
    sceneModifier.setup_scene(scene, robot, group)

    rs_man = RobotState()
    robot_state = robot.get_current_state()
    rs_man.joint_state.name = robot_state.joint_state.name
    filler_robot_state = list(robot_state.joint_state.position)

    dof=7

    tp=0
    fp=0

    et_tot = []
    neural_paths = {}
    bad_paths = {}

    goal_collision = []

    if not os.path.exists(args.good_path_sample_path):
        os.makedirs(args.good_path_sample_path)
    if not os.path.exists(args.bad_path_sample_path):
        os.makedirs(args.bad_path_sample_path)


    # experiment_name = args.model_path.split('models/')[1] + "test_"
    experiment_name = args.model_path.split('models/')[1] + args.experiment_name
    good_paths_path = args.good_path_sample_path + '/' + experiment_name
    bad_paths_path = args.bad_path_sample_path + '/' + experiment_name

    global counter
    global col_time
    global col_time_env

    for i, env_name in enumerate(envs):
        et=[]
        col_env = []
        tp_env = 0
        fp_env = 0
        neural_paths[env_name] = []
        bad_paths[env_name] = []

        if not os.path.exists(good_paths_path + '/' + env_name):
            os.makedirs(good_paths_path + '/' + env_name)
        if not os.path.exists(bad_paths_path + '/' + env_name):
            os.makedirs(bad_paths_path + '/' + env_name)

        global col_time_env
        col_time_env = []


        print("ENVIRONMENT: " + env_name)

        sceneModifier.delete_obstacles()
        new_pose = envDict['poses'][env_name]
        sceneModifier.permute_obstacles(new_pose)

        for j in range(0,path_lengths.shape[1]):
            print ("step: i="+str(i)+" j="+str(j))
            print("fp: " + str(fp_env))
            print("tp: " + str(tp_env))
            p1_ind=0
            p2_ind=0
            p_ind=0

            obs=np.array([obstacles[i]])
            obs=torch.from_numpy(obs).type(torch.FloatTensor)

            en_inp=to_var(obs)
            h=encoder(en_inp)[0]

            if path_lengths[i][j]>0:
                global counter
                global col_time
                counter = 0

                if (j%10 == 0):
                    print("running average collision time: ")
                    print(np.mean(col_time_env))

                print(path_lengths[i][j])
                start=np.zeros(dof,dtype=np.float32)
                goal=np.zeros(dof,dtype=np.float32)
                for l in range(0,dof):
                    start[l]=paths[i][j][0][l]

                for l in range(0,dof):
                    goal[l]=paths[i][j][path_lengths[i][j]-1][l]

                if (IsInCollision(goal)):
                    print("GOAL IN COLLISION --- BREAKING")
                    goal_collision.append(j)
                    continue

                start1=torch.from_numpy(start)
                goal2=torch.from_numpy(start)
                goal1=torch.from_numpy(goal)
                start2=torch.from_numpy(goal)

                ##generated paths
                path1=[]
                path1.append(start1)
                path2=[]
                path2.append(start2)
                path=[]
                target_reached=0
                step=0
                path=[] # stores end2end path by concatenating path1 and path2
                tree=0
                tic_start = time.clock()
                step_sz = DEFAULT_STEP


                while target_reached==0 and step<3000:
                    step=step+1

                    if tree==0:
                        inp1=torch.cat((start1,start2,h.data.cpu()))
                        inp1=to_var(inp1)
                        start1=mlp(inp1)
                        start1=start1.data.cpu()
                        path1.append(start1)
                        tree=1
                    else:
                        inp2=torch.cat((start2,start1,h.data.cpu()))
                        inp2=to_var(inp2)
                        start2=mlp(inp2)
                        start2=start2.data.cpu()
                        path2.append(start2)
                        tree=0
                    target_reached=steerTo(start1,start2, IsInCollision)

                tp=tp+1
                tp_env=tp_env+1
                print('ground truth path:')
                print(paths[i][j])
                print('path1:')
                print(path1)
                print('path2:')
                print(path2)
                if (step > 3000 or not target_reached):
                    save_feasible_path(path, bad_paths_path + '/' + env_name + '/bp_' + str(j))

                if target_reached==1:
                    for p1 in range(0,len(path1)):
                        path.append(path1[p1])
                    for p2 in range(len(path2)-1,-1,-1):
                        path.append(path2[p2])

                    path = lvc(path, IsInCollision, step_sz=step_sz)

                    # full dense collision check
                    indicator=feasibility_check(path, IsInCollision, step_sz=0.01, print_depth=True)

                    if indicator==1:
                        toc = time.clock()

                        t=toc-tic_start
                        et.append(t)
                        col_env.append(counter)
                        fp=fp+1
                        fp_env=fp_env+1
                        neural_paths[env_name].append(path)
                        save_feasible_path(path, good_paths_path + '/' + env_name + '/fp_' + str(j))
                        print("---path found---")
                        print("length: " + str(len(path)))
                        print("time: " + str(t))
                        print("count: " + str(counter))
                    else:
                        sp=0
                        indicator=0
                        step_sz = DEFAULT_STEP
                        while indicator==0 and sp<10 and path !=0:

                            # adaptive step size on replanning attempts
                            if (sp == 1):
                                step_sz = 0.04
                            elif (sp == 2):
                                step_sz = 0.03
                            elif (sp > 2):
                                step_sz = 0.02

                            sp=sp+1
                            g=np.zeros(dof,dtype=np.float32)
                            g=torch.from_numpy(paths[i][j][path_lengths[i][j]-1])

                            tic = time.clock()
                            path=replan_path(path, g, mlp, IsInCollision, obs=h, step_sz=step_sz) #replanning at coarse level
                            toc = time.clock()

                            if path !=0:
                                path=lvc(path, IsInCollision, step_sz=step_sz)

                                # full dense collision check
                                indicator=feasibility_check(path, IsInCollision, step_sz=0.01,print_depth=True)

                                if indicator==1:
                                    toc = time.clock()

                                    t=toc-tic_start
                                    et.append(t)
                                    col_env.append(counter)
                                    fp=fp+1
                                    fp_env=fp_env+1
                                    neural_paths[env_name].append(path)
                                    save_feasible_path(path, good_paths_path + '/' + env_name + '/fp_' + str(j))

                                    print("---path found---")
                                    print("length: " + str(len(path)))
                                    print("time: " + str(t))
                                    print("count: " + str(counter))

                        if (sp == 10):
                            save_feasible_path(path, bad_paths_path + '/' + env_name + '/bp_' + str(j))

        et_tot.append(et)

        print("total env paths: ")
        print(tp_env)
        print("feasible env paths: ")
        print(fp_env)
        print("average collision checks: ")
        print(np.mean(col_env))
        print("average time per collision check: ")
        print(np.mean(col_time_env))
        print("average time: ")
        print(np.mean(et))
        env_data = {}
        env_data['tp_env'] = tp_env
        env_data['fp_env'] = fp_env
        env_data['et_env'] = et
        env_data['col_env'] = col_env
        env_data['avg_col_time'] = np.mean(col_time_env)
        env_data['paths'] = neural_paths[env_name]

        with open(good_paths_path + '/' + env_name + '/env_data.pkl', 'wb') as data_f:
            pickle.dump(env_data, data_f)

    print("total paths: ")
    print(tp)
    print("feasible paths: ")
    print(fp)

    with open(good_paths_path+'neural_paths.pkl', 'wb') as good_f:
        pickle.dump(neural_paths, good_f)

    with open(good_paths_path+'elapsed_time.pkl', 'wb') as time_f:
        pickle.dump(et_tot, time_f)

    print(np.mean([np.mean(x) for x in et_tot]))
    print(np.std([np.mean(x) for x in et_tot]))

    acc = []

    for i, env in enumerate(envs):
        with open (good_paths_path+env+'_env_data.pkl', 'rb') as data_f:
            data = pickle.load(data_f)
        acc.append(100.0*data['fp_env']/data['tp_env'])
        print("env: " + env)
        print("accuracy: " + str(100.0*data['fp_env']/data['tp_env']))
        print("time: " + str(np.mean(data['et_env'])))
        print("min time: " + str(np.min(data['et_env'])))
        print("max time: " + str(np.max(data['et_env'])))
        print("\n")

    print(np.mean(acc))
    print(np.std(acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/sample/')
    parser.add_argument('--mlp_model_name', type=str, default='mlp_PReLU_ae_dd140.pkl')
    parser.add_argument('--enc_model_name', type=str, default='cae_encoder_140.pkl')

    parser.add_argument('--good_path_sample_path', type=str, default='./path_samples/good_path_samples')
    parser.add_argument('--bad_path_sample_path', type=str, default='./path_samples/bad_path_samples')
    parser.add_argument('--experiment_name', type=str, default='test_experiment')

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/test/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/test/pcd/')
    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainEnvironments_testPaths.pkl')
    parser.add_argument('--AE_type', type=str, default='linear')
    parser.add_argument('--exp_name', type=str, default='linear')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--NP', type=int, default=100)

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    main(args)
