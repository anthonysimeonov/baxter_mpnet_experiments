import os.path as osp
import os
import sys
sys.path.append('../../')
import argparse
import numpy as np
import pickle
from tools.path_data_loader import load_test_dataset_end2end
import math
from tools.import_tool import fileImport
import time
import rospy

from architectures.mlp_pipeline import mlp_pipeline

from architectures.AE.point_net_ae import PointNetAutoEncoder

from architectures.AE.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from architectures.AE.tf_utils import reset_tf_graph
from architectures.AE.general_utils import plot_3d_point_cloud

from architectures.mpnet import Configuration as Conf
from architectures.mpnet import MPNet

from architectures.mlp_models import baxter_mpnet_mlp
from architectures.AE.ae_templates import mlp_architecture_ala_iclr_18, default_train_params

from neuralplanner_functions import *

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

    # define basic parameters
    top_out_dir = args.trained_model_path  # Use to save Neural-Net check-points etc.
    n_pc_points = args.enc_input_size // 3 # Number of points per model.
    bneck_size = args.enc_output_size      # Bottleneck-AE size
    ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'

    envs = importer.environments_import(env_data_path + args.envs_file)
    with open (env_data_path+args.envs_file, 'rb') as env_f:
        envDict = pickle.load(env_f)

    obstacles, paths, path_lengths = load_test_dataset_end2end(envs, path_data_path, pcd_data_path, args.path_data_file, importer, NP=100)

    if not os.path.exists(args.trained_model_path):
        os.makedirs(args.trained_model_path)

    # write encoder and decoder pipeline
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
    # write mpnet pipeline
    mlp, mlp_args = baxter_mpnet_mlp(args.mlp_input_size, args.mlp_output_size)

    train_dir = create_dir(osp.join(top_out_dir, args.experiment_name))

    conf = Conf(loss = ae_loss,
            experiment_name = args.experiment_name,
            training_epochs = args.num_epochs,
            batch_size = args.batch_size,
            denoising = False,
            ae_learning_rate = args.ae_learning_rate,
            mlp_learning_rate = args.mlp_learning_rate,
            train_dir = train_dir,
            loss_display_step = args.loss_display_step,
            saver_step = args.saver_step,
            z_rotate =args.z_rotate,
            encoder = encoder,
            decoder = decoder,
            mlp = mlp,
            encoder_args = enc_args,
            decoder_args = dec_args,
            mlp_args = mlp_args,
            n_o_input = [n_pc_points, 3],
            n_x_input = [args.mlp_input_size-args.enc_output_size],
            n_output = [args.mlp_output_size],
            pretrain = args.pretrain,
            pretrain_epochs = args.pretrain_epochs,
            pretrain_batch_size = args.pretrain_batch_size,
            fixAE = args.fixAE
           )
    conf.held_out_step = 5   # How often to evaluate/print out loss on
                             # held_out data (if they are provided in ae.train() ).
    #conf.save(osp.join(train_dir, 'configuration'))
    # reload
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    mpnet = MPNet(conf.experiment_name, conf)
    if args.AE_restore_pretrain:
        # filename is pretrain_ae
        mpnet.restore_model(mpnet.AE_saver, conf.train_dir, epoch=args.AE_start_epoch, filename='pretrain_ae')
    else:
        mpnet.restore_model(mpnet.AE_saver, conf.train_dir, epoch=args.AE_start_epoch, filename='ae')
    if args.start_epoch > 0:
        mpnet.restore_model(mpnet.mlp_saver, conf.train_dir, epoch=args.start_epoch, filename='mlp')

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
    experiment_name = args.trained_model_path.split('models/')[1] + args.experiment_name
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

            obs=obstacles[i]
            obs=obs.reshape(1, -1, 3)

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

                start1=start
                goal2=start
                goal1=goal
                start2=goal

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
                        inp1=np.concatenate((start1,start2))
                        start1=mpnet.plan(obs,np.array([inp1]))
                        path1.append(start1)
                        tree=1
                    else:
                        inp2=np.concatenate((start2,start1))
                        start2=mpnet.plan(obs,np.array([inp2]))
                        path2.append(start2)
                        tree=0
                    target_reached=steerTo(start1,start2, IsInCollision)

                tp=tp+1
                tp_env=tp_env+1

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
                            g=np.array(paths[i][j][path_lengths[i][j]-1])
                            tic = time.clock()
                            path=replan_path(path, g, mpnet, IsInCollision, obs=h, step_sz=step_sz) #replanning at coarse level
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
        with open (good_paths_path+env+'/env_data.pkl', 'rb') as data_f:
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
    parser.add_argument('--good_path_sample_path', type=str, default='./path_samples/good_path_samples')
    parser.add_argument('--bad_path_sample_path', type=str, default='./path_samples/bad_path_samples')

    parser.add_argument('--env_data_path', type=str, default='./env/environment_data/')
    parser.add_argument('--path_data_path', type=str, default='./data/test/paths/')
    parser.add_argument('--pointcloud_data_path', type=str, default='./data/test/pcd/')
    parser.add_argument('--envs_file', type=str, default='trainEnvironments.pkl')
    parser.add_argument('--path_data_file', type=str, default='trainEnvironments_testPaths.pkl')

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_input_size', type=int, default=74)
    parser.add_argument('--mlp_output_size', type=int, default=7)

    parser.add_argument('--ae_learning_rate', type=float, default=0.001)
    parser.add_argument('--mlp_learning_rate', type=float, default=0.001)
    parser.add_argument('--experiment_name', type=str, default='test_experiment')
    parser.add_argument('--trained_model_path', type=str, default='./models/sample_train/', help='path for saving trained models')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--AE_start_epoch', type=int, default=0)
    parser.add_argument('--AE_restore_pretrain', type=int, default=0, help='indicate if AutoEncoder is going to restore the pretrained model or the end2end model')

    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--pretrain', type=int, default=0, help='indicate if pretraining AutoEncoder or not')
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--pretrain_batch_size', type=int, default=100)
    parser.add_argument('--fixAE', type=int, default=0, help='fix AutoEncoder or not when training with MLP')

    parser.add_argument('--loss_display_step', type=int, default=1)
    parser.add_argument('--saver_step', type=int, default=10)
    parser.add_argument('--z_rotate', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    main(args)
