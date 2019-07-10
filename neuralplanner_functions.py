import argparse
import numpy as np
import os
import pickle
# load_test_dataset_single_paths_only
import math

DEFAULT_STEP = 0.05

def steerTo(start, end, collHandle, step_sz=DEFAULT_STEP, print_depth=False, dof=7):

    DISCRETIZATION_STEP = step_sz
    dists = np.zeros(dof, dtype=np.float32)
    for i in range(0, dof):
        dists[i] = end[i] - start[i]

    distTotal = 0.0
    for i in range(0, dof):
        distTotal = distTotal + dists[i]*dists[i]

    distTotal = math.sqrt(distTotal)
    if distTotal > 0:
        incrementTotal = distTotal/DISCRETIZATION_STEP
        for i in range(0, dof):
            dists[i] = dists[i]/incrementTotal

        numSegments = int(math.floor(incrementTotal))

        stateCurr = np.zeros(7, dtype=np.float32)
        for i in range(0, dof):
            stateCurr[i] = start[i]
        for i in range(0, numSegments):

            if collHandle(stateCurr, print_depth=print_depth):
                return 0

            for j in range(0, dof):
                stateCurr[j] = stateCurr[j]+dists[j]

        if collHandle(end, print_depth=print_depth):
            return 0

    return 1

# checks the feasibility of entire path including the path edges


def feasibility_check(path, collHandle, step_sz=DEFAULT_STEP, print_depth=False):

    for i in range(0, len(path)-1):
        ind = steerTo(path[i], path[i+1], collHandle, step_sz=step_sz,
                      print_depth=print_depth)
        if ind == 0:
            return 0
    return 1


# checks the feasibility of path nodes only
def collision_check(path, collHandle):

    for i in range(0, len(path)):
        if collHandle(path[i]):
            return 0
    return 1



def is_reaching_target(start1, start2, dof=7):

    s1 = np.zeros(dof, dtype=np.float32)
    for i in range(dof):
        s1[i] = start1[i]

    s2 = np.zeros(dof, dtype=np.float32)
    for i in range(dof):
        s2[i] = start2[i]

    for i in range(0, dof):
        if abs(s1[i]-s2[i]) > 0.05:
            return False
    return True

#lazy vertex contraction


def lvc(path, collHandle, step_sz=DEFAULT_STEP):

    for i in range(0, len(path)-1):
        for j in range(len(path)-1, i+1, -1):
            ind = 0
            ind = steerTo(path[i], path[j], collHandle, step_sz=step_sz)
            if ind == 1:
                pc = []
                for k in range(0, i+1):
                    pc.append(path[k])
                for k in range(j, len(path)):
                    pc.append(path[k])

                return lvc(pc, collHandle)

    return path

def replan_path(p, g, mpnet, collHandle, obs, step_sz=DEFAULT_STEP):
    step = 0
    path = []
    path.append(p[0])
    for i in range(1, len(p)-1):
        if not collHandle(p[i]):
            path.append(p[i])
    path.append(g)
    new_path = []
    for i in range(0, len(path)-1):
        target_reached = False
        st = path[i]
        gl = path[i+1]
        steer = steerTo(st, gl, collHandle, step_sz=step_sz)
        if steer == 1:
            new_path.append(st)
            new_path.append(gl)
        else:
            itr = 0
            pA = []
            pA.append(st)
            pB = []
            pB.append(gl)
            target_reached = 0
            tree = 0
            while target_reached == 0 and itr < 3000:
                itr = itr+1
                if tree == 0:
                    ip1 = np.concatenate((st, gl))
                    st = mpnet.plan(obs, np.array([ip1]))
                    pA.append(st)
                    tree = 1
                else:
                    ip2 = np.concatenate((gl, st))
                    gl = mpnet.plan(obs, np.array([ip2]))
                    pB.append(gl)
                    tree = 0
                target_reached = steerTo(st, gl, collHandle, step_sz=step_sz)
            if target_reached == 0:
                print("failed to replan")
                return 0
            else:
                for p1 in range(0, len(pA)):
                    new_path.append(pA[p1])
                for p2 in range(len(pB)-1, -1, -1):
                    new_path.append(pB[p2])

    return new_path


def save_feasible_path(path, filename):
    with open(filename+'.pkl', 'wb') as good_f:
        pickle.dump(path, good_f)


# def check_full_path(overall_path):
#     invalid = []

#     valid = True
#     overall_valid = True
#     for i, state in enumerate(overall_path):
#         filler_robot_state[10:17] = moveit_scrambler(state)
#         rs_man.joint_state.position = tuple(filler_robot_state)
#         collision_free = sv.getStateValidity(rs_man, group_name="right_arm")

#         valid = valid and collision_free
#         overall_valid = overall_valid and collision_free

#         if not valid:
#             invalid.append(i)
#             valid = True

#     if (len(invalid) == 0 and overall_valid):
#         print("Full path valid!")
#     else:
#         print("Not valid")

#     return overall_valid


# def path_to_np(path):
#     path_np = []
#     for i, state in enumerate(path):
#         path_np.append(np.multiply(state.numpy(), joint_ranges))  # unnormalize
#     return path_np


# def make_overall_path(path_np):
#     dists = []
#     for i in range(1, len(path_np)):
#         dists.append(np.mean(abs(path_np[i] - path_np[i-1]), axis=0))

#     overall_dist = sum(dists)
#     fractions = [x for x in dists/overall_dist]
#     total_pts = 300
#     pts = [int(total_pts * x) for x in fractions]
#     path_full = []
#     for i, n_pts in enumerate(pts):
#         vec = np.transpose(np.linspace(
#             path_np[i][0], path_np[i+1][0], n_pts)[np.newaxis])
#         for j in range(1, 7):
#             vec = np.hstack([vec, np.transpose(np.linspace(
#                 path_np[i][j], path_np[i+1][j], n_pts)[np.newaxis])])
#         path_full.append(vec)

#     overall_path = []
#     for mini_path in path_full:
#         for state in mini_path:
#             overall_path.append(state)

#     return overall_path


# def play_smooth(overall_path, limb):

#     done = False
#     joint_state = limb.joint_angles()

#     while not done:
#         for i, name in enumerate(joint_state.keys()):
#             joint_state[name] = overall_path[k][i]
#         limb.set_joint_positions(joint_state)
#         time.sleep(0.025)
#         k += 1
#         if k > len(overall_path)-1:
#             done = True
