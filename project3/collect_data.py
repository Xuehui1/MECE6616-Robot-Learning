import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
import math
import time
import torch
#np.random.seed(0)
#torch.manual_seed(0)

np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    # X = np.zeros((arm_teacher.dynamics.get_state_dim() + arm_teacher.dynamics.get_action_dim(), 0))
    # Y = np.zeros((arm_teacher.dynamics.get_state_dim(), 0))
    X = []
    Y = []
    #for i,torque in enumerate([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]):

    for torque in range(0, 1500):
        torque=np.random.randint(-1600, 1600)
        torque=torque/1000

        if torque == 0:
            continue
    
     
        print('Collect data for torque', torque)

        #according to run.py, reset
        initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))
        initial_state[0] = -math.pi / 2.0
        arm_teacher.set_state(initial_state)
        action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
        action[0] = torque
        arm_teacher.set_action(action)
        arm_teacher.set_t(0)

        #
        dt = args.time_step
        time_limit=5
        while arm_teacher.get_t() < time_limit:
            t = time.time()

            #For X
            x_state = arm_teacher.get_state()
            x_action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
            x_action[0] = torque
            X.append(np.concatenate((x_state, x_action), axis=0))
            
            #For Y
            arm_teacher.advance()
            y = arm_teacher.get_state()
            Y.append(y)
            time.sleep(max(0, dt - (time.time() - t)))
    # ---

    X = np.hstack(X)
    Y = np.hstack(Y)


    print('X shape:',X.shape,'Y shape:',Y.shape)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
