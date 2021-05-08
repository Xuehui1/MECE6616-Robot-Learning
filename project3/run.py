import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch
np.set_printoptions(suppress=True)


def reset(arm_teacher, arm_student, torque):
    initial_state = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))  # position + velocity
    initial_state[0] = -math.pi / 2.0
    arm_teacher.set_state(initial_state)
    arm_student.set_state(initial_state)

    action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
    action[0] = torque
    arm_teacher.set_action(action)
    arm_student.set_action(action)

    arm_teacher.set_t(0)
    arm_student.set_t(0)


def main(args):
    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # Student arm 
    dynamics_student = ArmDynamicsStudent(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    if args.model_path is not None:
        dynamics_student.init_model(args.model_path, args.num_links, device=torch.device('cpu'))
    arm_student = Robot(dynamics_student)

    scores = []
    for i, torque in enumerate([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]):
        print("\n----------------------------------------")
        print(f'TEST {i+1} ( Torque = {torque} Nm)\n')
        reset(arm_teacher, arm_student, torque)

        if args.gui:
            renderer = Renderer()
            time.sleep(1)

        dt = args.time_step
        mse_list = []
        while args.time_limit < 0 or arm_teacher.get_t() < args.time_limit:
            t = time.time()
            arm_teacher.advance()
            arm_student.advance()
            if args.gui:
                renderer.plot([(arm_teacher, 'tab:blue'), (arm_student, 'tab:red')])
            mse = ((arm_student.get_state() - arm_teacher.get_state())**2).mean()
            time.sleep(max(0, dt - (time.time() - t)))
            mse_list.append(mse)

        if args.gui:
            renderer.plot(None)
            time.sleep(2)

        mse = np.array(mse_list).mean()
        print(f'average mse: {mse}')
        score = 2 if mse < 0.008 else 0
        scores.append(score)
        print(f'Score: {score}/{2}')
        print("----------------------------------------\n")

    print("\n----------------------------------------")
    print(f'Final Score: {np.array(scores).sum()}/{12}')
    print("----------------------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--gui', action='store_const', const=True, default=False)
    parser.add_argument('--model_path', type=str)
    main(parser.parse_args())
