from simple_maze import SimpleMaze
import random
import cv2
from data_utils import load_data

import numpy as np
import argparse


def test_model(policy, n_test=20, gui_enable=False, img_obs=False):
    success = 0
    
    goal = np.array([0.2, 0.4])
    init_pos = np.array([0.8, -0.4])

    max_dists = np.linalg.norm(goal - init_pos)
    min_dists = []
    for _ in range(n_test):
        env = SimpleMaze(gui_enabled=gui_enable, img_obs=img_obs)
        obs = env.reset()
        done = False
        positions = []
        for step in range(100):
            obs =  np.reshape(obs, newshape=[1, -1])
            act = policy.get_actions(obs)[0]
            obs, _, done, info = env.step(act)
            positions.append(info['agent_pos'])
            if done:
                success += 1
                break
        dist = np.min(np.linalg.norm(np.asarray(positions)[:, :2] - goal, axis=-1))
        if dist < 0.1:
            dist = 0
        min_dists.append(dist)
        env.close()

    success_rate = success / n_test
    mean_min_dist = np.mean(min_dists)

    score = (max_dists - mean_min_dist) / max_dists
    score = min(max(0., score), 1.)

    # compute the actual score for grading
    if img_obs:
        if score >= 0.99:
            score = 5
        elif score >= 0.9:
            score = 4
        elif score >= 0.8:
            score = 3
        elif score >= 0.6:
            score = 2
        else:
            score = 0
    else:
        if score >= 0.99:
            score = 5
        elif score >= 0.8:
            score = 3
        else:
            score = 0
    return success_rate, mean_min_dist, score


def score_position_bc(gui_enable=False):
    from solutions.pos_bc_robot import POSBCRobot
    policy = POSBCRobot()
    data = load_data('./data/bc_with_gtpos_data.pkl')
    policy.train(data)
    _, _, score = test_model(policy, n_test=20, gui_enable=gui_enable, img_obs=False)
    return score


def score_img_bc(gui_enable=False):
    from solutions.rgb_bc_robot import RGBBCRobot
    policy = RGBBCRobot()
    data = load_data('./data/bc_data.pkl')
    policy.train(data)
    _, _, score = test_model(policy, n_test=20,  gui_enable=gui_enable, img_obs=True)
    return score


def score_regressor():
    from solutions.pos_regressor import PositionRegressor
    regressor = PositionRegressor()
    data = load_data('./data/regression_data.pkl')
    regressor.train(data)
    positions = np.asarray([info['agent_pos'] for info in data['info']])
    pred_pos = regressor.predict(data['obs'])
    mse = np.mean(np.sum(np.square(positions - pred_pos), axis=1))
    target_mse = 1.1e-5
    if mse < target_mse:
        mse = 0.
    mse_dist = (mse - target_mse)
    score = 1. - mse_dist
    score = min(max(0., score), 1.)

    test_data = load_data('./data/reg_test_data.pkl')
    test_positions = np.asarray([info['agent_pos'] for info in test_data['info']])[:10,]
    test_pred_pos = regressor.predict(test_data['obs'][:10])
    test_mse = np.mean(np.sum(np.square(test_positions - test_pred_pos), axis=1))
    target_mse = 0.038
    if test_mse < target_mse:
        test_mse = 0.
    test_mse_dist = (mse - target_mse)
    test_score = 1. - mse_dist
    test_score = min(max(0., score), 1.)
    # print('train_mse:', mse)
    # print('test_mse:', test_mse)
    score = 0.5 * score + 0.5 * test_score
    # compute actual score for grading
    if score >= 0.99:
        score = 5
    elif score >= 0.95:
        score = 4
    elif score >= 0.8:
        score = 2
    else:
        score = 0
    return score


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', dest='gui_enable', action='store_true')
    args = parser.parse_args()
    score_pos = score_position_bc(gui_enable=args.gui_enable)
    score_img = score_img_bc(gui_enable=args.gui_enable)
    score_reg = score_regressor()
    final_score = np.sum([score_pos, score_img, score_reg])
    print('\n\n\n--------SCORES--------')
    print(f'Position regression: {score_reg}/5')
    print(f'BC with positions: {score_pos}/5')
    print(f'BC with rgb images: {score_img}/5')
    print(f'\nFinal score: {final_score}/15')
    print('----------------------')
