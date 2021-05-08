import torch
import argparse
import numpy as np
from cartpole_env import make_cart_pole_env, configure_pybullet
from q_network import QNetwork
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--gui', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    args = parser.parse_args()
    return args


def enjoy_dqn(env, args, device):
    if args.model_path is not None:
        q_network = QNetwork(env).to(device)
        q_network.load_state_dict(torch.load(args.model_path))

    episode_reward_list = []
    success_list = []
    for episode in range(100):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if args.model_path is not None:
                logits = q_network.forward(obs.reshape((1,) + obs.shape), device)
                action = torch.argmax(logits, dim=1).tolist()[0]
            else:
                action = np.random.choice(range(env.action_dims))
            obs, reward, done, _ = env.step(action)
            if args.gui:
                time.sleep(env.timestep)
            episode_reward += reward
        success = episode_reward == 500
        episode_reward_list.append(episode_reward)
        success_list.append(success)
        print(f'\nepisode: {episode}, reward: {episode_reward}, success: {success}')

    average_reward = np.mean(episode_reward_list)
    success_rate = np.mean(success_list)
    score = success_rate * 15
    print("\n----------------------------------------")
    print(f'average reward: {average_reward:.2f} / 500.00')
    print(f'success rate: {success_rate:.2f} / 1.00')
    print(f'final score: {score:.2f} / 15.00')
    print("----------------------------------------\n")


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(rendering=args.gui, debug=True)
    env = make_cart_pole_env()
    device = torch.device('cpu')

    enjoy_dqn(env, args, device)
