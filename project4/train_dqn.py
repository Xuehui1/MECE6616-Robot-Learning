import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import numpy as np
import time
import random
import os
from cartpole_env import make_cart_pole_env, configure_pybullet
from replay_buffer import ReplayBuffer
from q_network import QNetwork


class DQN(object):
    def __init__(self):
        # Hyper Parameters
        env = make_cart_pole_env()
        self.buffer_limit=10000
        self.r_b = ReplayBuffer(self.buffer_limit)
        self.batchsize = 64
        self.lr = 0.001                   # learning rate
        self.epsilon = 0.1               # greedy policy
        self.gamma = 0.99                 # reward discount
        self.target_replace_iter = 500   # Qtarget network update frequency
        self.N_ACTIONS = env.action_dims #3
        # observation, action, reward, next_observation, done
        self.N_STATES = env.observation_space.shape[0] #5 
        self.device = torch.device('cpu')



        self.eval_net= QNetwork(env)
        self.target_net = QNetwork(env)
        self.learn_step_counter = 0   # for target updating
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        if random.random() <  self.epsilon:
            action = env.action_space.sample()
        else:
            nums = self.eval_net.forward(x.reshape((1,) + x.shape), self.device)
            action = torch.argmax(nums,dim=1).tolist()[0]
        return action

    # observation, action, reward, next_observation, done
    def store_transition(self, s, a, r, s_,done):
        transition = (s,a , r, s_, done)
        self.r_b.put(transition)

    def learn(self):
        # Qeval network parameter update each time
        # Qtarget network parameter update sometimes
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        b_s ,b_a ,b_r ,b_s_, done= [], [], [], [], []
        b_s ,b_a ,b_r ,b_s_, done =self.r_b.sample(self.batchsize)
        b_s=torch.FloatTensor(b_s)
        b_a =torch.LongTensor(b_a)
        b_r =torch.FloatTensor(b_r)
        b_s_=torch.FloatTensor(b_s_)
        done =torch.FloatTensor(done)

        q_eval = self.eval_net.forward(b_s, device).gather(1, b_a.view(-1,1)).squeeze()
        target_max = torch.max(self.target_net.forward(b_s_, device).detach(), dim=1)[0]
        #gamma 0.9 0.95 0.99  * (1 - done)!!!important
        q_target = b_r + self.gamma * target_max * (1 - done)
        loss = self.loss_func(q_target, q_eval)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--save_dir', type=str, default='models',
                        help="the root folder for saving the checkpoints")
    parser.add_argument('--gui', action='store_true', default=False,
                        help="whether to turn on GUI or not")
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    if not args.seed:
        args.seed = int(time.time())
    return args


def train_dqn(env, args, device):
    # set up seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    # ---
    # Write your code here to train DQN
    dqn = DQN()
    print('\nStart Collecting experience...')
    best_reward=0
    ep_r = 0
    episode=0
    s = env.reset()
    for i in range(70000):
        #env.render()
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)
        # observation, action, reward, next_observation, done
        dqn.store_transition(s, a, r, s_,done)
        ep_r += r
        if len(dqn.r_b.buffer) == dqn.buffer_limit:
            dqn.learn()

        s = s_
        if done:
            s = env.reset()
            episode=episode+1
            print('episode: ', episode, '| reward: ', round(ep_r, 2))
            
            model_folder_name = f'episode_{episode:06d}_reward_{round(ep_r):03d}'
            if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
                os.makedirs(os.path.join(args.save_dir, model_folder_name))
            torch.save(dqn.eval_net.state_dict(), os.path.join(args.save_dir, model_folder_name, 'q_network.pth'))
            print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "q_network.pth")}\n')
            
            ep_r=0
    # ---


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(rendering=args.gui, debug=True)
    env = make_cart_pole_env()
    device = torch.device('cpu')

    train_dqn(env, args, device)
