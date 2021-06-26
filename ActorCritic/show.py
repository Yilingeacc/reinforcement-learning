import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
from copy import deepcopy

hidden_dim = 2 ** 9

env_name = 'LunarLanderContinuous-v2'
model_name = env_name[:env_name.find('-')]

learn_steps = 49328

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        return self.net(state).tanh()

    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        action = a_avg + noise * a_std
        return action, noise


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.learn_cntr = learn_steps
        self.act = Actor(state_dim, hidden_dim, action_dim).cuda()
        self.act.load_state_dict(torch.load(model_name + '-act' + str(self.learn_cntr) + '.pkl'))

    def choose_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32).cuda().detach()
        action, noise = self.act.get_action_noise(state)
        return action[0].detach().cpu().numpy(), noise[0].detach().cpu().numpy()


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, hidden_dim, action_dim)

    for episode in range(1000000):
        state = env.reset()

        done = False

        while True:
            env.render()

            action, _ = ppo.choose_action(state)

            state_, reward, done, info = env.step(np.tanh(action))

            if done:
                break

            state = state_

    env.close()