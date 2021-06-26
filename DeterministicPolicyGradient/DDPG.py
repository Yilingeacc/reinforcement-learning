import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
import ReplayBuffer as Buffer
from copy import deepcopy

# hyper-parameters
learning_rate = 1e-4  # learning rate
gamma = 0.99  # discount factor
tau = 2 ** -8  # soft update parameter
explore_noise = 0.3  # ou noise

hidden_dim = 2 ** 8

batch_size = 2 ** 6
buffer_capacity = 2 ** 17
repeat_times = 2 ** 10

save_point = 2 ** 13

env_name = ''
model_name = env_name[:env_name.find('-')]


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))

    def forward(self, state):
        return self.net(state).tanh()


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=1))


class OUnoise:
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """
        The noise of Ornstein-Uhlenbeck Process

            It makes Zero-mean Gaussian Noise more stable.
            It helps agent explore better in a inertial system.
            Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

            :int size: the size of noise, noise.shape==(-1, action_dim)
            :float theta: related to the not independent of OU-noise
            :float sigma: related to action noise std
            :float ou_noise: initialize OU-noise
            :float dt: derivative
        """
        self.size, self.theta, self.sigma, self.ou_noise, self.dt = size, theta, sigma, ou_noise, dt

    def __call__(self):
        noise = self.sigma * np.sqrt(self.dt) * random.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.learn_cntr = 0
        self.act = Actor(state_dim, hidden_dim, action_dim).cuda()
        self.act_target = deepcopy(self.act)
        self.cri = Critic(state_dim, hidden_dim, action_dim).cuda()
        self.cri_target = deepcopy(self.cri)

        self.act_optimizer = optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = optim.Adam(self.cri.parameters(), lr=learning_rate)

        self.loss = nn.MSELoss()

        self.buffer = Buffer.ReplayBuffer(buffer_capacity, state_dim, continuous=True)

        self.ou_noise = OUnoise(action_dim, ou_noise=explore_noise)

        self.state = None

    def choose_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32).cuda().detach_()
        action = self.act(state)[0].cpu().detach().numpy()
        return (action + self.ou_noise()).clip(-1., 1.)

    def store_transition(self, state, action, reward, mask):
        self.buffer.store(torch.tensor(state), torch.tensor(action), reward, mask)

    def explore_env(self, env):
        state = env.reset() if self.state is None else self.state
        for step in range(repeat_times):
            action = self.choose_action(state)
            state_, reward, done, info = env.step(action)
            self.store_transition(state, action, reward, 0. if done else gamma)
            state = env.reset() if done else state_
        self.state = state

    def prepare_for_trainning(self, env):
        with torch.no_grad():
            self.explore_env(env)
        self.learn()
        self.act_target.load_state_dict(self.act.state_dict())
        self.cri_target.load_state_dict(self.cri.state_dict())

    def learn(self):
        for step in range(repeat_times):
            with torch.no_grad():
                states, actions, rewards, masks, next_states = self.buffer.sample(batch_size)
                q_next = self.cri_target(next_states, self.act_target(next_states))
                q_target = rewards + masks * q_next
            q_eval = self.cri(states, actions)

            # prioritized experience replay buffer only
            if isinstance(self.buffer, Buffer.PERBuffer):
                with torch.no_grad():
                    errors = (q_eval - q_target).detach().cpu().numpy().squeeze()
                self.buffer.batch_update(np.abs(errors))

            cri_loss = self.loss(q_eval, q_target)

            self.cri_optimizer.zero_grad()
            cri_loss.backward()
            self.cri_optimizer.step()
            self.soft_update(self.cri, self.cri_target)

            act_loss = -self.cri(states, self.act(states)).mean()

            self.act_optimizer.zero_grad()
            act_loss.backward()
            self.act_optimizer.step()
            self.soft_update(self.act, self.act_target)

        self.learn_cntr += repeat_times
        if not self.learn_cntr % save_point:
            print('save to disk: ' + model_name + '-act' + str(self.learn_cntr) + '.pkl')
            print('save to disk: ' + model_name + '-cri' + str(self.learn_cntr) + '.pkl')
            torch.save(self.act.state_dict(), model_name + '-act' + str(self.learn_cntr) + '.pkl')
            torch.save(self.cri.state_dict(), model_name + '-cri' + str(self.learn_cntr) + '.pkl')

    @staticmethod
    def soft_update(eval_net, target_net):
        for eval, tar in zip(eval_net.parameters(), target_net.parameters()):
            tar.data.copy_(eval.data.__mul__(tau) + tar.data.__mul__(1 - tau))


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ddpg = DDPG(state_dim, hidden_dim, action_dim)

    ddpg.prepare_for_trainning(env)

    while True:
        with torch.no_grad():
            ddpg.explore_env(env)
        ddpg.learn()
