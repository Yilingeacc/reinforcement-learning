import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

# hyper-parameters
LR = 0.01
GAMMA = 0.9
eps = np.finfo(np.float32).eps.item()

hidden_dim = 24

save_point = 100

env_name = 'CartPole-v1'
model_name = env_name[:env_name.find('-')]


class Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.act = nn.Linear(hidden_dim, action_dim)
        self.cri = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        temp = self.net(state)
        action_probs = self.act(temp)
        state_value = self.cri(temp)
        return F.softmax(action_probs, dim=-1), state_value


class PolicyGradientWithBaseline:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.net = Net(state_dim, hidden_dim, action_dim).cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.save_tuple, self.episode_rewards = [], []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).cuda().detach_()
        probs, state_value = self.net(state)
        action_distribution = Categorical(probs)
        action = action_distribution.sample()
        self.save_tuple.append((state_value, action_distribution.log_prob(action)))
        return action.item()

    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def __get_rewards(self):
        Gt = 0
        rewards = deque()
        for r in reversed(self.episode_rewards):
            Gt = r + GAMMA * Gt
            rewards.appendleft(Gt)
        return rewards

    def learn(self):
        rewards = torch.tensor(self.__get_rewards())
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        policy_loss, value_loss = [], []

        for (state_value, action_log_prob), reward in zip(self.save_tuple, rewards):
            delta = reward - state_value.item()
            policy_loss.append(-action_log_prob * delta)
            value_loss.append(F.smooth_l1_loss(state_value, torch.tensor([reward]).cuda()))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.save_tuple, self.episode_rewards = [], []

    def save_state_dict(self):
        torch.save(self.net.state_dict(), model_name + '-baseline' + '.pkl')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    pg = PolicyGradientWithBaseline(state_dim, hidden_dim, action_dim)

    for episode in range(10000):
        state = env.reset()
        done = False

        while True:
            env.render()

            action = pg.choose_action(state)

            state_, reward, done, info = env.step(action)

            reward = -10. if done else reward

            pg.store_reward(reward)

            if done:
                pg.learn()
                break

            state = state_

        if episode and not episode % save_point:
            pg.save_state_dict()

    env.close()
