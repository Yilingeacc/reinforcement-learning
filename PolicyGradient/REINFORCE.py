import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque

# hyper-parameters
LR = 0.01  # learning rate
gamma = 0.9  # reward discount

hidden_dim = 24

save_point = 100

env_name = 'CartPole-v1'
model_name = env_name[:env_name.find('-')]

eps = np.finfo(np.float32).eps.item()


class Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))

    def forward(self, state):
        return F.softmax(self.net(state), dim=-1)


class PolicyGradient:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.net = Net(state_dim, hidden_dim, action_dim).cuda()

        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)

        self.episode_log_probs, self.episode_rewards = [], []

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).cuda().detach_()
        probs = self.net(state)
        action_distribution = Categorical(probs)
        action = action_distribution.sample()
        self.episode_log_probs.append(action_distribution.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def __get_rewards(self):
        Gt = 0
        rewards = deque()
        for r in reversed(self.episode_rewards):
            Gt = r + gamma * Gt
            rewards.appendleft(Gt)
        return rewards

    def learn(self):
        """
        learn when the episode is finished
        """
        rewards = torch.tensor(self.__get_rewards()).cuda()

        # standard score:
        # definition: zi = (xi - average(x)) / s
        # which s is the standard deviation of x
        # the standard score has the characteristic that
        # the average is 0 and the standard deviation is 1
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        policy_loss = []
        for log_prob, reward in zip(self.episode_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.episode_log_probs, self.episode_rewards = [], []

    def save_state_dict(self, episode):
        print('save to disk: ' + model_name + str(episode) + '.pkl')
        torch.save(self.net.state_dict(), model_name + str(episode) + '.pkl')


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    pg = PolicyGradient(state_dim, hidden_dim, action_dim)

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
            pg.save_state_dict(episode)

    env.close()
