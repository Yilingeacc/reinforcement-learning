import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy import random
import ReplayBuffer as Buffer
from copy import deepcopy

# hyper-parameters
epsilon = 0.1  # epsilon-greedy parameter
learning_rate = 1e-4  # learning rate
gamma = 0.99  # discount factor
tau = 2 ** -8  # soft update parameter

hidden_dim = 2 ** 8

batch_size = 2 ** 6
buffer_capacity = 2 ** 17
repeat_times = 2 ** 10

save_point = 2 ** 13

env_name = ''
model_name = env_name[:env_name.find('-')]


class Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.val_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, 1))
        self.adv_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, action_dim))

    def forward(self, state):
        temp = self.net(state)
        s_val = self.val_net(temp)
        q_adv = self.adv_net(temp)
        return s_val + q_adv - q_adv.mean(dim=1, keepdim=True)


class DuelingDQN:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.action_dim = action_dim
        self.learn_cntr = 0

        self.eval_net = Net(state_dim, hidden_dim, action_dim).cuda()
        self.target_net = deepcopy(self.eval_net)

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.buffer = Buffer.ReplayBuffer(buffer_capacity, state_dim)

        self.state = None

    def choose_action(self, state):
        if random.uniform() > epsilon:
            state = torch.tensor((state,), dtype=torch.float32).cuda().detach_()
            action_value = self.eval_net(state)[0]
            action = action_value.argmax(dim=0).item()
        else:
            action = random.randint(self.action_dim)
        return action

    def store_transition(self, state, action, reward, mask):
        self.buffer.store(torch.tensor(state), action, reward, mask)

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
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        for step in range(repeat_times):
            with torch.no_grad():
                states, actions, rewards, masks, next_states = self.buffer.sample(batch_size)
                q_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                q_target = rewards + masks * q_next
            q_eval = self.eval_net(states).gather(1, actions.type(torch.long))

            # prioritized experience replay buffer only
            if isinstance(self.buffer, Buffer.PERBuffer):
                with torch.no_grad():
                    errors = (q_eval - q_target).detach().cpu().numpy().squeeze()
                self.buffer.batch_update(np.abs(errors))

            loss = self.loss(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.soft_update(self.eval_net, self.target_net)

        self.learn_cntr += repeat_times
        if not self.learn_cntr % save_point:
            print('save to disk: ' + model_name + str(self.learn_cntr) + '.pkl')
            torch.save(self.eval_net.state_dict(), model_name + str(self.learn_cntr) + '.pkl')

    @staticmethod
    def soft_update(eval_net, target_net):
        for eval, tar in zip(eval_net.parameters(), target_net.parameters()):
            tar.data.copy_(eval.data.__mul__(tau) + tar.data.__mul__(1 - tau))


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ddqn = DuelingDQN(state_dim, hidden_dim, action_dim)

    ddqn.prepare_for_trainning(env)

    while True:
        with torch.no_grad():
            ddqn.explore_env(env)
        ddqn.learn()
