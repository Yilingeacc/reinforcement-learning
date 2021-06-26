import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ReplayBuffer as Buffer
from copy import deepcopy

# hyper-parameters
learning_rate = 1e-4  # learning rate
gamma = 0.99  # discount factor
tau = 2 ** -8  # soft update parameter
explore_noise = 0.1  # standard deviation of explore noise
policy_noise = 0.2  # standard deviation of policy noise
noise_clip = 0.5
update_freq = 2  # delay update frequency for soft target update

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

    def get_action(self, state):
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
        return (action + noise).clamp(-1., 1.)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.q1_net = nn.Linear(hidden_dim, 1)
        self.q2_net = nn.Linear(hidden_dim, 1)
        self.layer_norm(self.q1_net, std=.1)
        self.layer_norm(self.q2_net, std=.1)

    def forward(self, state, action):
        temp = self.net(torch.cat((state, action), dim=1))
        return self.q1_net(temp)

    def get_q1_q2(self, state, action):
        temp = self.net(torch.cat((state, action), dim=1))
        return self.q1_net(temp), self.q2_net(temp)

    @staticmethod
    def layer_norm(layer, std=1., bias_const=1e-6):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)


class TD3:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.learn_cntr = 0
        self.act = Actor(state_dim, hidden_dim, action_dim).cuda()
        self.act_target = deepcopy(self.act)
        self.cri = Critic(state_dim, hidden_dim, action_dim).cuda()
        self.cri_target = deepcopy(self.cri)

        self.act_optimizer = optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = optim.Adam(self.cri.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.buffer = Buffer.PERBuffer(buffer_capacity, state_dim, continuous=True)

        self.state = None

    def choose_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32).cuda().detach_()
        action = self.act(state)[0]
        noise = torch.randn_like(action) * explore_noise
        return (action + noise).clamp(-1., 1.).cpu().detach().numpy()

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
                next_actions = self.act_target.get_action(next_states)
                q_next = torch.min(*self.cri_target.get_q1_q2(next_states, next_actions))
                q_target = rewards + masks * q_next
            q1, q2 = self.cri.get_q1_q2(states, actions)

            # prioritized experience replay buffer only
            if isinstance(self.buffer, Buffer.PERBuffer):
                with torch.no_grad():
                    errors = (q1 + q2 - q_target * 2).detach().cpu().numpy().squeeze()
                self.buffer.batch_update(np.abs(errors))

            cri_loss = self.loss(q1, q_target) + self.loss(q2, q_target)

            self.cri_optimizer.zero_grad()
            cri_loss.backward()
            self.cri_optimizer.step()
            if not step % update_freq:
                act_loss = -self.cri(states, self.act(states)).mean()

                self.act_optimizer.zero_grad()
                act_loss.backward()
                self.act_optimizer.step()
                self.soft_update(self.cri, self.cri_target)
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
    td3 = TD3(state_dim, hidden_dim, action_dim)

    td3.prepare_for_trainning(env)

    while True:
        with torch.no_grad():
            td3.explore_env(env)
        td3.learn()
