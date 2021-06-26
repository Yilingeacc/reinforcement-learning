import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from tensorboardX import SummaryWriter
from make_env import make_env

# hyper parameters
learning_rate = 1e-4
gamma = 0.99
tau = 2 ** -8
grad_norm_clip = 10
hidden_dim = 2 ** 9

max_episode_len = 2 ** 8
batch_size = 2 ** 4

init_epsilon = 1.
epsilon_decay = .01
final_epsilon = .05

buffer_capacity = 2 ** 10
save_point = 2 ** 12

n_epochs = 2 ** 16
n_episodes = 2 ** 4
n_train_steps = 2 ** 4

eval_episodes = 2 ** 2

Argument = namedtuple('Argument', 'state_dim obs_dim hidden_dim n_actions n_agents')

scenario_name = 'simple_spread'
model_name = scenario_name[scenario_name.find('_') + 1:]

writer = SummaryWriter('./result')


class AgentNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_shape = args.obs_dim + args.n_actions + args.n_agents
        self.fc = nn.Linear(input_shape, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.out = nn.Linear(args.hidden_dim, args.n_actions)

    def forward(self, obs, hidden_state):
        x = F.relu(self.fc(obs))
        h = self.rnn(x, hidden_state.reshape(-1, self.args.hidden_dim))
        q = self.out(h)
        return q, h


class MixNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hyper_w1 = nn.Linear(args.state_dim, args.hidden_dim * args.n_agents)
        self.hyper_w2 = nn.Linear(args.state_dim, args.hidden_dim * 1)
        self.hyper_b1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_dim, args.hidden_dim), nn.ReLU(),
                                      nn.Linear(args.hidden_dim, 1))

    def forward(self, q_values, states):
        """
        q_values: (episode_num, max_episode_len, n_agents)
        states: (episode_num, max_episode_len, state_dim)
        """
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)  # (episode_num * max_episode_len, 1, n_agents)
        states = states.reshape(-1, self.args.state_dim)  # (episode_num * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(states)).view(-1, self.args.n_agents, self.args.hidden_dim)
        b1 = self.hyper_b1(states).view(-1, 1, self.args.hidden_dim)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        w2 = torch.abs(self.hyper_w2(states)).view(-1, self.args.hidden_dim, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        q_total = (torch.bmm(hidden, w2) + b2).view(episode_num, -1, 1)
        return q_total


class Qmix:
    def __init__(self, args):
        self.args = args
        self.agent_eval = AgentNet(args).cuda()
        self.agent_target = AgentNet(args).cuda()
        self.mix_eval = MixNet(args).cuda()
        self.mix_target = MixNet(args).cuda()

        self.agent_target.load_state_dict(self.agent_eval.state_dict())
        self.mix_target.load_state_dict(self.mix_eval.state_dict())

        self.eval_parameters = list(self.agent_eval.parameters()) + list(self.mix_eval.parameters())

        self.optimizer = optim.RMSprop(self.eval_parameters, lr=learning_rate)

        self.eval_hidden, self.target_hidden = None, None

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.hidden_dim)).cuda()
        self.target_hidden = torch.zeros((episode_num, self.args.n_agents, self.args.hidden_dim)).cuda()

    def choose_action(self, obs, last_action, agent_num, epsilon):
        agent_id = np.zeros(self.args.n_agents)
        agent_id[agent_num] = 1.

        inputs = torch.tensor(np.hstack((obs.copy(), last_action, agent_id)), dtype=torch.float32).unsqueeze(0).cuda()
        hidden_state = self.eval_hidden[:, agent_num, :]

        q_value, self.eval_hidden[:, agent_num, :] = self.agent_eval(inputs, hidden_state)
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.args.n_actions)
        else:
            action = torch.argmax(q_value)
        return action

    def learn(self, batch):
        """
        batch: (episode, transition, agent, obs)
        一次抽取多个episode，然后一次给神经网络，传入每个episode同一个位置的transition
        """
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            batch[key] = torch.tensor(batch[key], dtype=torch.long if key == 'u' else torch.float32).cuda()
        s, s_, u, r, m = batch['s'], batch['s_'], batch['u'], batch['r'], batch['m']

        q_evals, q_targets = self._get_values(batch)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.mix_eval(q_evals, s)
        q_total_target = self.mix_target(q_targets, s_)

        targets = r + m * q_total_target

        td_error = (q_total_eval - targets.detach())

        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_parameters, grad_norm_clip)
        self.optimizer.step()

        self._update_network()

    def _update_network(self):
        self._soft_update(self.agent_eval, self.agent_target)
        self._soft_update(self.mix_eval, self.mix_target)

    @staticmethod
    def _soft_update(eval_net, target_net):
        for eval, tar in zip(eval_net.parameters(), target_net.parameters()):
            tar.data.copy_(eval.data.__mul__(tau) + tar.data.__mul__(1 - tau))

    def _get_inputs(self, batch, transition_idx):
        o, o_, u_onehot = batch['o'][:, transition_idx], batch['o_'][:, transition_idx], batch['u_onehot'][:]
        episode_num = o.shape[0]
        inputs, inputs_ = [], []
        inputs.append(o)
        inputs_.append(o_)

        if transition_idx == 0:
            inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        else:
            inputs.append(u_onehot[:, transition_idx - 1])
        inputs_.append(u_onehot[:, transition_idx])

        inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).cuda())
        inputs_.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1).cuda())

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1).cuda()
        inputs_ = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_], dim=1).cuda()
        return inputs, inputs_

    def _get_values(self, batch):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_ = self._get_inputs(batch, transition_idx)
            q_eval, self.eval_hidden = self.agent_eval(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.agent_target(inputs_, self.target_hidden)

            q_evals.append(q_eval.view(episode_num, self.args.n_agents, -1))
            q_targets.append(q_target.view(episode_num, self.args.n_agents, -1))
        q_evals = torch.stack(q_evals, dim=1).cuda()
        q_targets = torch.stack(q_targets, dim=1).cuda()
        return q_evals, q_targets

    def save_model(self, train_step):
        num = str(train_step // save_point)
        print('train step: %d, save to disk' % train_step)
        torch.save(self.agent_eval.state_dict(), 'AgentNet_' + str(num) + '_params.pkl')
        torch.save(self.mix_eval.state_dict(), 'MixNet_' + str(num) + '_params.pkl')


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.buffers = {'o': np.empty((buffer_capacity, max_episode_len, args.n_agents, args.obs_dim)),
                        'u': np.empty((buffer_capacity, max_episode_len, args.n_agents, 1)),
                        's': np.empty((buffer_capacity, max_episode_len, args.state_dim)),
                        'r': np.empty((buffer_capacity, max_episode_len, 1)),
                        'o_': np.empty((buffer_capacity, max_episode_len, args.n_agents, args.obs_dim)),
                        's_': np.empty((buffer_capacity, max_episode_len, args.state_dim)),
                        'u_onehot': np.empty((buffer_capacity, max_episode_len, args.n_agents, args.n_actions)),
                        'm': np.empty((buffer_capacity, max_episode_len, 1))
                        }
        self.mem_cntr = 0

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]
        idx = self.mem_cntr % buffer_capacity
        idxs = np.arange(idx, idx + batch_size)
        self.buffers['o'][idxs] = episode_batch['o']
        self.buffers['u'][idxs] = episode_batch['u']
        self.buffers['s'][idxs] = episode_batch['s']
        self.buffers['r'][idxs] = episode_batch['r']
        self.buffers['o_'][idxs] = episode_batch['o_']
        self.buffers['s_'][idxs] = episode_batch['s_']
        self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
        self.buffers['m'][idxs] = episode_batch['m']
        self.mem_cntr += batch_size

    def sample(self, batch_size):
        mini_batch = {}
        idxs = np.random.randint(0, min(self.mem_cntr, buffer_capacity), batch_size)
        for key in self.buffers.keys():
            mini_batch[key] = self.buffers[key][idxs]
        return mini_batch

    def empty_buffer(self):
        self.mem_cntr = 0

    def __len__(self):
        return min(self.mem_cntr, buffer_capacity)


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.policy = Qmix(args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.epsilon = init_epsilon

    def generate_episode(self):
        o, u, r, s, u_onehot, m = [], [], [], [], [], []
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.policy.init_hidden(1)
        self.env.reset()
        for step in range(max_episode_len):
            obs = self.env.get_obs()
            state = np.concatenate(obs)
            actions, actions_onehot = [], []
            for agent_id in range(self.args.n_agents):
                action = self.policy.choose_action(obs[agent_id], last_action[agent_id], agent_id, self.epsilon)
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int(action))
                actions_onehot.append(action_onehot)
                last_action[agent_id] = action_onehot

            obs_, reward, *_ = self.env.step(actions)
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, (self.args.n_agents, 1)))
            u_onehot.append(actions_onehot)
            r.append([reward[0]])
            m.append([0. if step == max_episode_len - 1 else gamma])
        self._update_epsilon()
        obs = self.env.get_obs()
        state = np.concatenate(obs)
        o.append(obs)
        s.append(state)
        o_ = o[1:]
        s_ = s[1:]
        o = o[:-1]
        s = s[:-1]

        episode = dict(o=o.copy(), s=s.copy(), u=u.copy(), r=r.copy(), o_=o_.copy(), s_=s_.copy(),
                       u_onehot=u_onehot.copy(), m=m.copy())

        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        return episode

    def _update_epsilon(self):
        self.epsilon = max(self.epsilon - epsilon_decay, final_epsilon)

    def eval(self, train_step):
        eval_env = make_env(scenario_name)
        episode_reward = []
        for episode in range(eval_episodes):
            eval_env.reset()
            self.policy.init_hidden(1)
            last_action = np.zeros((self.args.n_agents, self.args.n_actions))
            for step in range(max_episode_len):
                obs = eval_env.get_obs()
                actions = []
                for agent_id in range(self.args.n_agents):
                    action = self.policy.choose_action(obs[agent_id], last_action[agent_id], agent_id, 0)
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1
                    actions.append(np.int(action))
                    last_action[agent_id] = action_onehot

                obs_, reward, *_ = eval_env.step(actions)

                episode_reward.append(reward[0])
        eval_env.close()
        average_reward = np.array(episode_reward).mean()
        print('train step: %d, average reward: %f' % (train_step, average_reward))
        writer.add_scalar('average reward', average_reward)

    def run(self):
        train_steps = 0
        for epoch in range(n_epochs):
            episodes = []
            for episode in range(n_episodes):
                episode = self.generate_episode()
                episodes.append(episode)
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(n_train_steps):
                mini_batch = self.buffer.sample(min(len(self.buffer), batch_size))
                self.policy.learn(mini_batch)
                train_steps += 1

            self.eval(train_steps)
            if train_steps % save_point == 0:
                self.policy.save_model(train_steps)
        self.close()

    def close(self):
        self.env.close()
        writer.close()


def main():
    env = make_env(scenario_name)
    args = Argument(54, env.observation_space[0].shape[0], hidden_dim, env.action_space[0].n, len(env.agents))
    runner = Runner(env, args)
    runner.run()


if __name__ == '__main__':
    main()
