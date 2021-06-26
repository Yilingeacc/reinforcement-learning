import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# hyper-parameters
learning_rate = 1e-4  # learning rate
gamma = 0.99  # discount factor
tau = 2 ** -8  # soft update parameter
ratio_clip = 0.3
lambda_entropy = 0.01
lambda_gae_adv = 0.98

hidden_dim = 2 ** 9

batch_size = 2 ** 8
buffer_capacity = 2 ** 14
target_step = 2 ** 12
repeat_times = 2 ** 4

save_point = 2 ** 13

env_name = ''
model_name = env_name[:env_name.find('-')]


# orthogonal initialize
def layer_norm(layer, std=1.0, bias_const=1e-6):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)


class Buffer:
    def __init__(self, capacity, state_dim, action_dim, continuous=True):
        self.capacity = capacity
        self.mem_cntr = 0

        self.states = torch.empty((capacity, state_dim), dtype=torch.float32).cuda()
        self.actions = torch.empty((capacity, action_dim), dtype=torch.float32 if continuous else torch.int32).cuda()
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32).cuda()
        self.masks = torch.empty((capacity, 1), dtype=torch.float32).cuda()
        self.noises = torch.empty((capacity, action_dim), dtype=torch.float32).cuda()

    def store(self, state, action, reward, mask, noise):
        self.states[self.mem_cntr] = state
        self.actions[self.mem_cntr] = action
        self.rewards[self.mem_cntr] = reward
        self.masks[self.mem_cntr] = mask
        self.noises[self.mem_cntr] = noise

        self.mem_cntr += 1

    def sample_all(self):
        return self.states[:self.mem_cntr], self.actions[:self.mem_cntr], self.rewards[:self.mem_cntr], \
               self.masks[:self.mem_cntr], self.noises[:self.mem_cntr]

    def empty_buffer(self):
        self.mem_cntr = 0

    def __len__(self):
        return self.mem_cntr


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))

        self.a_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        layer_norm(self.net[-1], std=.1)

    def forward(self, state):
        return self.net(state).tanh()

    def get_action_noise(self, state):
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()

        noise = torch.randn_like(a_avg)
        # reparameterization trick:
        # a sample from πθ(·|s) is drawn
        # by computing a deterministic function of state, policy parameters, and independent noise.
        # squashed Gaussian policy:
        # aθ(s, ξ) = tanh(μθ(s) + σθ(s) * ξ), ξ ~ N(0, 1)
        action = a_avg + noise * a_std
        return action, noise

    def compute_log_prob(self, state, action):
        # X ~ N(μ, σ)  Gaussian distribution: f(x) = e^(-(x-μ)² / 2σ²) / (√(2π) σ)
        # log(f(x)) = -(x-μ)² / 2σ² - log(√(2π)) - log(σ)
        a_avg = self.net(state)
        a_std = self.a_std_log.exp()
        delta = ((a_avg - action) / a_std).pow(2) * 0.5
        log_prob = -(self.a_std_log + self.sqrt_2pi_log + delta)
        return log_prob.sum(1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

        layer_norm(self.net[-1], std=.5)

    def forward(self, state):
        return self.net(state)


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.act = Actor(state_dim, hidden_dim, action_dim).cuda()
        self.cri = Critic(state_dim, hidden_dim).cuda()

        # loss functions:
        # L1Loss: L1范数(绝对值偏差) 如MAE(Mean Absolute Error)
        # loss(x, y) = 1/n ( ∑|yi - f(xi)| )
        # 适用: 稀疏规则算子, 方便提取特征。很少使用
        # ---------------------------------------------------
        # L2Loss: L2范数(均方值偏差) 如MSE(Mean Square Error)
        # loss(x, y) = 1/n ( ∑(yi - f(xi))² )
        # 适用: 回归任务, 数值特征不大, 问题维度不高
        # ---------------------------------------------------
        # SmoothL1Loss: 平滑版的L1Loss, 对离群点更鲁棒
        #                    / .5 * (yi - f(xi))² if |yi - f(xi)| < 1
        # loss(x, y) = 1/n ∑ \ |yi - f(xi)| - 0.5 otherwise
        # 绝对值差<1时，用L2Loss, 其他时候用L1Loss的平移
        # 1.差别小时, 梯度不至于太大, 损失函数更圆滑
        # 2.差别大时, 梯度值足够小, 较稳定, 不至于梯度爆炸
        # 适用: 回归任务, 特征中有较大的数值, 适合大多数问题
        self.loss = nn.SmoothL1Loss()
        self.act_optimizer = optim.Adam(self.act.parameters(), lr=learning_rate)
        self.cri_optimizer = optim.Adam(self.cri.parameters(), lr=learning_rate)

        self.buffer = Buffer(buffer_capacity, state_dim, action_dim)

        self.learn_cntr = 0

    def choose_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32).cuda().detach()
        action, noise = self.act.get_action_noise(state)
        return action[0].cpu().numpy(), noise[0].cpu().numpy()

    def store_transition(self, state, action, reward, mask, noise):
        self.buffer.store(torch.tensor(state), torch.tensor(action), reward, mask, torch.tensor(noise))

    def explore_env(self, env):
        self.buffer.empty_buffer()
        step = 0
        while step < target_step:
            state = env.reset()
            while True:
                action, noise = self.choose_action(state)
                state_, reward, done, info = env.step(np.tanh(action))
                step += 1
                self.store_transition(state, action, reward, 0. if done else gamma, noise)
                if done:
                    break
                state = state_
        self.learn_cntr += step

    @staticmethod
    def compute_reward(buffer_len, rewards, masks, value):
        """
        GAE(Generalize Advantage Estimator): adjust the bias-variance trade-off
        """
        returns = torch.empty(buffer_len, dtype=torch.float32).cuda()
        advantages = torch.empty(buffer_len, dtype=torch.float32).cuda()

        pre_return = 0
        pre_advantage = 0
        for i in range(buffer_len - 1, -1, -1):
            returns[i] = rewards[i] + masks[i] * pre_return
            pre_return = returns[i]

            # pre = V' + λAt+1
            # At = r + γ * pre - V
            #    = r + γV' + γλAt+1 - V
            #    = δ0 + γλAt+1
            #    = ∑(l=0→∞)(γλ)^l δl
            advantages[i] = rewards[i] + masks[i] * pre_advantage - value[i]
            pre_advantage = value[i] + advantages[i] * lambda_gae_adv

        # standard score
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        return returns, advantages

    def learn(self):
        buffer_len = len(self.buffer)
        with torch.no_grad():
            states, actions, rewards, masks, noises = self.buffer.sample_all()
            value = torch.cat([self.cri(states[i:i + batch_size]) for i in range(0, states.size(0), batch_size)], dim=0)
            log_prob = -(noises.pow(2) * 0.5 + self.act.a_std_log + self.act.sqrt_2pi_log).sum(1)

            returns, advantages = self.compute_reward(buffer_len, rewards, masks, value)
            del rewards, masks, noises

        for i in range(int(repeat_times * buffer_len / batch_size)):
            indices = torch.randint(buffer_len, size=(batch_size,), requires_grad=False).cuda()

            state = states[indices]
            action = actions[indices]
            old_value = returns[indices]
            old_log_prob = log_prob[indices]
            adv = advantages[indices]

            new_log_prob = self.act.compute_log_prob(state, action)
            # probability ratio:
            #        πθ(at|st)
            # rt = ————————————— = e^(log(πθ(at|st)) - log(πθ_old(at|st)))
            #      πθ_old(at|st)
            ratio = (new_log_prob - old_log_prob).exp()

            surrogate1 = adv * ratio
            surrogate2 = adv * ratio.clamp(1 - ratio_clip, 1 + ratio_clip)
            # L^clip(θ) = Et[min( rt(θ)At, clip(rt(θ), 1 - ε, 1 + ε) )]
            surrogate = -torch.min(surrogate1, surrogate2).mean()

            # adding an entropy bonus to ensure sufficient exploration
            # entropy: S(pi) = -K * ∑ ( pi * log(pi) )
            entropy = (new_log_prob.exp() * new_log_prob).mean()

            val = self.cri(state).squeeze(1)
            loss = self.loss(val, old_value)

            # maximize objective united (minimize -(objective united) ):
            # Lt^(clip+vf+s)(θ) = Et[ Lt^clip(θ) - c1 * Lt^vf(θ) + c2 * S[πθ](st) ]
            # Lt^vf is a square-error loss: (Vθ(st) - Vt^target)²
            # S denotes an entropy bonus
            united = surrogate + loss / (old_value.std() + 1e-5) + entropy * lambda_entropy

            self.act_optimizer.zero_grad()
            self.cri_optimizer.zero_grad()
            united.backward()
            self.act_optimizer.step()
            self.cri_optimizer.step()

        print('save to disk:' + model_name + '-act' + str(self.learn_cntr) + '.pkl')
        print('save to disk:' + model_name + '-cri' + str(self.learn_cntr) + '.pkl')
        torch.save(self.act.state_dict(), model_name + '-act' + str(self.learn_cntr) + '.pkl')
        torch.save(self.cri.state_dict(), model_name + '-cri' + str(self.learn_cntr) + '.pkl')


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, hidden_dim, action_dim)

    while True:
        with torch.no_grad():
            ppo.explore_env(env)
        ppo.learn()
