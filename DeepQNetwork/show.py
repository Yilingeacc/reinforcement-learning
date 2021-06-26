import gym
import torch
import torch.nn as nn

hidden_dim = 2 ** 8

env_name = 'MountainCar-v0'
algo = 'DQN'

learn_steps = 39936


class Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        if algo == 'DQN':
            self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, action_dim))
        elif algo == 'DoubleDQN':
            self.state_net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.q1_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))
            self.q2_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, action_dim))

        elif algo == 'DuelingDQN':
            self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.val_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, 1))
            self.adv_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, action_dim))
        else:
            self.state_net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.val_net1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
            self.adv_net1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, action_dim))
            self.val_net2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, 1))
            self.adv_net2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, action_dim))

    def forward(self, state):
        if algo == 'DQN':
            return self.net(state)
        elif algo == 'DoubleDQN':
            state_temp = self.state_net(state)
            return self.q1_net(state_temp)
        elif algo == 'DuelingDQN':
            temp = self.net(state)
            s_val = self.val_net(temp)
            q_adv = self.adv_net(temp)
            return s_val + q_adv - q_adv.mean(dim=1, keepdim=True)
        else:
            temp = self.state_net(state)
            s_val = self.val_net1(temp)
            q_adv = self.adv_net1(temp)
            return s_val + q_adv - q_adv.mean(dim=1, keepdim=True)


class DeepQNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.learn_cntr = learn_steps

        self.eval_net = Net(state_dim, hidden_dim, action_dim).cuda()
        self.eval_net.load_state_dict(torch.load(env_name[:env_name.find('-')] + str(self.learn_cntr) + '.pkl'))

    def choose_action(self, state):
        state = torch.tensor((state,), dtype=torch.float32).cuda()
        actions = self.eval_net.forward(state)
        action = torch.argmax(actions).item()
        return action


if __name__ == '__main__':
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    DQN = DeepQNetwork(state_dim, hidden_dim, action_dim)

    for episode in range(1000000):
        state = env.reset()

        done = False

        while True:
            env.render()

            action = DQN.choose_action(state)

            state_, reward, done, info = env.step(action)

            if done:
                break

            state = state_

    env.close()
