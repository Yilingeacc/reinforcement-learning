import numpy as np
from MazeEnv import MazeEnv

ALPHA = 0.1
INIT_EPSILON = 1
FINAL_EPSILON = 0.01
EPSILON_DECAY = 0.01
GAMMA = 0.9

NEGATIVE_INFINITY = float('-inf')


class MonteCarlo:
    def __init__(self, env):
        self.q_table = env.init_q_table()
        self.n_table = np.zeros_like(self.q_table, dtype=np.int32)
        self.epsilon = INIT_EPSILON

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action_value = self.q_table[state]
            action = np.random.choice(np.where(max(action_value) == action_value)[0])
        else:
            action = np.random.randint(0, 4)
            while self.q_table[state][action] == NEGATIVE_INFINITY:
                action = np.random.randint(0, 4)
        return action

    def learn(self, states, actions, rewards):
        returns = 0
        for state, action, reward in zip(reversed(states), reversed(actions), reversed(rewards)):
            returns = reward + GAMMA * returns
            self.n_table[state][action] += 1
            self.q_table[state][action] += (returns - self.q_table[state][action]) / self.n_table[state][action]
        self.update_epsilon()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - EPSILON_DECAY, FINAL_EPSILON)


if __name__ == '__main__':
    env = MazeEnv()

    RL = MonteCarlo(env)

    for episode in range(1000000):
        state = env.reset()
        sa_set = set()
        states, actions, rewards = [], [], []

        while True:
            if episode and not episode % 100:
                env.render()

            action = RL.choose_action(state)

            state_, reward, done, info = env.step(action)

            if (state, action) not in sa_set:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                sa_set.add((state, action))

            if done:
                break

            state = state_

        RL.learn(states, actions, rewards)

    env.close()
