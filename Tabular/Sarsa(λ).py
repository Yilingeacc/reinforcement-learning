import numpy as np
from MazeEnv import MazeEnv

ALPHA = 0.1
EPSILON = 0.01
GAMMA = 0.9
LAMBDA = 0.9

NEGATIVE_INFINITY = float('-inf')


class SarsaLambda:
    def __init__(self, env):
        self.q_table = env.init_q_table()
        # Eligibility Traces
        self.e_table = np.zeros_like(self.q_table)

    def choose_action(self, state):
        if np.random.random() > EPSILON:
            action_value = self.q_table[state]
            action = np.random.choice(np.where(max(action_value) == action_value)[0])
        else:
            action = np.random.randint(0, 4)
            while self.q_table[state][action] == NEGATIVE_INFINITY:
                action = np.random.randint(0, 4)
        return action

    def learn(self, s, a, r, s_):
        a_ = self.choose_action(s_)
        delta = r + GAMMA * self.q_table[s_][a_] - self.q_table[s][a]
        self.e_table[s][a] += 1
        self.q_table += ALPHA * delta * self.e_table
        self.e_table *= GAMMA * LAMBDA
        return a_


if __name__ == '__main__':
    env = MazeEnv()

    RL = SarsaLambda(env)

    for episode in range(1000000):

        state = env.reset()
        action = RL.choose_action(state)

        while True:
            if episode and not episode % 100:
                env.render()

            state_, reward, done, info = env.step(action)

            action_ = RL.learn(state, action, reward, state_)

            if done:
                break

            state, action = state_, action_

    env.close()
