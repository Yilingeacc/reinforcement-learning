import numpy as np
from MazeEnv import MazeEnv

ALPHA = 0.1
EPSILON = 0.01
GAMMA = 0.9

NEGATIVE_INFINITY = float('-inf')


class QLearning:
    def __init__(self, env):
        self.q1_table = env.init_q_table()
        self.q2_table = env.init_q_table()

    def choose_action(self, state):
        if np.random.random() > EPSILON:
            action_value = self.q1_table[state] + self.q2_table[state]
            action = np.random.choice(np.where(max(action_value) == action_value)[0])
        else:
            action = np.random.randint(0, 4)
            while self.q1_table[state][action] == NEGATIVE_INFINITY:
                action = np.random.randint(0, 4)
        return action

    def learn(self, s, a, r, s_):
        if np.random.random() < 0.5:
            self.q1_table[s][a] += \
                ALPHA * (r + GAMMA * self.q2_table[s_][np.argmax(self.q1_table[s_])] - self.q1_table[s][a])
        else:
            self.q2_table[s][a] += \
                ALPHA * (r + GAMMA * self.q1_table[s_][np.argmax(self.q2_table[s_])] - self.q2_table[s][a])


if __name__ == '__main__':
    env = MazeEnv()

    RL = QLearning(env)

    for episode in range(1000000):

        state = env.reset()

        while True:
            if episode and not episode % 100:
                env.render()

            action = RL.choose_action(state)

            state_, reward, done, info = env.step(action)

            RL.learn(state, action, reward, state_)

            if done:
                break

            state = state_

    env.close()
