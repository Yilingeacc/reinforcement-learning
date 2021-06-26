import numpy as np
from MazeEnv import MazeEnv

ALPHA = 0.1
EPSILON = 0.01
GAMMA = 0.9
n = 3

NEGATIVE_INFINITY = float('-inf')


class nStepSarsa:
    def __init__(self, env):
        self.q_table = env.init_q_table()

    def choose_action(self, state):
        if np.random.random() > EPSILON:
            action_value = self.q_table[state]
            action = np.random.choice(np.where(max(action_value) == action_value)[0])
        else:
            action = np.random.randint(0, 4)
            while self.q_table[state][action] == NEGATIVE_INFINITY:
                action = np.random.randint(0, 4)
        return action

    def learn(self, state, action, discounted_reward, state_=None, action_=None):
        action_value = discounted_reward + (pow(GAMMA, n) * self.q_table[state_][action_] if state_ else 0)
        self.q_table[state][action] += ALPHA * (action_value - self.q_table[state][action])


if __name__ == '__main__':
    env = MazeEnv()

    RL = nStepSarsa(env)

    for episode in range(10000):

        T = 0x7fffffff  # terminal time
        step = 0  # current step
        tao = -n  # update step (if τ < 0, no update)
        discount = 1
        discounted_reward = 0
        state_, action_ = 0, 0

        state = env.reset()
        action = RL.choose_action(state)
        states, actions, rewards = [state], [action], []
        while tao < T - 1:
            if episode and not episode % 100:
                env.render()

            # interact with environment
            if step < T:
                state_, reward, done, info = env.step(action)
                discounted_reward += discount * reward
                rewards.append(reward)
                states.append(state_)
                if done:
                    T = step + 1
                else:
                    action_ = RL.choose_action(state_)
                    actions.append(action_)

            tao = step - n + 1
            if tao < 0:
                discount *= GAMMA
            else:
                update_state, update_action = states.pop(0), actions.pop(0)
                if tao + n < T:
                    RL.learn(update_state, update_action, discounted_reward, state_, action_)
                else:
                    RL.learn(update_state, update_action, discounted_reward)
                discounted_reward = (discounted_reward - rewards.pop(0)) / GAMMA
            state, action = state_, action_
            step += 1

    env.close()
