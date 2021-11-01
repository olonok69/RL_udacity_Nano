import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.1
        self.alpha = 0.3
        self.gamma = 0.8
        self.episode = 1

    def select_action(self, state, i_episode, num_episodes):
        if i_episode != self.episode and i_episode % (num_episodes * 0.0005) == 0:
            self.epsilon -= 50 / num_episodes

        if i_episode != self.episode and i_episode % (num_episodes * 0.005) == 0:
            self.alpha -= 20 / num_episodes
            self.epsilon = 0.8

        if i_episode != self.episode and i_episode % (num_episodes * 0.05) == 0:
            self.gamma -= 20 / num_episodes
            self.epsilon = 0.8
            self.alpha = 0.8

        return np.argmax(self.Q[state])

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, env, Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

#     def step(self, state, action, reward, next_state, done):
#         """
#         Q-learning formula
#         """
#         next_action = np.argmax(self.Q[state])
#         self.Q[state][action] = self.Q[state][action] + (self.alpha * (reward + \
#             (self.gamma * np.max(self.Q[next_state])) - self.Q[state][action]))