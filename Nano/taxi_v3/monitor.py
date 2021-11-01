from collections import deque
import sys
import math
import numpy as np


def interact(env, agent, num_episodes=100000, window=100):
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes + 1):
        # begin the episode
        state = env.reset()
        # initialize the sampled reward
        samp_reward = 0
        while True:

            # get epsilon-greedy action probabilities
            policy_s = agent.epsilon_greedy_probs(env, agent.Q[state], i_episode)
            # pick next action A
            action = np.random.choice(np.arange(agent.nA), p=policy_s)
            # take action A, observe R, S'

            # agent performs the selected action
            next_state, reward, done, info = env.step(action)

            # update the sampled reward
            samp_reward += reward
            # update the state (s <- s') to next time step
            # update Q
            agent.Q[state][action] = agent.update_Q(agent.Q[state][action], np.max(agent.Q[next_state]), \
                                                    reward, agent.alpha, agent.gamma)

            state = next_state
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        print("\rEpisode {} || Best average reward {:.3f}".format(
            i_episode, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    return avg_rewards, best_avg_reward