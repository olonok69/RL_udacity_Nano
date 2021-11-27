import os
import datetime
import itertools
import collections
import numpy as np


import gym
import tensorflow as tf

import helpers
import importlib
importlib.reload(helpers)

def policy(st, model, eps,env):
    if np.random.rand() > eps:
        q_values = model.eval(np.stack([st]))
        return np.argmax(q_values)
    else:
        return env.action_space.sample()

def q_learning(env, frames, gamma, eps_decay_steps, eps_target,
               batch_size, model, mem, start_step=0,
               callback=None, trace=None, render=False):
    """Q-Learning, supprots resume

    Note: If resuming, all parameters should be identical to original call, with
        exception of 'start_step' and 'frames'.

    Params:
        env - environment
        frames - number of time steps to execute
        gamma - discount factor [0..1]
        eps_decay_steps - decay epsilon-greedy param over that many time steps
        eps_target - epsilon-greedy param after decay
        batch_size - neural network batch size from memory buffer
        model      - function approximator, already initialised, with methods:
                     eval(state, action) -> float
                     train(state, target) -> None
        mem - memory reply buffer
        start_step - if continuning, pass in return value (tts_) here
        callback - optional callback to execute
        trace - this object handles data logging, plotting etc.
        render - render openai gym environment?
    """

    def eps_schedule(tts, eps_decay_steps, eps_target):
        if tts > eps_decay_steps:
            return eps_target
        else:
            eps_per_step_change = ( 1 -eps_target) / eps_decay_steps
            return 1.0 - tts * eps_per_step_change


    assert len(mem) >= batch_size

    tts_ = start_step                        # total time step
    for _ in itertools.count():              # count from 0 to infinity

        S = env.reset()
        episode_reward = 0                   # purely for logging
        if render: env.render()

        for t_ in itertools.count():         # count from 0 to infinity

            eps = eps_schedule(tts_, eps_decay_steps, eps_target)

            A = policy(S, model, eps, env)

            S_, R, done, _ = env.step(A)
            episode_reward += R
            if render: env.render()

            mem.append(S, A, R, S_, done)

            if callback is not None:
                callback(tts_, t_, S, A, R, done, eps, episode_reward, model, mem, trace, env)

            states, actions, rewards, n_states, dones, _ = mem.get_batch(batch_size)
            targets = model.eval(n_states)
            targets = rewards + gamma * np.max(targets, axis=-1)
            targets[dones] = rewards[dones]  # return of next-to-terminal state is just R
            model.train(states, actions, targets)

            S = S_

            tts_ += 1
            if tts_ >= start_step + frames:
                return tts_                  # so we can pick up where we left

            if done:
                break


def evaluate(env, model, frames=None, episodes=None, eps=0.0, render=False):
    assert frames is not None or episodes is not None

    total_reward = 0

    tts_ = 0  # total time step
    for e_ in itertools.count():  # count from 0 to infinity
        if episodes is not None and e_ >= episodes:
            return total_reward

        S = env.reset()
        if render: env.render()

        for t_ in itertools.count():  # count from 0 to infinity

            A = policy(S, model, eps, env)

            S_, R, done, _ = env.step(A)
            total_reward += R
            if render: env.render()

            S = S_

            tts_ += 1
            if frames is not None and tts_ >= frames:
                return

            if done:
                break


def mem_fill(env, mem, steps=None, episodes=None, render=False):
    # Fill memory buffer using random policy
    tts_ = 0
    for e_ in itertools.count():
        if episodes is not None and e_ >= episodes:
            return

        S = env.reset()
        if render: env.render()

        for t_ in itertools.count():

            A = env.action_space.sample()  # random policy
            S_, R, done, _ = env.step(A)
            if render: env.render()

            mem.append(S, A, R, S_, done)

            S = S_

            tts_ += 1
            if steps is not None and tts_ >= steps:
                return

            if done:
                break

def callback(total_time_step, tstep, st, act, rew_, done_,
             eps, ep_reward, model, memory, trace, env):
    """
    Called from gradient_MC after every episode.

    Params:
        episode [int] - episode number
        tstep [int]   - timestep within episode
        model [obj]   - function approximator
        trace [list]  - list to write results to
    """

    assert total_time_step == trace.total_tstep

    trace.tstep = tstep

    trace.states.append(st)
    trace.actions.append(act)
    trace.rewards.append(rew_)
    trace.dones.append(done_)
    trace.epsilons.append(eps)

    if done_:
        trace.ep_rewards[total_time_step] = ep_reward
        trace.last_ep_reward = ep_reward

    #
    #   Print, Evaluate, Plot
    #
    if (trace.eval_every is not None) and (trace.total_tstep % trace.eval_every == 0):

        last_ep_rew = trace.last_ep_reward
        reward_str = str(round(last_ep_rew, 3)) if last_ep_rew is not None else 'None'
        print(f'wall: {datetime.datetime.now().strftime("%H:%M:%S")}   '
              f'ep: {len(trace.ep_rewards):3}   tstep: {tstep:4}   '
              f'total tstep: {trace.total_tstep:6}   '
              f'eps: {eps:5.3f}   reward: {reward_str}   ')

        if len(st) == 2:
            # We are working with 2D environment,
            # eval. Q-Value function across whole state space
            q_arr = helpers.eval_state_action_space(model, env, split=[128, 128])
            trace.q_values[trace.total_tstep] = q_arr
        else:
            # Environment is not 2D,
            # eval. on pre-defined random sample of states
            if trace.test_states is not None:
                y_hat = model.eval(trace.test_states)
                trace.q_values[trace.total_tstep] = y_hat

        if trace.enable_plotting:
            helpers.plot_all(env, model, memory, trace)
            print('■' * 80)

    trace.total_tstep += 1



class TFNeuralNet():
    def __init__(self, nb_in, nb_hid_1, nb_hid_2, nb_out, lr):
        self.nb_in = nb_in
        self.nb_hid_1 = nb_hid_1
        self.nb_hid_2 = nb_hid_2
        self.nb_out = nb_out
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()

        self._x = tf.compat.v1.placeholder(name='xx', shape=[None, nb_in], dtype=tf.float32)
        self._y = tf.compat.v1.placeholder(name='yy', shape=[None, nb_out], dtype=tf.float32)

        self._h_hid_1 = tf.compat.v1.layers.dense(self._x, units=nb_hid_1,
                                        activation=tf.nn.relu, name='Hidden_1')
        self._h_hid_2 = tf.compat.v1.layers.dense(self._h_hid_1, units=nb_hid_2,
                                        activation=tf.nn.relu, name='Hidden_2')
        self._y_hat = tf.compat.v1.layers.dense(self._h_hid_2, units=nb_out,
                                      activation=None, name='Output')
        self._loss = tf.losses.mean_squared_error(self._y, self._y_hat)

        self._optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=lr)
        self._train_op = self._optimizer.minimize(self._loss)

        self._sess = tf.compat.v1.Session()
        self._sess.run(tf.compat.v1.global_variables_initializer())

    def backward(self, x, y):
        assert x.ndim == y.ndim == 2
        _, y_hat, loss = self._sess.run([self._train_op, self._y_hat, self._loss],
                                        feed_dict={self._x: x, self._y: y})
        return y_hat, loss

    def forward(self, x):
        return self._sess.run(self._y_hat, feed_dict={self._x: x})

    def save(self, filepath):
        saver = tf.compat.v1.train.Saver()
        saver.save(self._sess, filepath)

    def load(self, filepath):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self._sess, filepath)


class TFFunctApprox():

    def __init__(self, model, st_low, st_high, rew_mean, rew_std, nb_actions):
        """Q-function approximator using Keras model

        Args:
            model: TFNeuralNet model
        """
        st_low = np.array(st_low)
        st_high = np.array(st_high)
        self._model = model

        assert st_low.ndim == 1 and st_low.shape == st_high.shape

        if len(st_low) != model.nb_in:
            raise ValueError('Input shape does not match state_space shape')

        if nb_actions != model.nb_out:
            raise ValueError('Output shape does not match action_space shape')

        # normalise inputs
        self._offsets = st_low + (st_high - st_low) / 2
        self._scales = 1 / ((st_high - st_low) / 2)

        self._rew_mean = rew_mean
        self._rew_std = rew_std

    def eval(self, states):
        assert isinstance(states, np.ndarray)
        assert states.ndim == 2

        inputs = (states - self._offsets) * self._scales

        y_hat = self._model.forward(inputs)

        # return y_hat
        return y_hat * self._rew_std + self._rew_mean

    def train(self, states, actions, targets):

        assert isinstance(states, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(targets, np.ndarray)
        assert states.ndim == 2
        assert actions.ndim == 1
        assert targets.ndim == 1
        assert len(states) == len(actions) == len(targets)

        targets = (targets - self._rew_mean) / self._rew_std  # normalise

        inputs = (states - self._offsets) * self._scales
        all_targets = self._model.forward(inputs)  # this should normalised already
        all_targets[np.arange(len(all_targets)), actions] = targets
        self._model.backward(inputs, all_targets)


class Memory:
    """Circular buffer for DQN memory reply. Fairly fast."""

    def __init__(self, max_len, state_shape, state_dtype):
        """
        Args:
            max_len: maximum capacity
        """
        assert isinstance(max_len, int)
        assert max_len > 0

        self.max_len = max_len  # maximum length
        self._curr_insert_ptr = 0  # index to insert next data sample
        self._curr_len = 0  # number of currently stored elements

        state_arr_shape = [max_len] + list(state_shape)

        self._hist_St = np.zeros(state_arr_shape, dtype=state_dtype)
        self._hist_At = np.zeros(max_len, dtype=int)
        self._hist_Rt_1 = np.zeros(max_len, dtype=float)
        self._hist_St_1 = np.zeros(state_arr_shape, dtype=state_dtype)
        self._hist_done_1 = np.zeros(max_len, dtype=bool)

    def append(self, St, At, Rt_1, St_1, done_1):
        """Add one sample to memory, override oldest if max_len reached.

        Args:
            St [np.ndarray]   - state
            At [int]          - action
            Rt_1 [float]      - reward
            St_1 [np.ndarray] - next state
            done_1 [bool]       - next state terminal?
        """
        self._hist_St[self._curr_insert_ptr] = St
        self._hist_At[self._curr_insert_ptr] = At
        self._hist_Rt_1[self._curr_insert_ptr] = Rt_1
        self._hist_St_1[self._curr_insert_ptr] = St_1
        self._hist_done_1[self._curr_insert_ptr] = done_1

        if self._curr_len < self.max_len:  # keep track of current length
            self._curr_len += 1

        self._curr_insert_ptr += 1  # increment insertion pointer
        if self._curr_insert_ptr >= self.max_len:  # roll to zero if needed
            self._curr_insert_ptr = 0

    def __len__(self):
        """Number of samples in memory, 0 <= length <= max_len"""
        return self._curr_len

    def get_batch(self, batch_len):
        """Sample batch of data, with repetition

        Args:
            batch_len: nb of samples to pick

        Returns:
            states, actions, rewards, next_states, next_done, indices
            Each returned element is np.ndarray with length == batch_len
        """
        assert self._curr_len > 0
        assert batch_len > 0

        indices = np.random.randint(  # randint much faster than np.random.sample
            low=0, high=self._curr_len, size=batch_len, dtype=int)

        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones_1 = np.take(self._hist_done_1, indices, axis=0)

        return states, actions, rewards_1, states_1, dones_1, indices

    def pick_last(self, nb):
        """Pick last nb elements from memory

        Returns:
            states, actions, rewards, next_states, done_1, indices
            Each returned element is np.ndarray with length == batch_len
        """
        assert nb <= self._curr_len

        start = self._curr_insert_ptr - nb  # inclusive
        end = self._curr_insert_ptr  # not inclusive
        indices = np.array(range(start, end), dtype=int)  # indices to pick, can be neg.
        indices[indices < 0] += self._curr_len  # loop negative to positive

        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones_1 = np.take(self._hist_done_1, indices, axis=0)

        return states, actions, rewards_1, states_1, dones_1, indices


class WrapFrameSkip():
    def __init__(self, env, frameskip):
        assert frameskip >= 1
        self._env = env
        self._frameskip = frameskip
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        sum_rew = 0
        for _ in range(self._frameskip):
            obs, rew, done, info = self._env.step(action)
            sum_rew += rew
            if done: break
        return obs, sum_rew, done, info

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()


class Trace():
    def __init__(self, eval_every, render=False, test_states=None, state_labels=None):
        if test_states is not None:
            assert test_states.ndim == 2

        self.enable_plotting = False

        self.eval_every = eval_every
        self.test_states = test_states
        self.state_labels = state_labels

        self.tstep = 0
        self.total_tstep = 0

        self.q_values = collections.OrderedDict()
        self.ep_rewards = collections.defaultdict(float)
        self.last_ep_reward = None

        self.states = []
        self.actions = []
        self.rewards = []  # t+1
        self.dones = []  # t+1
        self.epsilons = []


