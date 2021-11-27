import torch
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Deque
from src.sumtree import SumTree
from src.segment_tree import MinSegmentTree, SumSegmentTree
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    '''
    Implementation of prioritized experience replay. Adapted from:
    https://github.com/rlcode/per/blob/master/prioritized_memory.py
    '''

    def __init__(self, capacity):
        """
        initialize class . for the moment only capacity

        we use additional term ϵ in order to guarantee all transactions can be possibly sampled: pi=|δi|+ϵ, where ϵ is
        a small positive constant. value in e. The exponent  α  determines how much prioritization is
        used, with  α=0  corresponding to the uniform case.
        To remove correlation of observations, it uses uniformly random sampling from the replay buffer.
        Prioritized replay introduces bias because it doesn't sample experiences uniformly at random due to the
        sampling proportion correspoding to TD-error. We can correct this bias by using importance-sampling (IS)
        weights wi=(1N⋅1P(i))β that fully compensates for the non-uniform probabilities  P(i)  if  β=1 .
        These weights can be folded into the Q-learning update by using  wiδi  instead of  δi .
        In typical reinforcement learning scenarios, the unbiased nature of the updates is most important near
        convergence at the end of training, We therefore exploit the flexibility of annealing the amount of
        importance-sampling correction over time, by defining a schedule on the exponent  β  that reaches 1 only at
        the end of learning.
        Here instead to use and schedule we define a beta equal to a constant and an increment per sampling also
        constant. attributes beta and beta_increment_per_sampling
        :param capacity:
        """
        self.e = 0.01  # constant to add to TD error
        self.a = 0.6  # determines how much prioritization is  used, with  α=0  corresponding to the
                        # uniform case.
        self.beta = 0.6  # to correct bias by using importance-sampling
        self.beta_increment_per_sampling = 0.01  # increment of beta per each step

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        """
        Number of samples in memory
        Returns:
            [int] -- samples
        """
        return self.tree.n_entries

    def _get_priority(self, error):
        """
        Get priority based on error
        Arguments:
            error {float} -- TD error
        Returns:
            [float] -- priority
        """
        # error is equal to error + epsilon constant , elevated to the power of alpha
        return (error + self.e) ** self.a

    def add(self, error, sample):
        """Add sample to memory
        Arguments:
            error {float} -- TD error
            sample {tuple} -- tuple of (state, action, reward, next_state, done)
        """
        p = self._get_priority(error) # get priority of this sample
        self.tree.add(p, sample) # add the sample to the sumTree

    def sample(self, n):
        """
        Sample from prioritized replay memory
        Arguments:
            n {int} -- sample size
        Returns:
            [tuple] -- tuple of ((state, action, reward, next_state, done), indexes, weights)
        """
        batch = []
        indexes = []
        # total return Value of root node
        segment = self.tree.total() / n
        priorities = []
        # calculate beta. max value will be 1 , but we take the min between 1 and current calculation
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            # random value to decide which side of the tree explore
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if p > 0: # add only if weight is hight than 0
                priorities.append(p)
                batch.append(data)
                indexes.append(idx)

        # Calculate importance scaling for weight updates
        sampling_probabilities = priorities / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        # Paper states that for stability always scale by 1/max w_i so that we only scale downwards
        weights /= weights.max()

        # Extract (s, a, r, s', done)
        batch = np.array(batch).transpose()
        states = np.vstack(batch[0])
        actions = list(batch[1])
        rewards = list(batch[2])
        next_states = np.vstack(batch[3])
        dones = batch[4].astype(int)

        return (states, actions, rewards, next_states, dones), indexes, weights


    def update(self, idx, error):
        """
        Update the priority of a sample
        Arguments:
            idx {int} -- index of sample in the sumtree
            error {float} -- updated TD error
        """
        p = self._get_priority(error)
        self.tree.update(idx, p)

class NPReplayBuffer:
    """
    A simple numpy replay buffer.
    Reinforcement learning agent stores the experiences consecutively in the buffer, so adjacent ($s, a, r, s'$)
    transitions stored are highly likely to have correlation. To remove this, the agent samples experiences uniformly
    at random from the pool of stored samples $\big( (s, a, r, s') \sim U(D) \big)$.
    """

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        """

        """

        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


class N_Steps_ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            n_step: int = 3,
            gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rew: float,
            next_obs: np.ndarray,
            done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(
            self.size, size=self.batch_size, replace=False
        )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )

    def sample_batch_from_idxs(
            self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )

    def _get_n_step_info(
            self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class N_Steps_PrioritizedReplayBuffer(N_Steps_ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(N_Steps_PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight