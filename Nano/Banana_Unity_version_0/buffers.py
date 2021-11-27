import torch
import numpy as np
import random
from collections import namedtuple, deque
from typing import Dict, List, Tuple
from sumtree import SumTree
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

class ReplayBuffer:
    """Fixed-size buffer to store experience objects."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Sample a batch of experiences from memory based on TD Error priority.
           Return indexes of sampled experiences in order to update their
           priorities after learning from them.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class NReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
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

class PrioritizedReplayBuffer:
    '''
    Implementation of prioritized experience replay. Adapted from:
    https://github.com/rlcode/per/blob/master/prioritized_memory.py
    '''

    def __init__(self, capacity):
        """
        initialize class . for the moment only capacity

        we use additional term ϵ in order to guarantee all transactions can be possibly sampled: pi=|δi|+ϵ, where ϵ is
        a small positive constant. value in e.
        The exponent  α  determines how much prioritization is used, with  α=0  corresponding to the uniform case.
        Value in a.
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
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.6
        self.beta_increment_per_sampling = 0.01

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        """Number of samples in memory

        Returns:
            [int] -- samples
        """

        return self.tree.n_entries

    def _get_priority(self, error):
        """Get priority based on error

        Arguments:
            error {float} -- TD error

        Returns:
            [float] -- priority
        """

        return (error + self.e) ** self.a

    def add(self, error, sample):
        """Add sample to memory

        Arguments:
            error {float} -- TD error
            sample {tuple} -- tuple of (state, action, reward, next_state, done)
        """

        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """Sample from prioritized replay memory

        Arguments:
            n {int} -- sample size

        Returns:
            [tuple] -- tuple of ((state, action, reward, next_state, done), idxs, is_weight)
        """

        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if p > 0:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        # Calculate importance scaling for weight updates
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)

        # Paper states that for stability always scale by 1/max w_i so that we only scale downwards
        is_weight /= is_weight.max()

        # Extract (s, a, r, s', done)
        batch = np.array(batch).transpose()
        states = np.vstack(batch[0])
        actions = list(batch[1])
        rewards = list(batch[2])
        next_states = np.vstack(batch[3])
        dones = batch[4].astype(int)

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idx, error):
        """Update the priority of a sample

        Arguments:
            idx {int} -- index of sample in the sumtree
            error {float} -- updated TD error
        """

        p = self._get_priority(error)
        self.tree.update(idx, p)