import copy
from collections import namedtuple
import random
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import pickle

def load_env(worker_id, base_port, file="./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe",
             grap=True, train=True):
    """
    load Unity Environment
    :param worker_id: ID env
    :param base_port: communications port with unity agent
    :param file: Unity executable
    :param grap: If to show graphs or not. Typically in training I hide graphics
    :param train: if train mode or test mode
    """
    # load environtment
    env = UnityEnvironment(file_name=file, worker_id=worker_id, base_port=base_port, no_graphics=grap)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # reset the environment
    env_info = env.reset(train_mode=train)[brain_name]
    # number agents
    n_agents = len(env_info.agents)
    state = env_info.vector_observations[0]
    state_size = len(state)
    return env , brain_name, brain, action_size, env_info, state, state_size, n_agents

class OUNoise:
    """
    Ornstein-Uhlenbeck process.
    credit to https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class GaussianNoise:
    """
    Gaussian Noise.
    credits for https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        action_dim: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """
        Initialize.
        """
        self.action_dim = action_dim
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, t: int = 0) -> float:
        """
        Get gaussian noise to add to an action. Typically as is configured a random number between 0 and max_sigma
        :param t: parameter to implement decay
        :type t: int , default 0
        :return: random normal between 0 and max_sigma
        :rtype: float
        """

        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(0, sigma, size=self.action_dim)

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """

    def __init__(self, device, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.DEVICE = device
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory and push it to torch Device

        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.DEVICE)

        return (states, actions, rewards, next_states, dones)

    def sample2(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for exp in experiences:
            states.append(exp.state.squeeze(0))
            actions.append(exp.action.squeeze(0))
            rewards.append(exp.reward)
            dones.append(exp.done)
            next_states.append(exp.next_state.squeeze(0))

        states_v = torch.Tensor(np.array(states, dtype=np.float32)).to(self.DEVICE)
        actions_v = torch.Tensor(np.array(actions, dtype=np.float32)).to(self.DEVICE)
        rewards_v = torch.Tensor(np.array(rewards, dtype=np.float32)).to(self.DEVICE)
        next_states_v = torch.Tensor(np.array(next_states, dtype=np.float32)).to(self.DEVICE)
        dones_v = torch.ByteTensor(dones).to(self.DEVICE)

        return states_v, actions_v, rewards_v, next_states_v, dones_v
    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)

class ReplayBuffer_2:
    """Fixed-size buffer to store experience tuples."""

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
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Batcher:
    '''
    Select random batches from the dataset passed
    author: Shangtong Zhang
    source: https://bit.ly/2yrYoHy
    '''

    def __init__(self, batch_size, data):
        '''
        Initialize a Batcher object. sala all parameters as attributes
        :param batch_size: integer. The size of each batch
        :data: list. Dataset to be batched
        '''
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        '''
        start the counter
        '''
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        '''check if the dataset has been consumed'''
        return self.batch_start >= self.num_entries

    def next_batch(self):
        '''select the next batch'''
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size,
                             self.num_entries)
        return batch

    def shuffle(self):
        '''shuffle the datset passed in the constructor'''
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]

class Memory:
    def __init__(self, size, batch_size):
        self.keys = ['advantages', 'states', 'log_probs_old', 'returns', 'actions']
        self.size = size
        self.batch_size = batch_size
        self.reset()

    def add(self, data):
        for k, vs in data.items():
            for i in range(vs.size()[0]):
                getattr(self, k).append(vs[i].unsqueeze(0))

    def reset(self):
        for key in self.keys:
            setattr(self, key, deque(maxlen=self.size))

    def categories(self, keys=['advantages', 'states', 'log_probs_old', 'returns', 'actions']):
        data = [list(getattr(self, k))[:] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)

    def sample(self):
        batch_indices = np.random.permutation(len(getattr(self, self.keys[0])))[:len(getattr(
            self, self.keys[0])) // self.batch_size * self.batch_size].reshape(-1, self.batch_size)
        return batch_indices

class ReplayMemory:
    def __init__(self, seed, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.seed = random.seed(seed)
        self.seed1 = np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        ''' 
        https://github.com/quantumiracle/SOTA-RL-Algorithms/blob/master/sac_v2_multiprocess.py
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Environment():
    """Learning Environment."""

    def __init__(self, file_name="./envs/Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe", no_graphics=True):
        """Initialize parameters and build model.
        Params
        ======
            file_name (string): unity environment file
            no_graphics (boolean): Start environment with graphics
        """
        self.env = UnityEnvironment(file_name=file_name, no_graphics=no_graphics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.reset()
        self.action_space_size = self.brain.vector_action_space_size
        self.state_size = len(self.info.vector_observations[0])
        self.num_agents = len(self.info.agents)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.env.close()

    def reset(self, train_mode=False):
        self.info = self.env.reset(train_mode=train_mode)[self.brain_name]
        return self.info

    def step(self, action):
        self.info = self.env.step(action)[self.brain_name]
        return self.info

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def plot_scores(scores , algo, num_episodes, mode):
    """
    plot scores and save to disk
    :param scores: list of scores during training
    :param algo: type algorithm
    :param mode: Mode that use the algorithm or name of the algorithm
    :return:
    """
    # plot the scores
    text = ""
    if algo == "1":
        text= "DDPG Agent"
    elif algo == "2":
        text = "TD3 Agent"
    elif algo == "3":
        text = f"TD3 Agent 4 DQN with {mode}"
    elif algo == "4":
        text = f"TD3 Agent 4 DQN with {mode}"
    elif algo == "5":
        text = "D4PG Agent"
    elif algo == "6":
        text = f"TD3 Agent 4 DQN with {mode}"
    elif algo == "9":
        text = f"PPO Agent {mode}"

    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/scores_{mode}_{algo}.jpg')
    return

def plot_losses(losses , algo, num_episodes, type, mode):
    """
    plot scores and save to disk
    :param scores: list of scores during training
    :param algo: type algorithm
    :param mode: Mode that use the algorithm or name of the algorithm
    :return:
    """
    # plot the scores
    text = ""
    if algo == "1":
        text= "DDPG Agent"
    elif algo == "2":
        text = "TD3 Agent"
    elif algo == "3":
        text = f"TD3 Agent 4 DQN with {mode}"
    elif algo == "4":
        text = f"TD3 Agent 4 DQN with {mode}"
    elif algo == "5":
        text = "D4PG Agent"
    elif algo == "6":
        text =  f"TD3 Agent 4 DQN with {mode}"
    elif algo == "9":
        text = f"PPO Agent {mode}"

    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'Algo {text} Number episodes {num_episodes}')
    plt.plot(np.arange(len(losses)), losses)
    plt.ylabel(f'Loss_{type}')
    plt.xlabel('Episode #')
    plt.savefig(f'./images/losses_{mode}_{type}_{algo}.jpg')
    return

def save_pickle(outputs, scores, loss_actor, loss_critic, mode, fname, algo, times):
    """
    save outputs to pickle
    :param outputs: pickle file
    :type outputs:  pickle file
    :param scores: vector of scores
    :type scores: vector of float
    :param loss_actor: loss actor each episode
    :type loss_actor: vector of float
    :param loss_critic: loss critic each episodes
    :type loss_critic:  vector of floats
    :param mode: Mode that use the algorithm or name of the algorithm
    :type mode: string
    :param fname: file name
    :type fname: string
    :param algo: number type algo. come from cmd parameters
    :type algo: string
    :param times: time consumed if training for this algo in seconds
    :type times: float
    :return:
    :rtype:
    """
    algor= algo + "_" + mode
    if not(algor in outputs.keys()):
        outputs[str(algor)]= {}

    outputs[str(algor)]['scores']= scores
    outputs[str(algor)]['mode'] = mode
    outputs[str(algor)]['critic_loss'] = loss_critic
    outputs[str(algor)]['actor_loss'] = loss_actor
    outputs[str(algor)]['time'] = times

    with open(fname, 'wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return

def plot_scores_training_all():
    """
    plot all scores 2000 episodes
    """
    with open('outputs/outcomes.pkl', 'rb') as handle:
        data = pickle.load(handle)
    labels = []
    text = f"DDPG Agent ({max(data['1_DDPG']['scores']).round(2)})"
    labels.append("DDPG Agent ")
    num_episodes = "1000"
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm scores after solve environment score 35 average')
    plt.axhline(y=30, color='r', linestyle='dotted')
    plt.plot(np.arange(len(data['1_DDPG']['scores'])), data['1_DDPG']['scores'], label=text)
    text = f"TD3 (Twined Delayed DDPG) ({max(data['2_TD3']['scores']).round(2)})"
    labels.append("TD3 (Twined Delayed DDPG)")
    plt.plot(np.arange(len(data['2_TD3']['scores'])), data['2_TD3']['scores'], label=text)
    text = f"TD3 Agent with 4 DQN and min loss ({max(data['3_min']['scores']).round(2)})"
    labels.append("TD3 Agent with 4 DQN and min loss")
    plt.plot(np.arange(len(data['3_min']['scores'])), data['3_min']['scores'], label=text)
    text = f"TD3 Agent with 4 DQN and mean loss ({max(data['4_mean']['scores']).round(2)})"
    labels.append("TD3 Agent with 4 DQN and mean loss")
    plt.plot(np.arange(len(data['4_mean']['scores'])), data['4_mean']['scores'], label=text)
    text = f"TD3 Agent with 4 DQN and median loss ({max(data['6_median']['scores']).round(2)})"
    labels.append("TD3 Agent with 4 DQN and median loss")
    plt.plot(np.arange(len(data['6_median']['scores'])), data['6_median']['scores'], label=text)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    title = "Algorithm and Max Score"
    plt.legend(title=title)
    plt.savefig(f'images/scores_all.jpg')
    return labels

def plot_play_scores(labels):
    """
    plot scores in play mode per algorithm
    """

    with open('outputs/outcomes.pkl', 'rb') as handle:
        data = pickle.load(handle)

    num_episodes = 2000
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm scores playing with best policy after solve the environment')

    scores = []
    types = []
    for key in data.keys():
        scores.append(data[key]['score_play'])
        sc = data[key]['score_play']
        if key == "6_median":
            key = "5_median"
        types.append(key[:1])
        plt.bar(int(key[:1]), sc, label=labels[int(key[:1]) - 1] + " " +str(round(sc,2)))
    plt.ylabel('Score')
    plt.xlabel('Algorithm #')
    title = "Algorithm and reward in play mode"
    plt.legend(title=title)
    plt.ylim([0, 50])
    plt.tight_layout()

    plt.savefig(f'images/play_scores_all.jpg')
    return

def plot_time_all(labels):

    """
    plot time to win env 35+ mean average 100 episodes
    """
    with open('outputs/outcomes.pkl', 'rb') as handle:
        data = pickle.load(handle)

    num_episodes = 2000
    plt.figure(figsize=(16, 12))
    plt.subplot(111)
    plt.title(f'All Algorithm time to solve the environment. mean during at least 100 episodes of 35')

    scores = []
    types = []
    for key in data.keys():
        scores.append(data[key]['time'])
        sc = data[key]['time']
        if key == "6_median":
            key = "5_median"
        types.append(key[:1])

        plt.bar(int(key[:1]), sc, label=labels[int(key[:1]) - 1] + " " +str(round(sc,0)))
    plt.ylabel('Time')
    plt.xlabel('Algorithm #')
    title = "Algorithm and Time to solve Env"
    plt.legend(title=title)
    plt.ylim([0, 20000])
    plt.tight_layout()
    plt.savefig(f'images/time_scores_all.jpg')

    return