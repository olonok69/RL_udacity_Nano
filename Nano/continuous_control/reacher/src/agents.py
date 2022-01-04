import torch

from src.networks import *
from src.utils import *
import torch.optim as optim
import os
import random
from torch.autograd import Variable


class Agent_DDPG():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 checkpoint_folder = './checkpoints/',
                 train=True):
        """
        DDPG Agent implementation
        :param device: CPU or CUDA
        :type device: string
        :param state_size: env number of states (33)
        :type state_size: int
        :param n_agents: number of arms this agent (20 or 1)
        :type n_agents: int
        :param action_size: number of actions of this env (4)
        :type action_size: int
        :param random_seed: ramdom seed to repeat experiment and initialize weigths
        :type random_seed:  int
        :param buffer_size: size of buffer
        :type buffer_size: int
        :param batch_size: size of batch
        :type batch_size: int
        :param gamma: discount reward parameter
        :type gamma: float
        :param TAU: discount parameter soft update models
        :type TAU: float
        :param lr_actor: learning rate actor
        :type lr_actor: float
        :param lr_critic: learning rate critic
        :type lr_critic: float
        :param weight_decay: decay parameter learning rate
        :type weight_decay: float
        :param algo: type algorithm
        :type algo: string
        :param checkpoint_folder: folder for final models
        :type checkpoint_folder: strings
        :param train: mode train NN
        :type train: bool
        """
        self.DEVICE = device
        self.train = train
        self.algo = algo
        self.mode = "DDPG"

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network (w/ Target Network)
        self.actor_local = Actor_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_target = Actor_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.critic_target = Critic_ddpg(state_size, action_size, random_seed).to(self.DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC,
                                           weight_decay=self.WEIGHT_DECAY)

        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth') and self.train==False:
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))
            self.actor_target.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))

            self.critic_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth'))
            self.critic_target.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth'))

        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(device, action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        # losses
        self.losses_actor = []
        self.losses_critic = []

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents):
            self.memory.add(state[i ,:], action[i ,:], reward[i], next_state[i ,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, episodes, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)# we clip the action values to the constrains required on the problem statement

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # record loss
        self.losses_critic.append(critic_loss.item())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # record loss
        self.losses_actor.append(actor_loss.item())
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self, algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local.state_dict(), folder + f'checkpoint_critic_{algo}.pth')

class Agent_TD3():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 exploration_noise = 0.1,
                 target_policy_noise = 0.2,
                 target_policy_noise_clip = 0.5,
                 action_noise= False,
                 initial_random_steps = int(1e4),
                 checkpoint_folder = './checkpoints/',
                 train=True):
        """
        TD3 Agent implementation
        :param device: CPU or CUDA
        :type device: string
        :param state_size: env number of states (33)
        :type state_size: int
        :param n_agents: number of arms this agent (20 or 1)
        :type n_agents: int
        :param action_size: number of actions of this env (4)
        :type action_size: int
        :param random_seed: ramdom seed to repeat experiment and initialize weigths
        :type random_seed:  int
        :param buffer_size: size of buffer
        :type buffer_size: int
        :param batch_size: size of batch
        :type batch_size: int
        :param gamma: discount reward parameter
        :type gamma: float
        :param TAU: discount parameter soft update models
        :type TAU: float
        :param lr_actor: learning rate actor
        :type lr_actor: float
        :param lr_critic: learning rate critic
        :type lr_critic: float
        :param weight_decay: decay parameter learning rate
        :type weight_decay: float
        :param algo: type algorithm
        :type algo: string
        :param exploration_noise:  gaussian noise for policy
        :type exploration_noise: float
        :param target_policy_noise: gaussian noise for target policy
        :type target_policy_noise: float
        :param target_policy_noise_clip: clip target gaussian noise
        :type target_policy_noise_clip: float
        :param action_noise: use noise on target next_action
        :type action_noise: bool
        :param initial_random_steps:  initial random action steps
        :type initial_random_steps: int
        :param checkpoint_folder: folder for final models
        :type checkpoint_folder: strings
        :param train: mode train NN
        :type train: bool
        """

        # total steps count
        self.total_step = 0
        # update step for actor
        self.update_step = 2
        self.DEVICE = device # cpu/gpu
        self.train = train # training / no training
        self.algo = algo # cmd parameter type of algorithm
        self.mode = "TD3"

        self.state_size = state_size # observation space size
        self.n_agents = n_agents # number of agents
        self.action_size = action_size # Action space size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = TAU
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay

        self.chekpoint_folder = checkpoint_folder

        # noise
        # exploration
        self.exploration_noise = GaussianNoise(
            action_size, exploration_noise, exploration_noise
        )
        # Npise actions
        self.target_policy_noise = GaussianNoise(
            action_size, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip
        # introduce noise / not introduce Noise
        self.action_noise = action_noise

        # Actor Network (with Target Network)
        self.actor_local = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target.load_state_dict(self.actor_local.state_dict()) # initialize
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network 1 (with Target Network)
        self.critic_local_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1.load_state_dict(self.critic_local_1.state_dict()) # Initialize

        # Critic Network 2 (with Target Network)
        self.critic_local_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2.load_state_dict(self.critic_local_2.state_dict())

        self.initial_random_steps =initial_random_steps
        # concat critic parameters to use one optim
        critic_parameters = list(self.critic_local_1.parameters()) + list(
            self.critic_local_2.parameters()
        )
        self.critic_optimizer = optim.Adam(critic_parameters, lr=self.lr_critic)

        if os.path.isfile(self.chekpoint_folder + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                (self.chekpoint_folder + f'checkpoint_critic_1_{self.algo}.pth') and os.path.isfile \
                (self.chekpoint_folder + f'checkpoint_critic_2_{self.algo}.pth') and self.train==False:
            # load models from files
            self.actor_local.load_state_dict(torch.load(self.chekpoint_folder + f'checkpoint_actor_{self.algo}.pth'))
            self.critic_local_1.load_state_dict(torch.load(self.chekpoint_folder + f'checkpoint_critic_1_{self.algo}.pth'))
            self.critic_local_2.load_state_dict(torch.load(self.chekpoint_folder + f'checkpoint_critic_2_{self.algo}.pth'))


        # Replay memory
        self.memory = ReplayBuffer(device, action_size, self.buffer_size, self.batch_size, random_seed)
        # losses
        self.losses_actor = []
        self.losses_critic = []

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents) :
            self.memory.add(state[i ,:], action[i ,:], reward[i], next_state[i ,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        if self.train == True :
            self.total_step = episode + 1
        if self.total_step < self.initial_random_steps and self.train ==True:
            pass
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.exploration_noise.sample()
        return np.clip(action, -1, 1)# we clip the action values to the constrains required on the problem statement

    def reset(self):
        pass


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.train = True
        states, actions, rewards, next_states, dones = experiences

        # get actions with noise
        if self.action_noise:
            noise = torch.FloatTensor(self.target_policy_noise.sample()).to(self.DEVICE)
            clipped_noise = torch.clamp(
                noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
            )
            next_actions = (self.actor_target(next_states) + clipped_noise).clamp(
                -1.0, 1.0
            )
        else:
            next_actions = self.actor_target(next_states).clamp(
            -1.0, 1.0
            )
        # min (Q_1', Q_2')
        #print(next_actions)
        next_values1 = self.critic_target_1(next_states, next_actions)
        next_values2 = self.critic_target_2(next_states, next_actions)
        #print(next_values1, next_values2)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        Q_targets = rewards + (self.gamma * next_values * (1 - dones))
        Q_targets = Q_targets.detach()

        # critic loss
        values1 = self.critic_local_1(states, actions)
        values2 = self.critic_local_2(states, actions)
        critic1_loss = F.mse_loss(values1, Q_targets)
        critic2_loss = F.mse_loss(values2, Q_targets)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        # record loss
        self.losses_critic.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor #
        if self.total_step % self.update_step == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()

            # record loss
            self.losses_actor.append(actor_loss.item())

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic_local_1, self.critic_target_1)
            self.soft_update(self.actor_local, self.actor_target)
            self.soft_update(self.critic_local_2, self.critic_target_2)
        else:
            self.losses_actor.append(torch.zeros(1))

        return


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.tau
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self,algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local_1.state_dict(), folder + f'checkpoint_critic_1_{algo}.pth')
        torch.save(self.critic_local_2.state_dict(), folder + f'checkpoint_critic_2_{algo}.pth')

class Agent_TD3_4():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 exploration_noise = 0.1,
                 target_policy_noise = 0.2,
                 target_policy_noise_clip = 0.5,
                 action_noise= False,
                 initial_random_steps = int(1e4),
                 checkpoint_folder = './checkpoints/',
                 train=True,
                 mode='min'):
        """
        TD3 Agent implementation 4 critic estimators. use mode to select estimator
        possible modes min, mean and median

        :param device: CPU or CUDA
        :type device: string
        :param state_size: env number of states (33)
        :type state_size: int
        :param n_agents: number of arms this agent (20 or 1)
        :type n_agents: int
        :param action_size: number of actions of this env (4)
        :type action_size: int
        :param random_seed: ramdom seed to repeat experiment and initialize weigths
        :type random_seed:  int
        :param buffer_size: size of buffer
        :type buffer_size: int
        :param batch_size: size of batch
        :type batch_size: int
        :param gamma: discount reward parameter
        :type gamma: float
        :param TAU: discount parameter soft update models
        :type TAU: float
        :param lr_actor: learning rate actor
        :type lr_actor: float
        :param lr_critic: learning rate critic
        :type lr_critic: float
        :param weight_decay: decay parameter learning rate
        :type weight_decay: float
        :param algo: type algorithm
        :type algo: string
        :param exploration_noise:  gaussian noise for policy
        :type exploration_noise: float
        :param target_policy_noise: gaussian noise for target policy
        :type target_policy_noise: float
        :param target_policy_noise_clip: clip target gaussian noise
        :type target_policy_noise_clip: float
        :param action_noise: use noise on target next_action
        :type action_noise: bool
        :param initial_random_steps:  initial random action steps
        :type initial_random_steps: int
        :param checkpoint_folder: folder for final models
        :type checkpoint_folder: strings
        :param train: mode train NN
        :type train: bool
        :param mode: Use min or mean to calculate critic target error
        :type mode: string
        """

        # total steps count
        self.total_step = 0
        # update step for actor
        self.update_step = 2

        self.DEVICE = device
        self.train = train
        self.algo = algo
        self.mode = mode

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay

        self.CHECKPOINT_FOLDER = checkpoint_folder

        # noise
        # exploration
        self.exploration_noise = GaussianNoise(
            action_size, exploration_noise, exploration_noise
        )
        # Npise actions
        self.target_policy_noise = GaussianNoise(
            action_size, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip
        # introduce noise / not introduce Noise
        self.action_noise = action_noise

        # Actor Network (with Target Network)
        self.actor_local = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target = Actor_TD3(state_size, action_size).to(self.DEVICE)
        self.actor_target.load_state_dict(self.actor_local.state_dict()) # initialize
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network 1 (with Target Network)
        self.critic_local_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_1.load_state_dict(self.critic_local_1.state_dict())# initialize

        # Critic Network 2 (with Target Network)
        self.critic_local_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2 = Critic_TD3(state_size+action_size).to(self.DEVICE)
        self.critic_target_2.load_state_dict(self.critic_local_2.state_dict())# initialize

        # Critic Network 3 (with Target Network)
        self.critic_local_3 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_3 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_3.load_state_dict(self.critic_local_3.state_dict())# initialize

        # Critic Network 4 (with Target Network)
        self.critic_local_4 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_4 = Critic_TD3(state_size + action_size).to(self.DEVICE)
        self.critic_target_4.load_state_dict(self.critic_local_4.state_dict())# initialize

        self.initial_random_steps =initial_random_steps
        # concat critic parameters to use one optimizer
        critic_parameters = list(self.critic_local_1.parameters()) + list(
            self.critic_local_2.parameters()) + list(self.critic_local_3.parameters()) +  \
                            list(self.critic_local_4.parameters())

        self.critic_optimizer = optim.Adam(critic_parameters, lr=self.LR_CRITIC)
        # load models in test mode
        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.mode}_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_1_{self.mode}_{self.algo}.pth') and os.path.isfile \
                (self.CHECKPOINT_FOLDER + f'checkpoint_critic_2_{self.mode}_{self.algo}.pth') and self.train==False:
            # load models from files
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER +
                                                        f'checkpoint_actor_{self.mode}_{self.algo}.pth'))
            self.critic_local_1.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_1_{self.mode}_{self.algo}.pth'))
            self.critic_local_2.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_2_{self.mode}_{self.algo}.pth'))
            self.critic_local_3.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_3_{self.mode}_{self.algo}.pth'))
            self.critic_local_4.load_state_dict(
                torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic_4_{self.mode}_{self.algo}.pth'))


        # Replay memory
        self.memory = ReplayBuffer(device, action_size, self.BUFFER_SIZE, self.BATCH_SIZE, random_seed)
        # playholders to save loss of actor and critic
        self.losses_actor = []
        self.losses_critic = []

    def step(self, state, action, reward, next_state, done):
        """
        Agent step implementation
        :param state: state t
        :type state:  tensor
        :param action: action t
        :type action: tensor
        :param reward: reward action t
        :type reward: tensor
        :param next_state: state t+1
        :type next_state: tensor
        :param done: done bool
        :type done: bool
        :return:
        :rtype:
        """
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        """
        # Save experience / reward
        for i in range(self.n_agents) :
            self.memory.add(state[i ,:], action[i ,:], reward[i], next_state[i ,:], done[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, episode, add_noise=True):
        """Returns actions for given state as per current policy."""
        if self.train == True :
            self.total_step = episode + 1
        if self.total_step < self.initial_random_steps and self.train ==True:
            pass # not implemented for now
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.exploration_noise.sample()
        return np.clip(action, -1, 1)# we clip the action values to the constrains required on the problem statement

    def reset(self):
        pass


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.train = True
        states, actions, rewards, next_states, dones = experiences

        # get actions with noise
        if self.action_noise:
            noise = torch.FloatTensor(self.target_policy_noise.sample()).to(self.DEVICE)
            clipped_noise = torch.clamp(
                noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
            )
            next_actions = (self.actor_target(next_states) + clipped_noise).clamp(
                -1.0, 1.0
            )
        else:
            next_actions = self.actor_target(next_states).clamp(
            -1.0, 1.0
            )
        # min (Q_1', Q_2')

        next_values1 = self.critic_target_1(next_states, next_actions)
        next_values2 = self.critic_target_2(next_states, next_actions)
        next_values3 = self.critic_target_3(next_states, next_actions)
        next_values4 = self.critic_target_4(next_states, next_actions)
        if self.mode == "mean":
            next_values_1 = torch.add(next_values1, next_values2) / 2
            next_values_2 = torch.add(next_values3, next_values4) / 2
            next_values = torch.add(next_values_1, next_values_2) / 2
        elif self.mode == "min":
            next_values_1 = torch.min(next_values1, next_values2)
            next_values_2 = torch.min(next_values3, next_values4)
            next_values = torch.min(next_values_1, next_values_2)
        elif self.mode == "median":
            vector=[]
            for i in range(0, len(next_values1)):
                d = torch.stack((next_values1[i], next_values3[i],next_values3[i],next_values4[i]))
                c= torch.median(d,dim=0).values
                c = c.cpu().data.numpy()[0]
                vector.append(c)

            next_values= torch.from_numpy(np.array(vector)).resize(len(vector),1).to(self.DEVICE)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        Q_targets = rewards + (self.GAMMA * next_values * (1 - dones))
        Q_targets = Q_targets.detach()

        # critic loss
        values1 = self.critic_local_1(states, actions)
        values2 = self.critic_local_2(states, actions)
        values3 = self.critic_local_3(states, actions)
        values4 = self.critic_local_4(states, actions)
        critic1_loss = F.mse_loss(values1, Q_targets)
        critic2_loss = F.mse_loss(values2, Q_targets)
        critic3_loss = F.mse_loss(values3, Q_targets)
        critic4_loss = F.mse_loss(values4, Q_targets)
        # train critic
        critic_loss = critic1_loss + critic2_loss + critic3_loss + critic4_loss
        # record loss
        self.losses_critic.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        if self.total_step % self.update_step == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()

            # record loss
            self.losses_actor.append(actor_loss.item())

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic_local_1, self.critic_target_1)
            self.soft_update(self.actor_local, self.actor_target)
            self.soft_update(self.critic_local_2, self.critic_target_2)
            self.soft_update(self.critic_local_3, self.critic_target_3)
            self.soft_update(self.critic_local_4, self.critic_target_4)
        else:
            self.losses_actor.append(torch.zeros(1))

        # self.actor_loss = actor_loss.data
        # self.critic_loss = critic_loss.data
        return


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self,algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{self.mode}_{algo}.pth')
        torch.save(self.critic_local_1.state_dict(), folder + f'checkpoint_critic_1_{self.mode}_{algo}.pth')
        torch.save(self.critic_local_2.state_dict(), folder + f'checkpoint_critic_2_{self.mode}_{algo}.pth')
        torch.save(self.critic_local_3.state_dict(), folder + f'checkpoint_critic_3_{self.mode}_{algo}.pth')
        torch.save(self.critic_local_4.state_dict(), folder + f'checkpoint_critic_4_{self.mode}_{algo}.pth')

class Agent_D4PG():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 buffer_size,
                 batch_size,
                 gamma,
                 TAU,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 algo,
                 noise= True,
                 Vmax = 5,
                 Vmin = 0,
                 N_ATOMS = 51,
                 N_step=1,
                 UPDATE_EVERY=4,
                 checkpoint_folder = './models/',
                 train=True):
        """
        D4PG Agent Implementation

        """
        self.DEVICE = device
        self.train = train
        self.mode = "D4PG"
        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        # algo number
        self.algo =algo
        # categorical parameters
        self.vmax = Vmax # value max action
        self.vmin = Vmin # value min action
        self.n_atoms = N_ATOMS # number of bins for action distribution
        self.delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
        # total steps count
        self.total_step = 0
        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon = 0.3

        # Hyperparameters
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay
        # replay Memory
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.DEVICE, self.action_size, self.buffer_size, self.batch_size, self.seed)
        # checkpoint folder
        self.CHECKPOINT_FOLDER = checkpoint_folder

        # Actor Network
        # Actor Network (w/ Target Network)
        self.actor_local = Actor_D4PG(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_target = Actor_D4PG(state_size, action_size, random_seed).to(self.DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # initialize with its own Learning Rate
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)

        # Critic Network
        self.critic_local = CriticD4PG(state_size, action_size, random_seed, n_atoms=self.n_atoms, v_min=self.vmin,
                                       v_max=self.vmax).to(self.DEVICE)
        self.critic_target = CriticD4PG(state_size, action_size, random_seed, n_atoms=self.n_atoms, v_min=self.vmin,
                                        v_max=self.vmax).to(self.DEVICE)
        # initialize with its own Learning Rate
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC)

        # Noise process
        # Ornstein-Uhlenbeck process
        self.add_noise=noise
        self.noise = OUNoise(self.action_size, self.seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.train_start = 2000 # case we need to use warming
        self.UPDATE_EVERY = UPDATE_EVERY
        # helpers for rewards and states
        self.REWARD_STEPS  = N_step
        self.rewards_queue = deque(maxlen=self.REWARD_STEPS)
        self.states_queue = deque(maxlen=self.REWARD_STEPS)
        # losses
        self.losses_actor = []
        self.losses_critic = []
        # is train is  False , load trained model from folder
        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth') and os.path.isfile \
                    (self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.algo}.pth') and self.train == False:
            self.actor_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_actor_{self.algo}.pth'))
            self.critic_local.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + f'checkpoint_critic.pth_{self.algo}'))

    def reset(self):
        self.noise.reset()

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # self.states_queue.appendleft([state, action])
        # #print(reward, self.GAMMA, self.N_step)
        # self.rewards_queue.appendleft(reward[0] * self.GAMMA ** self.N_step)
        # for i in range(len(self.rewards_queue)):
        #     self.rewards_queue[i] = self.rewards_queue[i] / self.GAMMA
        #
        # if len(self.rewards_queue) >= self.N_step:  # N-steps return: r= r1+gamma*r2+..+gamma^(t-1)*rt
        #     temps = self.states_queue.pop()
        #     self.memory.add(temps[0], temps[1], sum(self.rewards_queue), next_state, done)
        #     self.rewards_queue.pop()
        #     if done:
        #         self.states_queue.clear()
        #         self.rewards_queue.clear()
        # # If enough samples are available in memory, get random subset and learn
        # self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        # if self.t_step == 0:
        #     if len(self.memory) > self.batch_size:
        #         experiences = self.memory.sample()
        #         self.learn(experiences, self.GAMMA)

        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # activate learning every few steps
        self.t_step = self.t_step + 1
        if self.t_step % self.UPDATE_EVERY == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                for _ in range(10):  # update 10 times per learning
                    experiences = self.memory.sample2()
                    self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy."""

        state = torch.tensor(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise and self.train:
            # add noise from Ornstein-Uhlenbeck process. only during training
            #action += self.noise.sample()
            action += self.epsilon * np.random.normal(size=action.shape)
            action = np.clip(action, -1.0, 1.0)
        elif self.train == False:
            action = np.clip(action, -1.0, 1.0)
        return action



    def learn(self, experiences, gamma, idxs =None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.train=True
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets_next = F.softmax(Q_targets_next, dim=1)
        Q_targets_next = self.distr_projection(Q_targets_next, rewards, dones, gamma ** self.REWARD_STEPS)
        Q_targets_next = -F.log_softmax(Q_expected, dim=1) * Q_targets_next
        critic_loss = Q_targets_next.sum(dim=1).mean()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)

        crt_distr_v = self.critic_local(states, actions_pred)
        actor_loss = -self.critic_local.distr_to_q(crt_distr_v)
        actor_loss = actor_loss.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        self.losses_actor.append(actor_loss.item())
        self.losses_critic.append(critic_loss.item())

    def distr_projection(self, next_distr_v, rewards_v, dones_mask_t, gamma):
        """

        The code is referred and adapted from
        1. https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter14

        :param next_distr_v:
        :type next_distr_v:
        :param rewards_v:
        :type rewards_v:
        :param dones_mask_t:
        :type dones_mask_t:
        :param gamma:
        :type gamma:
        :return:
        :rtype:
        """
        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, self.n_atoms), dtype=np.float32)
        dones_mask = np.squeeze(dones_mask)
        rewards = rewards.reshape(-1)

        for atom in range(self.n_atoms):
            tz_j = np.minimum(self.vmax, np.maximum(self.vmin, rewards + (self.vmin + atom * self.delta_z) * gamma))
            b_j = (tz_j - self.vmin) / self.delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l

            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l

            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(self.vmax, np.maximum(self.vmin, rewards[dones_mask]))
            b_j = (tz_j - self.vmin) / self.delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            if dones_mask.shape == ():
                if dones_mask:
                    proj_distr[0, l] = 1.0
                else:
                    ne_mask = u != l
                    proj_distr[0, l] = (u - b_j)[ne_mask]
                    proj_distr[0, u] = (b_j - l)[ne_mask]
            else:
                eq_dones = dones_mask.copy()

                eq_dones[dones_mask] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l[eq_mask]] = 1.0
                ne_mask = u != l
                ne_dones = dones_mask.copy()
                ne_dones[dones_mask] = ne_mask
                if ne_dones.any():
                    proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr).to(self.DEVICE)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        tau = self.TAU
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau *local_param.data + (1.0 -tau ) *target_param.data)

    def checkpoint(self, algo, folder):
        torch.save(self.actor_local.state_dict(), folder + f'checkpoint_actor_{algo}.pth')
        torch.save(self.critic_local.state_dict(), folder + f'checkpoint_critic_{algo}.pth')

def agent_train(env,brain_name, agent, n_agents ,algo, num_episodes):
    """
    train agent
    :param env: unity environtment
    :type env:
    :param brain_name: from unity
    :type brain_name: string
    :param agent: intance of agent object look above the diferent agent classes
    :type agent: Agent_DDPG, Agent_TD3 , Agent_TD3_4, Agent_D4PG
    :param n_agents: number of arms
    :type n_agents: int
    :param algo: type of algo. Command line argument
    :type algo: string
    :param num_episodes: number of episodes
    :type num_episodes:
    :return:
    :rtype:
    """
    scores = []
    loss_actor = []
    loss_critic = []
    scores_window = deque(maxlen=100)
    n_episodes = num_episodes

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations
        agent.reset()  # reset the agent noise
        score = np.zeros(n_agents)

        while True:
            actions = agent.act(states, episode)

            env_info = env.step(actions)[brain_name]  # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            dones = env_info.local_done  # see if episode has finished

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards  # update the score

            states = next_states  # roll over the state to next time step

            if np.any(dones):  # exit loop if episode finished
                break

        agent.checkpoint(algo, "./checkpoints/")

        scores.append(np.mean(score))

        loss_actor.append(np.mean(agent.losses_actor))
        loss_critic.append(np.mean(agent.losses_critic))
        scores_window.append(np.mean(score))
        if episode % 100 == 0:
            print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f} \tLoss critic: \t{:.2f} '
                  '\tloss Actor: \t{:.2f}'.format(episode, np.mean(score), np.mean(scores_window), np.mean(loss_critic)
                                                        ,np.mean(loss_actor)), end="")

        if np.mean(scores_window) >= 35.0:
            # if the agent hit 35 as score mean we consider solved the enviroment and we save the model in models
            agent.checkpoint(algo, "./models/")
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            break


    plot_scores(scores,algo ,n_episodes, agent.mode)
    plot_losses(loss_actor, algo, n_episodes, "actor", agent.mode)
    plot_losses(loss_critic, algo, n_episodes, "critic" ,agent.mode)
    return scores, loss_actor, loss_critic, agent.mode