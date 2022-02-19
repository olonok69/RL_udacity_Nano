import torch
from torch.utils.tensorboard import SummaryWriter
from src.networks import *
from src.utils import *
import torch.optim as optim
import os
import random
import copy
from torch.autograd import Variable
from torch.cuda.amp.grad_scaler import GradScaler


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

class Agent_A2C():
    def __init__(self,
                 device,
                 state_size,
                 n_agents,
                 action_size,
                 random_seed,
                 gamma,
                 lrate,
                 n_steps,
                 algo,
                 checkpoint_folder = './checkpoints/',
                 train = True,
                 mode='a2c'):



        # steps per trayectory
        self.n_steps = n_steps
        self.train = train # training or play
        self.DEVICE = device # CPU/GPU
        self.algo = algo # Type algo
        self.mode = mode # Mode (only in TD3, here name of algo)

        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Hyperparameters

        self.gamma = gamma
        self.LR = lrate
        self.CHECKPOINT_FOLDER = checkpoint_folder
        # Actor Network (with Target Network)
        self.model = A2CModel(self.state_size, self.action_size,  self.DEVICE).to(self.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR)

        # load models in test mode
        if os.path.isfile(self.CHECKPOINT_FOLDER + f'checkpoint_model_{self.mode}_{self.algo}.pth') \
                and self.train==False:
            # load models from files
            self.model.load_state_dict(torch.load(self.CHECKPOINT_FOLDER +
                                                        f'checkpoint_model_{self.mode}_{self.algo}.pth'))


        self.losses_model = []


    def learn(self, batch_s, batch_a, batch_v_t):
        '''
        Params
        ======
            batch_s (T, n_process, state_size) (numpy)
            batch_a (T, n_process, action_size) (numpy): batch of actions
            batch_v_t (T, n_process) (numpy): batch of n-step rewards (aks target value)
            model (object): A2C model
            optimizer (object): model parameter optimizer
        Returns
        ======
            total_loss (int): mean actor-critic loss for each batch
        '''

        batch_s_ = torch.from_numpy(batch_s).float().to(self.DEVICE)
        batch_s_ = batch_s_.view(-1, batch_s.shape[-1])  # shape from (T,n_process,state_size) -> (TxN, state_size)

        batch_a_ = torch.from_numpy(batch_a).float().to(self.DEVICE)
        batch_a_ = batch_a_.view(-1, batch_a.shape[-1])  # shape from (T,n_process,action_size) -> (TxN, action_size)

        values = self.model.get_state_value(batch_s_)  # shape (TxN,)
        values = values.view(*batch_s.shape[:2])  # shape (T,n)

        # pytorch's problem of negative stride -> require .copy() to create new numpy
        batch_v_t_ = torch.from_numpy(batch_v_t.copy()).float().to(self.DEVICE)
        td = batch_v_t_ - values  # shape (T, n_process) (tensor)
        c_loss = td.pow(2).mean()

        mus, stds, log_probs = self.model.get_action_prob(batch_s_, batch_a_)
        log_probs_ = log_probs.view(*batch_a.shape[:2])  # shape from (TxN,) -> (T,n) (tensor)

        a_loss = -((log_probs_ * td.detach()).mean())
        total_loss = c_loss + a_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        losses = total_loss.detach().cpu().data.numpy()
        # record loss
        self.losses_model.append(losses.item())
        means= mus.detach().cpu().data.numpy()
        stds = stds.cpu().data.numpy()
        # stds is constnat -> no gradient, no detach()
        return losses, means, stds

    def checkpoint(self,algo, folder):
        torch.save(self.model.state_dict(), folder + f'checkpoint_model_{self.mode}_{algo}.pth')

class Agent_A2C_b:

    def __init__(self,
                 state_dim,  # dimension of the state vector
                 action_dim,  # dimension of the action vector
                 num_envs,  # number of parallel agents (20 in this experiment)
                 device,
                 algo,
                 rollout_length=5,  # steps to sample before bootstraping
                 lr=1e-4,  # learning rate
                 lr_decay=.95,  # learning rate decay rate
                 gamma=.99,  # reward discount rate
                 value_loss_weight=1.0,  # strength of value loss
                 gradient_clip=5,  # threshold of gradient clip that prevent exploding
                 checkpoint_folder='./checkpoints/',
                 train=True,
                 mode='a2c'
                 ):
        self.model = ActorCriticNetwork_A2C(state_dim, action_dim).to(device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.total_steps = 0
        self.n_envs = num_envs
        self.value_loss_weight = value_loss_weight
        self.gradient_clip = gradient_clip
        self.rollout_length = rollout_length
        self.total_steps = 0
        self.device = device  # CPU/GPU
        self.train = train  # training or play
        self.CHECKPOINT_FOLDER = checkpoint_folder
        self.algo = algo  # Type algo
        self.mode = mode  # Mode (only in TD3, here name of algo)
        self.losses = []
        self.scaler= GradScaler()
        self.train= train

        if os.path.isfile \
                    (self.CHECKPOINT_FOLDER + f'checkpoint_critic_{self.mode}_{self.algo}.pth') and self.train == False:
            self.model.load_state_dict(torch.load(self.CHECKPOINT_FOLDER +
                                                  f'checkpoint_critic_{self.mode}_{self.algo}.pth'))


    def sample_action(self, state):
        """
        Sample action along with outputting the log probability and state values, given the states
        """
        state = torch.from_numpy(state).float().to(device=self.device)
        action, log_prob, state_value = self.model(state)
        return action, log_prob, state_value

    def update_model(self, experience):
        """
        Updates the actor critic network given the experience
        experience: list [[action,reward,log_prob,done,state_value]]
        """
        processed_experience = [None] * (len(experience) - 1)

        _advantage = torch.tensor(np.zeros((self.n_envs, 1))).float().to(device=self.device)  # initialize advantage Tensor
        _return = experience[-1][-1].detach()  # get returns
        for i in range(len(experience) - 2, -1, -1):  # iterate from the last step
            _action, _reward, _log_prob, _not_done, _value = experience[i]  # get training data
            _not_done = torch.tensor(_not_done, device=self.device).unsqueeze(
                1).float()  # masks indicating the episodes not finished
            _reward = torch.tensor(_reward, device=self.device).unsqueeze(1)  # get the rewards of the parallel agents
            _next_value = experience[i + 1][-1]  # get the next states
            _return = _reward + self.gamma * _not_done * _return  # compute discounted return
            _advantage = _reward + self.gamma * _not_done * _next_value.detach() - _value.detach()  # advantage
            processed_experience[i] = [_log_prob, _value, _return, _advantage]

        log_prob, value, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*processed_experience))
        policy_loss = -log_prob * advantages  # loss of the actor
        #value_loss = 0.5 * (returns - value).pow(2)  # loss of the critic (MSE)
        value_loss =F.smooth_l1_loss(returns, value)  # loss of the critic (MSE)
        loss  = (policy_loss + self.value_loss_weight * value_loss).mean()

        # record loss
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()  # total loss
        # Unscales the gradients of optimizer's assigned params in-place
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)  # clip gradient
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.total_steps += self.rollout_length * self.n_envs

    def act(self, state):
        """
        Conduct an action given input state vector
        Used in eveluation
        """
        state = torch.from_numpy(state).float().to(device=self.device)
        self.model.eval()
        action, _, _ = self.model(state)
        self.model.train()
        return action

    def reset(self):
        self.model.reset_parameters()
        #pass

    def save(self, path):
        """
        Save state_dict of the model
        """
        #torch.save({"state_dict": self.model.state_dict}, path)
        torch.save(self.model.state_dict(), path)
    #

class Agent_SAC():

    def __init__(self,
                 state_size,
                 action_size,
                 lr,
                 gamma,
                 batch_size,
                 buffer_size,
                 alpha,
                 tau,
                 target_update_interval,
                 gradient_steps,
                 n_agents,
                 device,
                 algo,
                 checkpoint_folder='./checkpoints/',
                 train=True,
                 mode='sac'
                 ):

        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.train = train
        self.mode = "SAC"
        self.checkpoint_folder= checkpoint_folder
        self.scaler = GradScaler()
        self.gradient_clip = 5
        self.seed = np.random.seed(4)


        # algo number
        self.algo = algo
        self.mode = mode

        self.value_max_grad_norm = float('inf')
        self.policy_max_grad_norm = float('inf')

        self.critic1 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(device=self.device)
        self.critic_optim1 = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(device=self.device)
        self.critic_optim2 = optim.Adam(self.critic2.parameters(), lr=lr)
        self.critic3 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(device=self.device)
        self.critic_optim3 = optim.Adam(self.critic3.parameters(), lr=lr)
        self.critic4 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(device=self.device)
        self.critic_optim4 = optim.Adam(self.critic4.parameters(), lr=lr)

        self.critic_target1 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(self.device)
        hard_update(self.critic_target1, self.critic1)
        self.critic_target2 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(self.device)
        hard_update(self.critic_target2, self.critic2)
        self.critic_target3 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(self.device)
        hard_update(self.critic_target3, self.critic3)
        self.critic_target4 = QNetwork(self.seed, self.state_size, self.action_size, 256).to(self.device)
        hard_update(self.critic_target4, self.critic4)




        # Initialize Replay Memory
        #self.memory = ReplayBuffer_2(self.action_size, self.buffer_size, self.batch_size, 0)
        self.memory = ReplayBuffer(device, action_size, self.buffer_size, self.batch_size, 0)

        self.target_entropy = -torch.prod(torch.Tensor(self.action_size).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=0.001)

        self.policy = GaussianPolicy(self.seed, self.state_size, self.action_size,
                                     256).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=0.0005)

        # Step counter
        self.step_counter = 0
        #self.writer = SummaryWriter()

        self.policy_losses = []
        self.alpha_losses = []
        # is train is  False , load trained model from folder
        if os.path.isfile(self.checkpoint_folder + f'checkpoint_policy_{self.algo}_{self.mode}.pth') \
            and self.train == False:
            self.policy.load_state_dict(torch.load(self.checkpoint_folder +
                                                           f'checkpoint_policy_{self.algo}_{self.mode}.pth'))

    def checkpoint(self, algo, folder):
        torch.save(self.policy.state_dict(), folder + f'checkpoint_policy_{self.algo}_{self.mode}.pth')

    def reset(self):
        #self.model.reset_parameters()
        pass


    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.train == False:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        action = torch.clamp(action, -1, 1)
        return action.detach().cpu().numpy()[0]

    def learn(self):
        self.train =True
        for _ in range(self.gradient_steps):

            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()

            #print(next_state_batch.shape, state_batch.shape)
            current_actions, logpi_s, _ = self.policy.sample(state_batch)

            target_alpha = (logpi_s + self.target_entropy).detach()
            alpha_loss = -(self.log_alpha * target_alpha).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha = self.log_alpha.exp()
            # add loss
            self.alpha_losses.append(alpha_loss)



            current_q_sa_a1 = self.critic1(state_batch, current_actions)
            current_q_sa_a2 = self.critic2(state_batch, current_actions)
            current_q_sa_b1 = self.critic3(state_batch, current_actions)
            current_q_sa_b2 = self.critic4(state_batch, current_actions)
            current_q_sa_a = torch.min(current_q_sa_a1, current_q_sa_b1)
            current_q_sa_b = torch.min(current_q_sa_a2, current_q_sa_b2)
            current_q_sa = torch.min(current_q_sa_a, current_q_sa_b)
            policy_loss = (alpha * logpi_s - current_q_sa.detach()).mean()


            # Q loss
            ap, logpi_sp, _ = self.policy.sample(next_state_batch)
            q_spap_a1= self.critic_target1(next_state_batch, ap)
            q_spap_a2 = self.critic_target2(next_state_batch, ap)
            q_spap_b1 = self.critic_target3(next_state_batch, ap)
            q_spap_b2 = self.critic_target4(next_state_batch, ap)
            q_spap_a = torch.min(q_spap_a1, q_spap_a2)
            q_spap_b = torch.min(q_spap_b1, q_spap_b2)
            q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
            target_q_sa = (reward_batch + self.gamma * q_spap * (1 - mask_batch)).detach()

            q_sa_a1 = self.critic1(state_batch, action_batch)
            q_sa_a2 = self.critic2(state_batch, action_batch)
            q_sa_b1 = self.critic3(state_batch, action_batch)
            q_sa_b2 = self.critic4(state_batch, action_batch)

            qa_loss1 = (q_sa_a1 - target_q_sa).pow(2).mul(0.5).mean()
            qa_loss2 = (q_sa_a2 - target_q_sa).pow(2).mul(0.5).mean()
            qb_loss1 = (q_sa_b1 - target_q_sa).pow(2).mul(0.5).mean()
            qb_loss2 = (q_sa_b2 - target_q_sa).pow(2).mul(0.5).mean()

            self.critic_optim1.zero_grad()
            qa_loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(),
                                           self.value_max_grad_norm)
            self.critic_optim1.step()

            self.critic_optim2.zero_grad()
            qa_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(),
                                           self.value_max_grad_norm)
            self.critic_optim2.step()

            self.critic_optim3.zero_grad()
            qb_loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.critic3.parameters(),
                                           self.value_max_grad_norm)
            self.critic_optim3.step()

            self.critic_optim4.zero_grad()
            qb_loss2.backward()
            torch.nn.utils.clip_grad_norm_(self.critic4.parameters(),
                                           self.value_max_grad_norm)
            self.critic_optim4.step()

            self.policy_optim.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                           self.policy_max_grad_norm)
            self.policy_optim.step()
            self.policy_losses.append(policy_loss)
            #
            soft_update(self.critic_target1, self.critic1, self.tau)
            soft_update(self.critic_target2, self.critic2, self.tau)
            soft_update(self.critic_target3, self.critic3, self.tau)
            soft_update(self.critic_target4, self.critic4, self.tau)
            return



    def soft_update(self, local_model, target_model, tau=0.005):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def step(self, states, actions, rewards, next_states, dones):

        #for i in range(self.n_agents):
        self.memory.add(states, actions, rewards, next_states, dones)
        self.step_counter += 1

        if self.step_counter >= self.target_update_interval and len(self.memory) > self.batch_size:
            self.learn()
            self.step_counter = 0

class soft_actor_critic_agent(object):
    def __init__(self, num_inputs, action_space,
                 device, hidden_size, seed, lr, gamma, tau, alpha):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.device = device
        self.seed = seed
        self.seed = torch.manual_seed(seed)

        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic=True

        self.critic = QNetwork(seed, num_inputs, action_space, hidden_size).to(device=self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(seed, num_inputs, action_space, hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -torch.prod(torch.Tensor(action_space).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        self.policy = GaussianPolicy(seed, num_inputs, action_space,
                                     hidden_size, action_space).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()  # For TensorboardX logs

        soft_update(self.critic_target, self.critic, self.tau)

class Agent_PPO():
    def __init__(self, env, hyper_params):
        self.env = env
        self.num_agents = env.num_agents
        self.action_size = env.action_space_size
        self.state_size = env.state_size
        self.hyper_params = hyper_params
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Learning objects
        self.memory = Memory(hyper_params['memory_size'], hyper_params['batch_size'])
        self.policy = ActorCritic_PPO(self.state_size, self.action_size, seed=0)

        self.opt = torch.optim.Adam(self.policy.parameters(), 3e-4)

        # Starting Values
        self.states = env.info.vector_observations
        self.epsilon = hyper_params['epsilon']
        self.c_entropy = hyper_params['c_entropy']
        self.losses = []

    def collect_trajectories(self):
        done = False
        states = self.env.info.vector_observations

        state_list = []
        reward_list = []
        prob_list = []
        action_list = []
        value_list = []

        # Random steps to start
        if self.hyper_params['t_random'] > 0:
            for _ in range(self.hyper_params['t_random']):
                actions = np.random.randn(self.num_agents, self.action_size)
                actions = np.clip(actions, -1, 1)
                env_info = self.env.step(actions)
                states = env_info.vector_observations

        # Finish trajectory using policy
        for t in range(self.hyper_params['t_max']):
            states = torch.FloatTensor(states)
            dist, values = self.policy(states)
            actions = dist.sample()
            probs = dist.log_prob(actions).sum(-1).unsqueeze(-1)

            env_info = self.env.step(actions.cpu().detach().numpy())
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # store the result
            state_list.append(states)
            reward_list.append(rewards)
            prob_list.append(probs)
            action_list.append(actions)
            value_list.append(values)

            states = next_states

            if np.any(dones):
                done = True
                break

        value_arr = torch.stack(value_list)
        reward_arr = torch.FloatTensor(np.array(reward_list)[:, :, np.newaxis])

        advantage_list = []
        return_list = []

        _, next_value = self.policy(torch.FloatTensor(states))
        returns = next_value.detach()

        advantages = torch.FloatTensor(np.zeros((self.num_agents, 1)))
        for i in reversed(range(len(state_list))):
            returns = reward_arr[i] + self.hyper_params['discount'] * returns
            td_error = reward_arr[i] + self.hyper_params['discount'] * next_value - value_arr[i]
            advantages = advantages * self.hyper_params['gae_param'] * self.hyper_params['discount'] + td_error
            next_value = value_arr[i]
            advantage_list.insert(0, advantages.detach())
            return_list.insert(0, returns.detach())

        return_arr = torch.stack(return_list)
        indices = return_arr >= np.percentile(return_arr, self.hyper_params['curation_percentile'])
        indices = torch.squeeze(indices, dim=2)

        advantage_arr = torch.stack(advantage_list)
        state_arr = torch.stack(state_list)
        prob_arr = torch.stack(prob_list)
        action_arr = torch.stack(action_list)

        self.memory.add({'advantages': advantage_arr[indices],
                         'states': state_arr[indices],
                         'log_probs_old': prob_arr[indices],
                         'returns': return_arr[indices],
                         'actions': action_arr[indices]})

        rewards = np.sum(np.array(reward_list), axis=0)
        return rewards, done

    def update(self):
        advantages_batch, states_batch, log_probs_old_batch, returns_batch, actions_batch = self.memory.categories()
        actions_batch = actions_batch.detach()
        log_probs_old_batch = log_probs_old_batch.detach()
        advantages_batch = (advantages_batch - advantages_batch.mean()) / advantages_batch.std()

        batch_indices = self.memory.sample()

        # Gradient ascent
        for _ in range(self.hyper_params['num_epochs']):
            for batch_idx in batch_indices:
                batch_idx = torch.LongTensor(batch_idx)

                advantages_sample = advantages_batch[batch_idx]
                states_sample = states_batch[batch_idx]
                log_probs_old_sample = log_probs_old_batch[batch_idx]
                returns_sample = returns_batch[batch_idx]
                actions_sample = actions_batch[batch_idx]

                dist, values = self.policy(states_sample)

                log_probs_new = dist.log_prob(actions_sample).sum(-1).unsqueeze(-1)
                entropy = dist.entropy().sum(-1).unsqueeze(-1).mean()
                vf_loss = (returns_sample - values).pow(2).mean()

                ratio = (log_probs_new - log_probs_old_sample).exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                # Loss is negative for ascent.
                # clipped_surrogate_loss = -torch.min(ratio * advantages_sample, clipped_ratio * advantages_sample).mean()
                # loss = clipped_surrogate_loss - self.c_entropy * entropy + self.hyper_params['c_vf'] * vf_loss
                clipped_surrogate_loss = torch.min(ratio * advantages_sample, clipped_ratio * advantages_sample).mean()
                loss = -(clipped_surrogate_loss - (self.hyper_params['c_vf'] * vf_loss) + (self.c_entropy * entropy))

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.hyper_params['gradient_clip'])
                self.opt.step()
                self.losses.append(loss.item())

        # Reduce clipping range to create smaller changes in policy over time
        self.epsilon *= .999
        # Reduce entropy coeffiecient to reduce entropy/exploration
        self.c_entropy *= .995
        return np.mean(self.losses)

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

def agent_train_ac2(env,brain_name, agent, n_agents ,algo, num_episodes):
    """

    :param env:
    :type env:
    :param brain_name:
    :type brain_name:
    :param agent:
    :type agent:
    :param n_agents:
    :type n_agents:
    :param algo:
    :type algo:
    :param num_episodes:
    :type num_episodes:
    :return:
    :rtype:
    """
    num_episodes = num_episodes
    rollout_length = 5

    total_rewards = []
    loss_actor = []
    avg_scores = []
    max_avg_score = -1
    max_score = -1
    worsen_tolerance = 10  # for early-stopping training if consistently worsen for # episodes
    rollout = []
    for i_episode in range(1, num_episodes + 1):
        env_inst = env.reset(train_mode=True)[brain_name]  # reset the environment

        states = env_inst.vector_observations  # get the current state
        scores = np.zeros(n_agents)  # initialize the score
        dones = [False] * n_agents
        steps_taken = 0
        experience = []
        while not np.any(dones):  # finish if any agent is done
            steps_taken += 1
            actions, log_probs, state_values = agent.sample_action(states)  # select actions for 20 envs
            env_inst = env.step(actions.detach().cpu().numpy())[brain_name]  # send the actions to the environment
            next_states = env_inst.vector_observations  # get the next states
            rewards = env_inst.rewards  # get the rewards
            dones = env_inst.local_done  # see if episode has finished
            not_dones = [1 - done for done in dones]
            experience.append([actions, rewards, log_probs, not_dones, state_values])
            if steps_taken % rollout_length == 0:
                agent.update_model(experience)
                del experience[:]

            scores += rewards  # update the scores
            states = next_states  # roll over the states to next time step
        episode_score = np.mean(scores)  # compute the mean score for 20 agents
        episode_loss =  np.mean(agent.losses)
        total_rewards.append(episode_score)
        loss_actor.append(np.mean(agent.losses))

        print("Episodic {} Score: {} loss: {}".format(i_episode, np.mean(scores), np.mean(episode_loss)))
        if np.mean(scores) > max_score:
            path = f"./checkpoints/checkpoint_critic_{agent.mode}_{algo}.pth"
            agent.save(path)
            max_score = np.mean(scores)

        if len(total_rewards) % 100 == 0:  # record avg score for the latest 100 steps
            latest_avg_score = sum(total_rewards[(len(total_rewards) - 100):]) / 100
            print("100 Episodic Everage Score: {}".format(latest_avg_score))
            avg_scores.append(latest_avg_score)

        if sum(total_rewards[(len(total_rewards) - 100):]) / 100 >= 35.0:
            # if the agent hit 35 as score mean we consider solved the enviroment and we save the model in models
            path = f"./models/checkpoint_critic_{agent.mode}_{algo}.pth"
            agent.save(path)
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                       np.mean(avg_scores)))
            break
    return  total_rewards, loss_actor, agent.mode

def agent_train_ppo(env,brain_name, agent, n_agents ,algo, num_episodes):

    def play_round(env, brain_name, policy, n_agents):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(n_agents)
        while True:
            actions, _, _, _ = policy(states)
            env_info = env.step(actions.cpu().detach().numpy())[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            if np.any(dones):
                break

        return np.mean(scores)

    n_episodes = num_episodes
    all_scores = []
    averages = []
    losses = []
    last_max = 30.0

    for episode in range(n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        agent.states = env_info.vector_observations


        states = agent.states
        loss = agent.step(states)
        losses.append(loss)
        last_mean_reward = play_round(env, brain_name, agent.model, n_agents)
        all_scores.append(last_mean_reward)

        last_average = np.mean(np.array(all_scores[-100:])) if len(all_scores) > 100 else np.mean(np.array(all_scores))

        averages.append(last_average)
        agent.checkpoint(algo, "./checkpoints/")

        if last_average > last_max:
            agent.checkpoint(algo, "./models/")
            break

        print('Episode: {} Total score this episode: {} Last {} average: {} loss: {}'.format(episode + 1,
                                                                                             last_mean_reward,
                                                                                             min(episode + 1, 100),
                                                                                             last_average,
                                                                                             loss))

    return all_scores, averages,   losses, agent.mode

def agent_train_sac(env,brain_name, agent, n_agents ,algo, num_episodes, BATCH_SIZE):

    import progressbar as pb
    def interact(action, num_agents):
        action = action.reshape(num_agents, -1)
        env_info = env.step(action)[brain_name]
        next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
        return next_state.reshape(num_agents, -1), np.array(reward).reshape(num_agents, -1), np.array(done).reshape(
            num_agents, -1)

    def reset(env, num_agents ):
        state = env.reset()[brain_name].vector_observations.reshape(num_agents, -1)
        return state

    print_every = 100
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=print_every)  # last 100 scores
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=num_episodes).start()
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    frame_counter = 0
    t_max = 1000

    for i_episode in range(1, num_episodes + 1):
        states = reset(env, n_agents)
        score = 0
        for t in range(t_max):
            frame_counter += 1
            if (frame_counter % 1000) != 0:
                #actions = agent.act(states, i_episode)
                actions = agent.select_action(states)
                actions = np.clip(actions, -1, 1)
            else:
                actions = np.random.randn(n_agents, action_size)
                actions = np.clip(actions, -1, 1)


            next_states, rewards, dones = interact(actions, n_agents)

            #if len(agent.memory) > BATCH_SIZE:
            #print(len(agent.memory))
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards.mean()
            if np.any(dones):
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        #agent.writer.add_scalar('score/mean', score, i_episode)
        print('\rEpisode {}\tScore : {:.2f}'.format(i_episode, score))
        if i_episode % print_every == 0:
            print('\rEpisode {}\tScore Mean: {:.2f}\tScore STD: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                               np.std(scores_window)))
        if np.mean(scores_window) >= 30:
            print("Environment solved")
            break

        timer.update(i_episode)
    return scores


def sac_train(max_steps, threshold, env, start_steps, agent,memory, batch_size, brain_name):
    def save(agent, directory, filename, episode, reward):
        torch.save(agent.policy.state_dict(), '%s/%s_actor_%s_%s.pth' % (directory, filename, episode, reward))
        torch.save(agent.critic.state_dict(), '%s/%s_critic_%s_%s.pth' % (directory, filename, episode, reward))

    import time
    total_numsteps = 0
    updates = 0
    num_episodes = 20000
    updates = 0

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []
    # brain = env.brains[env.brain_name]
    # action_size = brain.vector_action_space_size
    num_agents =  env.num_agents
    for i_episode in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        for step in range(max_steps):
            if start_steps > total_numsteps:
                action = np.random.randn(env.num_agents, env.action_space_size)  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > batch_size:
                # Update parameters of all the networks
                agent.update_parameters(memory, batch_size, updates)

                updates += 1

            #next_state, reward, done, _ = env.step(action)  # Step
            env_info = env.step(action)
            next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
            next_state = next_state.reshape(num_agents, -1)
            reward = np.array(reward).reshape(num_agents, -1)
            done = np.array(done).reshape(num_agents, -1)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            #mask = 1 if episode_steps == max_steps else float(not done)

            memory.push(state, action, reward, next_state, done)  # Append transition to memory

            state = next_state

            if done.any():
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        max_score = np.max(scores_deque)

        if i_episode % 100 == 0 and i_episode > 0:
            reward_round = round(episode_reward, 2)
            save(agent, 'checkpoints/sac', 'weights', str(i_episode), str(reward_round))

        s = (int)(time.time() - time_start)

        print(
            "Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.3f}, Avg.Score: {:.3f}, Max.Score: {:.3f}, Time: {:02}:{:02}:{:02}".
            format(i_episode, total_numsteps, episode_steps, episode_reward, avg_score, max_score, s // 3600, s % 3600 // 60, s % 60))

        if (avg_score > threshold):
            print('Solved environment with Avg Score:  ', avg_score)
            break

    return scores_array, avg_scores_array


def collect_trajectories(model, env, brain_name, init_states, episode_end, n_steps, device, gamma):
    '''
    Params
    ======
        model (object): A2C model
        env (object): environment
        brain_name (string): brain name of environment
        init_states (n_process, state_size) (numpy): initial states for loop
        episode_end (bool): tracker of episode end, default False
        n_steps (int): number of steps for reward collection
        device (string): cuda / cpu
        gamma (float): horizon discount factor

    Returns
    =======
        batch_s (T, n_process, state_size) (numpy): batch of states
        batch_a (T, n_process, action_size) (numpy): batch of actions
        batch_v_t (T, n_process) (numpy): batch of n-step rewards (aks target value)
        accu_rewards (n_process,) (numpy): accumulated rewards for process (being summed up on all process)
        init_states (n_process, state_size) (numpy): initial states for next batch
        episode_end (bool): tracker of episode end
    '''

    batch_s = []
    batch_a = []
    batch_r = []

    states = init_states
    accu_rewards = np.zeros(init_states.shape[0])

    t = 0
    while True:
        t += 1

        model.eval()
        with torch.no_grad():
            states = torch.from_numpy(states).float().to(device)
            actions_tanh, actions = model.get_action(states)
        model.train()
        # actions_tanh (n_process, action_size) (tensor), actions limited within (-1,1)
        # actions (n_process, action_size) (tensor)

        env_info = env.step(actions_tanh.cpu().data.numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        # next_states (numpy array)
        # rewards (list)
        # dones (list)
        rewards = np.array(rewards)
        dones = np.array(dones)

        accu_rewards += rewards

        batch_s.append(states.cpu().data.numpy())  # final shape of batch_s (T, n_process, state_size) (list of numpy)
        batch_a.append(actions.cpu().data.numpy())  # final shape of batch_a (T, n_process, action_size) (list of numpy)
        batch_r.append(rewards)  # final shape of batch_r (T, n_process) (list of numpy array)

        if dones.any() or t >= n_steps:
            model.eval()
            next_states = torch.from_numpy(next_states).float().to(device)
            final_r = model.get_state_value(next_states).detach().cpu().data.numpy()  # final_r (n_process,) (numpy)
            model.train()

            for i in range(len(dones)):
                if dones[i] == True:
                    final_r[i] = 0
                else:
                    final_r[i] = final_r[i]

            batch_v_t = []  # compute n-step rewards (aks target value)
            batch_r = np.array(batch_r)

            for r in batch_r[::-1]:
                mean = np.mean(r)
                std = np.std(r)
                r = (r - mean) / (std + 0.0001)  # normalize rewards in n_process on each timestep
                final_r = r + gamma * final_r
                batch_v_t.append(final_r)
            batch_v_t = np.array(batch_v_t)[::-1]  # final shape (T, n_process) (numpy)

            break

        states = next_states

    if dones.any():
        env_info = env.reset(train_mode=True)[brain_name]
        init_states = env_info.vector_observations
        episode_end = True

    else:
        init_states = next_states.cpu().data.numpy()  # if not done, continue batch collection from last states

    batch_s = np.stack(batch_s)
    batch_a = np.stack(batch_a)

    return batch_s, batch_a, batch_v_t, np.sum(accu_rewards), init_states, episode_end