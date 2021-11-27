
from Networks import *
from buffers import *
from typing import Dict, List, Tuple
from torch.autograd import Variable

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

class DQN_Agent():
    """
    DQN Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 # PER parameters
                 alpha: float = 0.2,
                 beta: float = 0.6,
                 prior_eps: float = 1e-6, ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0: # sampling from Memory
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # original from Udacity DQN algo
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Double DQN
        # Q_targets_next = self.qnetwork_target(next_states).gather(  # Double DQN
        #     1, self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)).detach()
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets) replaced mse loss woth huber loss.
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class Categorical_DQN_Agent():
    """
    DQN Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 # Categorical DQN parameters
                 v_min: float = 0.0,
                 v_max: float = 200.0,
                 atom_size: int = 51, ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(device)

        # Q-Network
        self.qnetwork_local = Categorical_DQN_Network(state_size, action_size, self.atom_size, self.support).to(device)
        self.qnetwork_target = Categorical_DQN_Network(state_size, action_size, self.atom_size, self.support).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0: # sampling from Memory
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """

        states, actions, rewards, next_states, dones = experiences
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            next_action = self.qnetwork_target(next_states).argmax(1)
            next_dist = self.qnetwork_target.dist(next_states)
            next_dist = next_dist[range(BATCH_SIZE), next_action]

            t_z = rewards + (1 - dones) * GAMMA * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (BATCH_SIZE - 1) * self.atom_size, BATCH_SIZE
                ).long()
                    .unsqueeze(1)
                    .expand(BATCH_SIZE, self.atom_size)
                    .to(device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.qnetwork_local.dist(states)
        log_p = torch.log(dist[range(BATCH_SIZE), actions])

        loss = -(proj_dist * log_p).sum(1).mean()

        # Get max predicted Q values (for next states) from target model
        # original from Udacity DQN algo
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Double DQN
        # # Q_targets_next = self.qnetwork_target(next_states).gather(  # Double DQN
        # #     1, self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)).detach()
        # # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #
        # # Get expected Q values from local model
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        #
        # # Compute loss
        # # loss = F.mse_loss(Q_expected, Q_targets) replaced mse loss woth huber loss.
        # loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Dueling_DQN_Agent_PR():
    """
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size,
                 action_size,
                 seed,
                 prioritary_buffer=False,
                 noisy=False,
                ):
        """
        Initialize an Agent object.
        :param state_size: (int): dimension of each state
        :param action_size: action_size (int): dimension of each action
        :param seed: seed (int): random seed
        :param prioritary_buffer: bool True-->with priority Buffer Replay, False --> Normal deque implementation
        :param noisy: Bool Use Noisy Layer True
        :param atom_size: the unit number of support
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.prioritary_buffer = prioritary_buffer
        self.noisy = noisy


        self.loss_list = []

        # Q-Network
        if self.noisy:
            # with Noisy Layer--> remove epsilon greedy
            self.qnetwork_local = Noisy_DuelingNetwork(state_size, action_size).to(device)
            self.qnetwork_target = Noisy_DuelingNetwork(state_size, action_size).to(device)
        else:
            # No Noisy Layer
            self.qnetwork_local = DuelingNetwork(state_size, action_size).to(device)
            self.qnetwork_target = DuelingNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if prioritary_buffer:
            self.memory2 = PrioritizedReplayBuffer(BUFFER_SIZE)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def local_prediction(self, state):
        """Predict Q values for given state using local Q network

        Arguments:
            state {array-like} -- Dimension of state space

        Returns:
            [array] -- Predicted Q values for each action in state
        """

        pred = self.qnetwork_local(
            Variable(torch.FloatTensor(state)).to(device)
        )
        pred = pred.data
        return pred

    def step(self, state, action, reward, next_state, done):
        """

        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        if self.prioritary_buffer:
            # Get the timporal difference (TD) error for prioritized replay
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            with torch.no_grad():
                # Get old Q value. Not that if continous we need to account for batch dimension
                old_q = self.local_prediction(state)[action]

                # Get the new Q value.
                new_q = reward
                if not done:
                    new_q += GAMMA * torch.max(
                        self.qnetwork_target(
                            Variable(torch.FloatTensor(next_state)).to(device)
                        ).data
                    )

                td_error = abs(old_q - new_q)
            self.qnetwork_local.train()
            self.qnetwork_target.train()
            # Save experience in replay memory
            self.memory2.add(td_error.item(), (state, action, reward, next_state, done))
            # Learn every UPDATE_FREQUENCY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:

                # If enough samples are available in memory, get random subset and learn
                if len(self.memory2) > BATCH_SIZE:
                    experiences, idxs, is_weight = self.memory2.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA, idxs, is_weight)

        else:
            # Save experience in replay memory (NO PRIORITIZATION)
            self.memory.add(state, action, reward, next_state, done)

            # Learn every UPDATE_EVERY time steps.
            self.t_step = (self.t_step + 1) % UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA,0) # add 0 for idx case no PRB

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """
        if self.noisy:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            # Choose action values according to local model
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
        # Epsilon-greedy action selection
            if random.random() > eps:
                state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                # Choose action values according to local model
                self.qnetwork_local.eval()
                with torch.no_grad():
                    action_values = self.qnetwork_local(state)
                self.qnetwork_local.train()

                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, idxs, is_weight=0):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences
        # Convertions
        states = Variable(torch.Tensor(states)).float().to(device)
        next_states = Variable(torch.Tensor(next_states)).float().to(device)
        actions = torch.LongTensor(actions).view(-1, 1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        is_weight = torch.FloatTensor(is_weight).unsqueeze(1).to(device)


        if self.prioritary_buffer:
            # Dueling Network with Priority buffer
            q_local_argmax = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_local_argmax).detach()

            # Get Q values for chosen action
            predictions = self.qnetwork_local(states).gather(1, actions)
            # Calculate TD targets
            targets = (rewards + (GAMMA * q_targets_next * (1 - dones)))
            # Update priorities
            errors = torch.abs(predictions - targets).data.cpu().numpy()
            for i in range(len(errors)):
                self.memory2.update(idxs[i], errors[i])

            # Get the loss, using importance sampling weights
            loss = (is_weight * nn.MSELoss(reduction='none')(predictions, targets)).mean()

            # Run optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update list losses
            with torch.no_grad():
                self.loss_list.append(loss.item())

            if self.noisy:
                # NoisyNet: reset noise
                self.qnetwork_local.reset_noise()
                self.qnetwork_target.reset_noise()
            # update target network
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        else:


        # Double DQN
            Q_targets_next = self.qnetwork_target(next_states).gather(  # Double DQN
                1, self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)).detach()
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

            # Get expected Q values from local model
            Q_expected = self.qnetwork_local(states).gather(1, actions)

            # Compute loss
            # loss = F.mse_loss(Q_expected, Q_targets) replaced mse loss woth huber loss.
            loss = F.smooth_l1_loss(Q_expected, Q_targets)

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # NoisyNet: reset noise
            if self.noisy:

                self.qnetwork_local.reset_noise()
                self.qnetwork_target.reset_noise()
            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Dueling_DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network

        self.qnetwork_local = DuelingNetwork(state_size, action_size).to(device)
        self.qnetwork_target = DuelingNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Choose action values according to local model
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted actions (for next states) from local model
        # next_local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)

        # Evaluate the max predicted actions from the local model on the target model
        # based on Double DQN
        # Q_targets_next_values = self.qnetwork_target(next_states).detach().gather(1, next_local_actions)

        # Compute Q targets for current states
        # Q_targets = rewards + (gamma * Q_targets_next_values * (1 - dones))

        # Get expected Q values from local
        # Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets)

        # Double DQN
        Q_targets_next = self.qnetwork_target(next_states).gather(  # Double DQN
            1, self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)).detach()
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets) replaced mse loss woth huber loss.
        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class Double_DQN_Agent():
    """
    Double DQN Agent
    Interacts with and learns from the environment.
    """

    def __init__(self, state_size
                 , action_size,
                 seed,
                 # PER parameters
                 alpha: float = 0.2,
                 beta: float = 0.6,
                 prior_eps: float = 1e-6, ):
        """
        Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # Replay memory. Simply Buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """
        Step implementation
        :param state: state agent
        :param action: action agent
        :param reward: reward r after taken action a in state s
        :param next_state: state after taken action a in state s
        :param done:
        :return:
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0: # sampling from Memory
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.
        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return:
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.
        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return:
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        # original from Udacity DQN algo
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Double DQN
        Q_targets_next = self.qnetwork_target(next_states).gather(  # Double DQN
            1, self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)).detach()
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        # loss = F.mse_loss(Q_expected, Q_targets) replaced mse loss woth huber loss.
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def dqn(env,
        brain_name: str,
        agent,
        algo: int,
        n_episodes : int,
        max_t : int,
        eps_start : float,
        eps_end : float,
        eps_decay : float) -> Tuple[list, deque, int]:
    """
    Deep Dueling DQN

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    epi = 0 # number of episodes
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            action = action.astype(int)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 16.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(
                                                                                             scores_window)))
            epi = i_episode - 100
            torch.save(agent.qnetwork_local.state_dict(), f'models\checkpoint_{algo}.pth')
            break
        epi = i_episode - 100
    return scores, scores_window, epi
