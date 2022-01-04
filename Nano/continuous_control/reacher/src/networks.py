import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def hidden_init(layer):
    """
    helper to initialize layer
    :param layer: torch layer
    :type layer:
    :return:
    :rtype:
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

class Actor_ddpg(nn.Module):
    """Actor DDPG Model Policy"""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """
        Initialize parameters and build model.
            adapted from https://github.com/MrSyee/pg-is-all-you-need
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor_ddpg, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layers
        :return:
        :rtype:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        :param state: agent state
        :type state: torch Tensor
        :return: action
        :rtype: torch tensor
        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic_ddpg(nn.Module):
    """
    Critic DDPG Model (Value).
    """
    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """
        Initialize parameters and build model.
            adapted from https://github.com/MrSyee/pg-is-all-you-need
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic_ddpg, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layers
        :return:
        :rtype:
        """
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Build a critic (value) network that maps (state, action) pairs -> Q-values.
        :param state: agent state
        :type state: torch Tensor
        :param action: action agent
        :type action: torch Tensor
        :return: Q-values
        :rtype: torch tensor :
        """

        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Actor_TD3(nn.Module):
    """
    adapted from https://github.com/MrSyee/pg-is-all-you-need
    Actor TD3 Class Policy Network
    """
    def __init__(self, in_dim: int, out_dim: int, init_w: float = 3e-3):
        """
        Initialize.

        """
        super(Actor_TD3, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)
        # initialization of layers
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation.
        :param state: agent state
        :type state: torch Tensor
        :return: state
        :rtype: torch tensor
        """

        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        action = self.out(x).tanh() # on the agent we will need to clip to -1/1

        return action

class Critic_TD3(nn.Module):
    """
    Critic TD3 Class Value Function Network
    adapted from https://github.com/MrSyee/pg-is-all-you-need
    """
    def __init__(self, in_dim: int, init_w: float = 3e-3):
        """Initialize."""
        super(Critic_TD3, self).__init__()

        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        # Initialize layers
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward method implementation. Build a critic (value) network that maps (state, action) pairs -> Q-values.
        :param state: agent state
        :type state: torch Tensor
        :param action: action agent
        :type action: torch Tensor
        :return: Q-values
        :rtype: torch tensor :
        """

        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        value = self.out(x)

        return value

class Actor_D4PG(nn.Module):
    """
    Actor D4PG (Policy) Model.
    Actor - return action value given states.
    """
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """
        Initialize parameters and build model.
        adapted from https://github.com/schatty/d4pg-pytorch
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        """
        super(Actor_D4PG, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, action_size)
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layers
        :return:
        :rtype:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build an actor (policy) network that maps states -> actions.
        :param state: agent state
        :type state: torch Tensor
        :return: action
        :rtype: torch tensor
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.tanh(x)
        return x

class CriticD4PG(nn.Module):
    """
    Critic D4PG (distribution) Model.
    Critic - return Q value from given states and actions.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128, fc3_units=128,
                 n_atoms=51, v_min=-1, v_max=1):
        """Initialize parameters and build model.
        adapted from https://github.com/schatty/d4pg-pytorch
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """
        super(CriticD4PG, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc2_units + action_size, fc3_units)
        self.fc3 = nn.Linear(fc3_units, n_atoms)
        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layers
        :return:
        :rtype:
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))

    def forward(self, state, action):
        """
        Forward method implementation. Build a critic (value) network that maps (state, action) pairs -> Q-values.
        :param state: agent state
        :type state: torch Tensor
        :param action: action agent
        :type action: torch Tensor
        :return: Q-values
        :rtype: torch tensor :
        """
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def distr_to_q(self, distr):
        """
        likelihood each bin of each distribution
        :param distr:
        :type distr:
        :return:
        :rtype:
        """
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
