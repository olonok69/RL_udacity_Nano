import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.distributions import Normal

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(0.001)
        nn.init.constant_(m.bias.data, 0)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, seed, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic=True

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # # Q2 architecture
        # self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        # self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # x2 = F.relu(self.linear4(xu))
        # x2 = F.relu(self.linear5(x2))
        # x2 = self.linear6(x2)

        return x1


class GaussianPolicy(nn.Module):
    def __init__(self, seed, num_inputs, num_actions, hidden_dim, action_space=None,
                 min=-20, max=2, epsilon=1e-6):
        super(GaussianPolicy, self).__init__()

        torch.manual_seed(4)
        torch.cuda.manual_seed(4)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic=True
        self.epsilon = epsilon
        self.min = min
        self.max = max
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
#        if action_space is None:
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)


    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.min, max=self.max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2).clamp(0,1)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

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

class A2CModel(nn.Module):
    def __init__(self,state_size, action_size, device):
        """
        Instantiate object A2CModel
        :param state_size:
        :type state_size:
        :param action_size:
        :type action_size:
        :param seed:
        :type seed:
        :param device:
        :type device:
        """
        super(A2CModel, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)
        self.std = torch.ones(action_size).to(device)
        self.dist = torch.distributions.Normal

    def forward(self, s):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states
        '''
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))

        return s

    def get_action(self, s):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states
        Returns
        ======
            action_tanh (n_process, action_size) (tensor): action limited within (-1,1)
            action (n_process, action_size) (tensor): raw action
        '''
        s = self.forward(s)
        mu = self.actor(s)
        dist_ = self.dist(mu, self.std)
        action = dist_.sample()
        action_tanh = torch.tanh(action)
        return action_tanh, action

    def get_action_prob(self, s, a):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states
            a (n_process, action_size) (tensor): actions

        Returns
        =======
            mu (n_process, action_size) (tensor): mean value of action distribution
            self.std (action_size,) (tensor): the standard deviation of every action
            log_prob (n_process,) (tensor): log probability of input action
        '''

        s = self.forward(s)
        mu = self.actor(s)
        dist_ = self.dist(mu, self.std)
        log_prob = dist_.log_prob(a)
        log_prob = torch.sum(log_prob, dim=1, keepdim=False)
        return mu, self.std, log_prob

    def get_state_value(self, s):
        '''
        Params
        ======
            s (n_process, state_size) (tensor): states
        Returns
        =======
            value (n_process,) (tensor)
        '''
        s = self.forward(s)
        value = self.critic(s).squeeze(1)
        return value

class ActorCritic_PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs, seed, hidden_size_1=64, hidden_size_2=64):
        super(ActorCritic_PPO, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.shared = nn.Sequential(
            nn.Linear(num_inputs, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_size_2, num_outputs),
            nn.Tanh()
        )
        self.actor.apply(init_weights)

        self.critic = nn.Linear(hidden_size_2, 1)
        self.critic.apply(init_weights)

        self.std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        x = torch.FloatTensor(x)
        x = self.shared(x)
        value = self.critic(x)
        mu = self.actor(x)
        dist = Normal(mu, F.softplus(self.std))

        return dist, value

class ActorCriticNetwork_A2C(nn.Module):
    """
    The actor critic network
    The Actor and the Critic Share the same input encoder
    """

    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork_A2C, self).__init__()
        self.state_dim =state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 128)

        # Actor head: output mean and std
        self.actor_fc = nn.Linear(128, 128)
        self.actor_out = nn.Linear(128, action_dim)

        self.std = nn.Parameter(torch.ones(1, action_dim, dtype=torch.float))

        # critic head: output state value
        self.critic_fc = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)
        self.reset_parameters()

    def forward(self, state):
        """
        Compute forward pass
        Input: state tensor
        Output: tuple of (clampped action, log probabilities, state values)
        """
        x = F.relu(self.fc1(state))
        mean = self.actor_out(F.relu(self.actor_fc(x)))
        # std=0
        # if torch.any(self.std) <= 0:
        #     self.std = nn.Parameter(torch.ones(1, self.action_dim))
        # else:
        #     std = self.std
        #self.std = torch.where(self.std > 0, self.std, 0.01)
        dist = torch.distributions.Normal(mean, self.std, validate_args=False)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic_out(F.relu(self.critic_fc(x)))
        return torch.clamp(action, -1, 1), log_prob, value

    def reset_parameters(self):
        """
        Reset parameters to the initial states
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.actor_fc.weight.data.uniform_(*hidden_init(self.actor_fc))
        self.critic_fc.weight.data.uniform_(*hidden_init(self.critic_fc))
        self.actor_out.weight.data.uniform_(-3e-3, 3e-3)
        self.critic_out.weight.data.uniform_(-3e-3, 3e-3)

class Policy_SAC(nn.Module):

    def __init__(self, state_size, action_size=1, n_agents=1, fc1_size=128, fc2_size=128, init_w=3e-3):
        super(Policy_SAC, self).__init__()

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(fc1_size)
        self.fc3_mu = nn.Linear(fc2_size, action_size)
        self.fc3_mu.weight.data.uniform_(-init_w, init_w)
        self.fc3_mu.bias.data.uniform_(-init_w, init_w)

        self.fc3_std = nn.Linear(fc2_size, action_size)
        self.fc3_std.weight.data.uniform_(-init_w, init_w)
        self.fc3_std.bias.data.uniform_(-init_w, init_w)
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state, log_std_min=-20, log_std_max=2):
        #x = self.bn0(state)
        x = torch.relu(self.bn1(self.fc1(state)))
        x = torch.relu(self.bn2(self.fc2(x)))

        mean = self.fc3_mu(x)
        std = self.fc3_std(x)
        std = torch.clamp(std, log_std_min, log_std_max)#.exp()

        return mean, std

    def evaluate(self, state, device, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        #print(mean)
        normal = Normal(mean, std)
        #z = normal.sample()  # for reparameterization trick (mean + std * N(0,1))
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        action.to(device)
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob = log_prob.clone() - torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        # log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, x_t, mean, log_std

    def get_action(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()
        return action[0]

class Value_SAC(nn.Module):

    def __init__(self, state_size, action_size=1, n_agents=1, fc1_size=128, fc2_size=128, init_w=3e-3):
        super(Value_SAC, self).__init__()

        #self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    # def forward(self, x):
    #     #x = self.bn0(x)
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     return self.fc3(x)
    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        #x = self.bn0(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Q_SAC(nn.Module):

    def __init__(self, state_size, action_size, n_agents=1, fc1_size=128, fc2_size=128, init_w=3e-3):
        super(Q_SAC, self).__init__()

        #self.bn0 = nn.BatchNorm1d(state_size + action_size)
        self.fc1 = nn.Linear(state_size + action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        #x = self.bn0(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)