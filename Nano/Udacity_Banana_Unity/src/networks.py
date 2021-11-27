import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import random

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        # First layer input defined by a vector size observations space and
        # 128 nodes. Hidden layer 128 nodes and output layer size equal to
        # number of actions of the env
        # Candidate parameters to be optimized are the number of nodes on
        # the hidden layer
        # Activation function ReLU(x)=(x) + =max(0,x)

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
    # forward propagation
    def forward(self, x):
        return self.layers(x)

class DDQN(nn.Module):
    """
    Dueling DQ Network . State + advantage value functions
    """
    def __init__(self, num_inputs, num_outputs):
        super(DDQN, self).__init__()
       # input layer = size observation spaces
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU()
        )
     # hidden layer which calculate the state-value function
     # output as in DQN number of possible actions
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )
      # hidden layer for advantage function
      # output continuous value for advantage function
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
   # forward propagation of calculations and combination advantage and value function
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

class Categorical_DQN(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int, # number of bins
            support: torch.Tensor
    ):
        """
            Initialization. The Network is the same than a plain DQN
        """
        super(Categorical_DQN, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward method implementation. The output of this Network is a matrix size number of actions (rows)
            where each action have atom_size slices / bins of a distribution
         """
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get distribution for atoms. Here we get a distribution per each action represented by a N (atom_size)
        discrete bins. We get these distributions applying softmax to each bin
        """
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # clapping the lower band for avoiding nans

        return dist

class NoisyLinear(nn.Module):
    """
    Noisy linear module for NoisyNet.
    References:

    https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
    https://github.com/Kaixhin/Rainbow/blob/master/model.py

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Noisy_DuelingNetwork(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,

    ):
        """Initialization."""
        super(Noisy_DuelingNetwork, self).__init__()


        self.out_dim = out_dim

        # set common feature layer. it doesn’t change from original implementation
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer, which now is a noisy layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim)

        # set value layer , which now is a noisy layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)
        # forward is defined as the value of this action together with the advantage to take this action
        # -minus the mean of the expected advantage function values from this state to done
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q


    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

class Rainbow_DQN(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super(Rainbow_DQN, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
    	Forward method implementation.
        """

        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """
    	Get distribution for atoms/ Bins
        """
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        # advantage layer shape number of output * number of atoms/bins
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size)
        # value layer shape 1 * number of atoms/bins
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        # we apply the softmax function to the combined layer
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # clapping for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()