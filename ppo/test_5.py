import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = (torch.tanh(self.fc_mu(x)) + 1)*np.pi
        std = F.softplus(self.fc_std(x))
        return mu, std


class PPO:
    """ 处理连续动作的PPO算法 """
    def __init__(self, state_dim, hidden_dim, action_dim):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim)

    def take_action(self, state):
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action


if __name__ == '__main__':
    observation = torch.Tensor(4)
    P_ = PPO(4, 20, 2)
    a = P_.take_action(observation)
    print("action: ", a, (a+1)*2)
