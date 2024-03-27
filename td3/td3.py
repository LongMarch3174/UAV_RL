import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

directory = "model/"


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = F.relu(self.l1(state))
        action = torch.tanh(self.l2(x))
        action = action * self.action_bound
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        cat = torch.cat([state, action], 1)
        x = F.relu(self.l1(cat))
        x = F.relu(self.l2(x))
        return self.l_out(x)


class TD3(object):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, gamma, tau, device, policy_noise=0.1, noise_clip=0.5, policy_freq=4):
        self.actor = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.actor_target = Actor(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, hidden_dim, action_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.sigma = sigma
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()  # This will give you a numpy array of actions
        # Add noise to each action dimension independently to increase exploration
        noise = self.sigma * np.random.randn(self.action_dim)
        action = action + noise
        return action

    def take_action_eval(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()  # This will give you a numpy array of actions
        return action

    def update(self, transition_dict):
        self.total_it += 1

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = torch.ones_like(actions).data.normal_(0, self.policy_noise).to(self.device)
            next_action = (self.actor_target(next_states) + noise).clamp(-self.action_bound, self.action_bound)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_states, next_action)
            target_Q2 = self.critic_2_target(next_states, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (self.gamma * target_Q * (1 - dones)).detach()

        # Optimize Critic 1:
        current_Q1 = self.critic_1(states, actions)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2:
        current_Q2 = self.critic_2(states, actions)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates:
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss:
            actor_loss = - self.critic_1(states, self.actor(states)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

    def save_net(self):
        torch.save(self.actor.state_dict(), directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), directory+'critic_2_target.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load_net(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(directory + 'critic_2_target.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
