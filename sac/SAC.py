import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

log_dir = "./logs"


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = ((action + 1) / 2) * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim*2)
        self.fc_out = torch.nn.Linear(hidden_dim*2, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x


class SAC:
    """ 处理连续动作的SAC算法 """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy,
                 tau, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim, action_dim).to(device)  # 第二个目标Q网络

        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.ac_loss = []
        self.step = 0
        self.writer_ac = SummaryWriter(log_dir=log_dir)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, _ = self.actor(state)
        action = action.cpu().detach().numpy().reshape(-1)
        return action

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = (torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy).mean(dim=1, keepdim=True)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        self.step += 1

        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.ac_loss.append(actor_loss.cpu().item())
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                self.writer_ac.add_scalar(f'ac Gradients/{name}', param.grad.norm(), self.step)

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def take_action_eval(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action, _ = self.actor(state)
        action = action.cpu().detach().numpy().reshape(-1)
        return action

    def save_net(self):
        torch.save(self.actor.state_dict(), "actor_model.pth")
        torch.save(self.critic_1.state_dict(), "critic1_model.pth")
        torch.save(self.critic_2.state_dict(), "critic2_model.pth")
        torch.save(self.target_critic_1.state_dict(), "t_critic1_model.pth")
        torch.save(self.target_critic_2.state_dict(), "t_critic2_model.pth")
        torch.save({'log_alpha': self.log_alpha}, "alpha.pth")

    def load_net(self):
        self.actor.load_state_dict(torch.load("actor_model.pth"))
        self.critic_1.load_state_dict(torch.load("critic1_model.pth"))
        self.critic_2.load_state_dict(torch.load("critic2_model.pth"))
        """self.target_critic_1.load_state_dict(torch.load("t_critic1_model.pth"))
        self.target_critic_2.load_state_dict(torch.load("t_critic2_model.pth"))
        alpha_checkpoint = torch.load("alpha.pth")
        self.log_alpha.data = alpha_checkpoint['log_alpha']"""

    '''def plot_loss(self):
        ac_list = list(range(len(self.ac_loss)))
        plt.plot(ac_list, self.ac_loss)
        plt.xlabel('Episodes')
        plt.ylabel('ac_loss')
        plt.title('SAC on {}'.format("UAV"))
        plt.savefig("loss_ac.png")
        plt.show()

        cr_list = list(range(len(self.loss_cr)))
        plt.plot(cr_list, self.loss_cr)
        plt.xlabel('Episodes')
        plt.ylabel('cr_loss')
        plt.title('DDPG on {}'.format("UAV"))
        plt.savefig("loss_cr.png")
        plt.show()'''

    def write_close(self):
        self.writer_ac.close()
