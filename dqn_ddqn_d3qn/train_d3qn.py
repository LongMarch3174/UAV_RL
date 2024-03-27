import random
import numpy as np
import torch
import rl_utils

from DQN_DDQN_D3QN import DQN
from env import Env
from train_method import train_DQN, plot

lr = 1e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 1000
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'UAV'
env = Env()
state_dim = env.observation_space.shape[0]
action_dim = 15  # 将连续动作分成11个离散动作

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device, 'DuelingDoubleDQN')

# agent.load_net("DuelingDoubleDQN")

return_list, rev_list = train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size)

agent.save_net("DuelingDoubleDQN")
agent.plot_loss()

plot(return_list, rev_list, env_name, "DuelingDoubleDQN")
