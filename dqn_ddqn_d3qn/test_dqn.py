import random
import time
import numpy as np
import torch

from DQN_DDQN_D3QN import DQN
from env_v0 import Env
from train_method import test


lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 50
buffer_size = 5000
minimal_size = 2000
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'UAV'
env = Env()
state_dim = env.observation_space.shape[0]
action_dim = 15

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

agent.load_net("DQN")
agent.q_net.eval()
agent.target_q_net.eval()

test(agent, env, "DQN")

env.steps("DQN")
time.sleep(1)
env.close()
