import csv
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import rl_utils
from SAC import SAC
from env import UAVEnvironment


env_name = 'UAV-IRS'
env = UAVEnvironment()
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

actor_lr = 1e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 200
hidden_dim = 256
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 2000
batch_size = 128
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

agent.load_net()
agent.actor.eval()
agent.critic_1.eval()
agent.critic_2.eval()

R_eav = []
x = []
y = []

for i_episode in range(1):
    state = env.reset()
    done = False
    while not done:
        #env.render()
        action = agent.take_action_eval(state)
        print("动作为: ", action)

        next_state, reward, done, _ = env.step(action)
        x.append(next_state[0] * 100)
        y.append(next_state[1] * 100)
        print(env.current_step, " X:", next_state[0] * 3000, " Y:", next_state[1] * 1000, " Z:", next_state[2] * 4000,
              " Dis:", next_state[4] * 5000, "Reav", next_state[7] / 0.1, "k:", action[3], "reward:", reward)

        R_eav.append(next_state[6] * 10)

        state = next_state

plt.figure(2)
plt.plot(R_eav)
plt.ylabel('R_EAV')
plt.xlabel('Step')
plt.savefig("R_EAV.png")
plt.show()

with open('steps.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(x)):
        writer.writerow([x[i], y[i]])

time.sleep(1)
env.close()
