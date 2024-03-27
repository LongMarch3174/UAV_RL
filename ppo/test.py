import csv
import random
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import rl_utils
from PPO import PPO
from env import UAVEnvironment


env_name = 'UAV-IRS'
env = UAVEnvironment()
env.read_gauss()
random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

actor_lr = 1e-4
critic_lr = 5e-3
num_episodes = 2000
hidden_dim = 256
gamma = 0.9
lmbda = 0.9
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]  # 连续动作空间
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
agent.load_net()
agent.actor.eval()
agent.critic.eval()

R_eav = []
x = []
y = []

for i_episode in range(1):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.take_action(state)
        print("动作为: ", action)

        next_state, reward, done, _ = env.step(action)
        x.append(next_state[0] * 100)
        y.append(next_state[1] * 100)
        print(env.current_step, " X:", next_state[0] * 100, " Y:", next_state[1] * 100, " Dis:", next_state[3] * 100,
              " R_EAV:", next_state[6] * 10)
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
