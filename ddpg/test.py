import csv
import random
import time

import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import rl_utils
from ddpg import DDPG
from env import UAVEnvironment


env_name = 'UAV'
env = UAVEnvironment()

actor_lr = 3e-5
critic_lr = 3e-4
num_episodes = 500
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 2000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

agent.load_net()
agent.actor.eval()
agent.critic.eval()

x = []
y = []
z = []
rho = []
k = []
reav = []

for i_episode in range(1):
    state = env.reset()
    done = False
    while not done:
        # env.render()
        action = agent.take_action_eval(state)
        print("动作为: ", action)
        rho.append(action[2])
        k.append(action[3])

        next_state, reward, done, _ = env.step(action)
        print(next_state)
        x.append(next_state[0] * 3000)
        y.append(next_state[1] * 1000)
        z.append(next_state[2] * 4000)
        reav.append(next_state[8] / 0.01)
        print(env.current_step, " X:", next_state[0] * 3000, " Y:", next_state[1] * 1000, " Z:", next_state[2] * 4000,
              " Dis:", next_state[5] * 5000, "Reav", next_state[8] / 0.1, "k:", action[3], "reward:", reward)

        state = next_state

# 绘制三维轨迹图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)

# 设置图形属性
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D UAV Trajectory')
plt.savefig("UAV_Trajectory.png")

# 显示图形
plt.show()
steps_list = list(range(len(rho)))

with open('steps.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(x)):
        writer.writerow([x[i], y[i], z[i]])

with open('reav_step.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(x)):
        writer.writerow([steps_list[i], reav[i]])

with open('rho.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(x)):
        writer.writerow([steps_list[i], rho[i]])

with open('k.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(x)):
        writer.writerow([steps_list[i], k[i]])

time.sleep(1)
env.close()
