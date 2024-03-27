import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import rl_utils
from td3 import TD3
from env import UAVEnvironment


env_name = 'UAV'
env = UAVEnvironment()

actor_lr = 5e-5
critic_lr = 1e-4
num_episodes = 500
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
sigma = 0.01  # 高斯噪声标准差
buffer_size = 50000
minimal_size = 2000
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = TD3(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, gamma, tau, device)

agent.load_net()

return_list = []
Reav_list = []
Rd_list = []

for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            episode_reav = 0
            episode_rd = 0
            state = env.reset()
            done = False

            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)

                # print(state)

                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                episode_rd += next_state[6] / 0.01
                episode_reav += next_state[8] / 0.01

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)

                if done:
                    episode_reav = episode_reav / (next_state[4] * env.time)

            return_list.append(episode_return)
            Rd_list.append(episode_rd)
            Reav_list.append(episode_reav)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

agent.save_net()
# agent.plot_loss()

env.close()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.savefig("Reward.png")
plt.show()

with open('Reward.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(episodes_list)):
        writer.writerow([i, return_list[i]])

plt.plot(episodes_list, Reav_list)
plt.xlabel('Episodes')
plt.ylabel('Reav')
plt.title('DDPG on {}'.format(env_name))
plt.savefig("Reav.png")
plt.show()

with open('Reav.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(episodes_list)):
        writer.writerow([i, Reav_list[i]])

plt.plot(episodes_list, Rd_list)
plt.xlabel('Episodes')
plt.ylabel('Rd')
plt.title('DDPG on {}'.format(env_name))
plt.savefig("Rd.png")
plt.show()

with open('Rd.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(episodes_list)):
        writer.writerow([i, Rd_list[i]])

'''mv_reav = rl_utils.moving_average(Reav_list, 9)
plt.plot(episodes_list, mv_reav)
plt.xlabel('Episodes')
plt.ylabel('Reav')
plt.title('DDPG on {}'.format(env_name))
plt.savefig("Reav_.png")
plt.show()'''
