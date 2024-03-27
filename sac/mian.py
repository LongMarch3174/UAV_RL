import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import rl_utils
from SAC import SAC
from env import UAVEnvironment


env_name = 'UAV-IRS'
env = UAVEnvironment()
env.read_gauss()
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
minimal_size = 1000
batch_size = 128
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SAC(state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device)

agent.load_net()

return_list = []
Reav_list = []
Rsd_list = []
Rsl_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            episode_reav = 0
            episode_rsd = 0
            episode_rsl = 0

            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state

                episode_return += reward
                episode_rsd += next_state[4] * 10
                episode_rsl += next_state[5] * 10
                episode_reav += next_state[6] * 10

                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    agent.update(transition_dict)

                if done:
                    episode_reav = episode_reav / (next_state[2] * 200)
                    episode_rsd = episode_rsd / (next_state[2] * 200)
                    episode_rsl = episode_rsl / (next_state[2] * 200)

            return_list.append(episode_return)
            Reav_list.append(episode_reav)
            Rsd_list.append(episode_rsd)
            Rsl_list.append(episode_rsl)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                  'return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

agent.save_net()
# agent.plot_loss()
agent.write_close()

env.close()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.savefig("Reward.png")
plt.show()

with open('Reward.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(episodes_list)):
        writer.writerow([i, return_list[i]])

plt.plot(episodes_list, Reav_list)
plt.xlabel('Episodes')
plt.ylabel('Reav')
plt.title('SAC on {}'.format(env_name))
plt.savefig("Reav.png")
plt.show()

with open('Reav.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(episodes_list)):
        writer.writerow([i, Reav_list[i]])

mv_reav = rl_utils.moving_average(Reav_list, 9)
plt.plot(episodes_list, mv_reav)
plt.xlabel('Episodes')
plt.ylabel('Reav')
plt.title('SAC on {}'.format(env_name))
plt.savefig("Reav_.png")
plt.show()

plt.plot(episodes_list, Rsd_list)
plt.xlabel('Episodes')
plt.ylabel('Rsd')
plt.title('SAC on {}'.format(env_name))
plt.savefig("Rsd.png")
plt.show()

plt.plot(episodes_list, Rsl_list)
plt.xlabel('Episodes')
plt.ylabel('Rsl')
plt.title('SAC on {}'.format(env_name))
plt.savefig("Rsl.png")
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.savefig("Reward_.png")
plt.show()
