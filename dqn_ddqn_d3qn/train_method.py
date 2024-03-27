import numpy as np
from tqdm import tqdm

import csv
import matplotlib.pyplot as plt


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    rev_list = []

    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                episode_rev = 0

                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值

                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state

                    episode_return += reward
                    episode_rev += state[7] * 10

                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)

                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }

                        agent.update(transition_dict)

                return_list.append(episode_return)
                episode_rev = episode_rev / (state[3] * 1000)
                rev_list.append(episode_rev)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    return return_list, rev_list


def plot(return_plot_list, rev_plot_list, env_plot_name, type_name):
    episodes_list = list(range(len(return_plot_list)))
    plt.plot(episodes_list, return_plot_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(type_name + ' on {}'.format(env_plot_name))
    plt.savefig(type_name + "_Reward.jpg")
    plt.show()

    plt.plot(episodes_list, rev_plot_list)
    plt.xlabel('Episodes')
    plt.ylabel('Rev')
    plt.title(type_name + ' on {}'.format(env_plot_name))
    plt.savefig(type_name + "_Rev.jpg")
    plt.show()

    with open(type_name + '_reward.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(return_plot_list)):
            writer.writerow([i, return_plot_list[i]])

    with open(type_name + '_rev.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(return_plot_list)):
            writer.writerow([i, rev_plot_list[i]])


def test(agent, env, type_name, env_plot_name):
    state = env.reset()
    done = False
    while not done:
        # env.render()

        action = agent.take_action_eval(state)

        next_state, reward, done, _ = env.step(action)
        state = next_state

        print(state[0]*1000, state[1]*1000)

    plt.plot(env.list_steps, env.list_an)
    plt.xlabel('Episodes')
    plt.ylabel('Alpha')
    plt.title(type_name + ' on {}'.format(env_plot_name))
    plt.savefig(type_name + "_Alpha.jpg")
    plt.show()

    with open(type_name + '_Alpha.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(env.list_steps)):
            writer.writerow([i, env.list_an[i]])
