import math
import random
import time
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import csv


class Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = ['fl', 'ff', 'fs', 'el', 'ef', 'es', 'rl', 'rf', 'rs', 'll', 'lf', 'ls', 'sl', 'sf', 'ss']
        self.n_actions = len(self.action_space)
        self.totalstep = 0
        self.final_reward = 0.0
        self.T = 600  # 飞行时间
        self.speed = 1  # V速度
        self.theta = 1  # 最小时间间隙
        self.D = self.T * self.theta  # 最大水平飞行距离
        self.total_uav_power = 10  # uav总功率
        self.total_s_power = 10  # s功率
        self.an = 0  # 功率分配因子
        self.power_UE_use = self.an * self.total_uav_power
        self.power_SD_use = (1 - self.an) * self.total_uav_power
        self.h = 10
        self.xs = 15
        self.ys = 90
        self.zs = 0
        self.xd = 95
        self.yd = 90
        self.zd = 0
        self.xe = 55
        self.ye = 15
        self.ze = 0
        self.xu = 0
        self.yu = 0
        self.zu = self.h
        self.Destination_x = 0
        self.Destination_y = 10
        self.d_ud = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xd, 2) + pow(self.yu - self.yd, 2))  # 距离
        self.d_su = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xs, 2) + pow(self.yu - self.ys, 2))
        self.d_ue = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xe, 2) + pow(self.yu - self.ye, 2))
        self.d_start2end = math.sqrt(
            pow(self.h - self.h, 2) + pow(self.xu - self.Destination_x, 2) + pow(self.yu - self.Destination_y, 2))
        self.B0 = 10 ** (-2)  # 参考距离d0=1m处的信道功率
        self.h_ud = self.B0 * (self.d_ud ** (-2))  # 信道功率增益
        self.h_su = self.B0 * (self.d_su ** (-2))
        self.h_ue = self.B0 * (self.d_ue ** (-2))
        self.h_sd = 10 ** (-7)
        self.N0 = 10 ** (-11)  # 噪声功率
        self.Ysu = (self.total_s_power * self.h_su) / self.N0  # 有效信噪比
        self.Yue = (self.power_UE_use * self.h_ue) / self.N0
        self.Ysd = (self.total_s_power * self.h_sd) / (self.power_SD_use * self.h_ud + self.N0)
        self.Ysue = min(self.Ysu, self.Yue)
        self.Rsue = math.log(1 + self.Ysue, 2) / 2  # 可达率
        self.Rsd = math.log(1 + self.Ysd, 2)
        self.Rev = 0
        self.obs_num = 8
        self.list_steps = []
        self.list_an = []
        self.list_Rsd = []
        self.list_x = []
        self.list_y = []
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_num,), dtype=np.float32)
        self.state = np.zeros((self.obs_num,), dtype=np.float32)
        self.Viewer = None
        self.done = False
        self.massage = 1

    def reset(self):
        random.seed(None)
        self.xu = 110
        self.yu = 45
        self.total_uav_power = 10  # uav总功率
        self.total_s_power = 10  # s功率
        self.an = 0  # 功率分配因子
        self.totalstep = 0
        self.final_reward = 0.0
        self.power_UE_use = self.an * self.total_uav_power
        self.power_SD_use = (1 - self.an) * self.total_uav_power
        self.d_ud = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xd, 2) + pow(self.yu - self.yd, 2))  # 距离
        self.d_su = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xs, 2) + pow(self.yu - self.ys, 2))
        self.d_ue = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xe, 2) + pow(self.yu - self.ye, 2))
        self.d_start2end = math.sqrt(
            pow(self.h - self.h, 2) + pow(self.xu - self.Destination_x, 2) + pow(self.yu - self.Destination_y, 2))
        self.h_ud = self.B0 * (self.d_ud ** (-2))  # 信道功率增益
        self.h_su = self.B0 * (self.d_su ** (-2))
        self.h_ue = self.B0 * (self.d_ue ** (-2))
        self.Ysu = (self.total_s_power * self.h_su) / self.N0  # 有效信噪比
        self.Yue = (self.power_UE_use * self.h_ue) / self.N0
        self.Ysd = (self.total_s_power * self.h_sd) / (self.power_SD_use * self.h_ud + self.N0)
        self.Ysue = min(self.Ysu, self.Yue)
        self.Rsue = math.log(1 + self.Ysue, 2) / 2  # 可达率
        self.Rsd = math.log(1 + self.Ysd, 2)
        self.Rev = 0
        a = [self.xu / 1000, self.yu / 1000, self.an, self.totalstep / 1000, (self.Rsue - self.Rsd) / 100,
             self.d_start2end / 1000, self.Ysd / 1000000, self.Rev / 10]
        self.state = np.asarray(a)
        self.done = False
        return self.state.reshape(self.obs_num, )

    def step(self, action: int):
        self.final_reward = 0
        self.totalstep += 1
        observation_xu = self.xu
        observation_yu = self.yu
        observation_an = self.an
        observation_hud = self.h_ud
        observation_start2end = self.d_start2end
        if action == 0:
            self.xu = self.xu + self.theta * self.speed
            self.yu = self.yu
            self.an = self.an + 0.05
            self.an = round(self.an, 2)
        elif action == 1:
            self.xu = self.xu + self.theta * self.speed
            self.yu = self.yu
            self.an = self.an - 0.05
            self.an = round(self.an, 2)
        elif action == 2:
            self.xu = self.xu + self.theta * self.speed
            self.yu = self.yu
            self.an = self.an
            self.an = round(self.an, 2)
        elif action == 3:
            self.xu = self.xu - self.theta * self.speed
            self.yu = self.yu
            self.an = self.an + 0.05
            self.an = round(self.an, 2)
        elif action == 4:
            self.xu = self.xu - self.theta * self.speed
            self.yu = self.yu
            self.an = self.an - 0.05
            self.an = round(self.an, 2)
        elif action == 5:
            self.xu = self.xu - self.theta * self.speed
            self.yu = self.yu
            self.an = self.an
            self.an = round(self.an, 2)
        elif action == 6:
            self.xu = self.xu
            self.yu = self.yu + self.theta * self.speed
            self.an = self.an + 0.05
            self.an = round(self.an, 2)
        elif action == 7:
            self.xu = self.xu
            self.yu = self.yu + self.theta * self.speed
            self.an = self.an - 0.05
            self.an = round(self.an, 2)
        elif action == 8:
            self.xu = self.xu
            self.yu = self.yu + self.theta * self.speed
            self.an = self.an
            self.an = round(self.an, 2)
        elif action == 9:
            self.xu = self.xu
            self.yu = self.yu - self.theta * self.speed
            self.an = self.an + 0.05
            self.an = round(self.an, 2)
        elif action == 10:
            self.xu = self.xu
            self.yu = self.yu - self.theta * self.speed
            self.an = self.an - 0.05
            self.an = round(self.an, 2)
        elif action == 11:
            self.xu = self.xu
            self.yu = self.yu - self.theta * self.speed
            self.an = self.an
            self.an = round(self.an, 2)
        elif action == 12:
            self.xu = self.xu
            self.yu = self.yu
            self.an = self.an + 0.05
            self.an = round(self.an, 2)
        elif action == 13:
            self.xu = self.xu
            self.yu = self.yu
            self.an = self.an - 0.05
            self.an = round(self.an, 2)
        elif action == 14:
            self.xu = self.xu
            self.yu = self.yu
            self.an = self.an
            self.an = round(self.an, 2)

        self.list_x.append(self.xu)
        self.list_y.append(self.yu)

        if self.an < 0 or self.an > 1:
            self.an = round(observation_an, 2)
            self.final_reward = self.final_reward - 1
        if pow(self.xu - observation_xu, 2) + pow(self.yu - observation_yu, 2) > pow(self.D, 2):
            self.xu = observation_xu
            self.yu = observation_yu
            self.an = round(observation_an, 2)
            self.final_reward = self.final_reward - 0.01

        self.power_UE_use = self.an * self.total_uav_power
        self.power_SD_use = (1 - self.an) * self.total_uav_power
        self.d_ud = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xd, 2) + pow(self.yu - self.yd, 2))  # 距离
        self.d_su = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xs, 2) + pow(self.yu - self.ys, 2))
        self.d_ue = math.sqrt(pow(self.h, 2) + pow(self.xu - self.xe, 2) + pow(self.yu - self.ye, 2))
        self.d_start2end = math.sqrt(
            pow(self.h - self.h, 2) + pow(self.xu - self.Destination_x, 2) + pow(self.yu - self.Destination_y, 2))
        self.h_ud = self.B0 * (self.d_ud ** (-2))  # 信道功率增益
        self.h_su = self.B0 * (self.d_su ** (-2))
        self.h_ue = self.B0 * (self.d_ue ** (-2))
        self.Ysu = (self.total_s_power * self.h_su) / self.N0  # 有效信噪比
        self.Yue = (self.power_UE_use * self.h_ue) / self.N0
        self.Ysd = (self.total_s_power * self.h_sd) / (self.power_SD_use * self.h_ud + self.N0)
        self.Ysue = min(self.Ysu, self.Yue)
        self.Rsue = math.log(1 + self.Ysue, 2) / 2  # 可达率
        self.Rsd = math.log(1 + self.Ysd, 2)
        self.list_Rsd.append(self.Rsd)

        if self.Rsue >= self.Rsd:
            self.Rev = self.Rsd
            self.final_reward = self.final_reward + self.Rsd*1.605
        else:
            self.Rev = 0
            self.final_reward = self.final_reward - self.Rsue*1.605

        if self.d_start2end < observation_start2end:
            self.final_reward = self.final_reward - np.exp(-self.totalstep * 0.002)
        else:
            self.final_reward = self.final_reward - np.exp(self.totalstep * 0.003)
        self.final_reward += -self.d_start2end*0.003

        self.done = False

        if self.xu == self.Destination_x and self.yu == self.Destination_y:
            self.final_reward = self.final_reward + 50
            print("complete")
            self.done = True

        if self.totalstep >= self.T:
            self.final_reward = self.final_reward - 50
            self.done = True

        self.list_an.append(self.an)
        self.list_steps.append(self.totalstep)

        a = [self.xu / 1000, self.yu / 1000, self.an, self.totalstep / 1000, (self.Rsue - self.Rsd) / 100,
             self.d_start2end / 1000, self.Ysd / 1000000, self.Rev / 10]
        self.state = np.asarray(a)
        return self.state.reshape(self.obs_num, ), self.final_reward, self.done, self.massage

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.Viewer is None:
            self.Viewer = rendering.Viewer(500, 500)
        target_s = rendering.make_circle(2)
        target_s.set_color(0, 0, 0)
        target_s_transform = rendering.Transform(translation=(self.xs + 250, self.ys + 250))
        target_s.add_attr(target_s_transform)
        self.Viewer.add_geom(target_s)
        target_d = rendering.make_circle(2)
        target_d.set_color(0, 0, 0)
        target_d_transform = rendering.Transform(translation=(self.xd + 250, self.yd + 250))
        target_d.add_attr(target_d_transform)
        self.Viewer.add_geom(target_d)
        target_e = rendering.make_circle(2)
        target_e.set_color(0, 0, 0)
        target_e_transform = rendering.Transform(translation=(self.xe + 250, self.ye + 250))
        target_e.add_attr(target_e_transform)
        self.Viewer.add_geom(target_e)
        target_u = rendering.make_circle(1)
        target_u.set_color(1, 0, 0)
        target_u_transform = rendering.Transform(translation=(self.xu + 250, self.yu + 250))
        target_u.add_attr(target_u_transform)
        self.Viewer.add_geom(target_u)
        target_end = rendering.make_circle(1)
        target_end.set_color(0, 0, 0)
        target_end_transform = rendering.Transform(translation=(self.Destination_x + 250, self.Destination_y + 250))
        target_end.add_attr(target_end_transform)
        self.Viewer.add_geom(target_end)
        return self.Viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.Viewer:
            self.Viewer.close()
            self.Viewer = None
        return

    def plot_an(self):
        plt.figure(1)
        plt.plot(self.list_steps, self.list_an)
        plt.ylabel('alpha[n]')
        plt.xlabel('Step')
        plt.savefig("Alpha.jpg")
        with open('alpha[n].csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(self.list_steps)):
                writer.writerow([i, self.list_an[i]])
        print("success")

    def plot_rsd(self):
        plt.figure(2)
        plt.plot(self.list_steps, self.list_Rsd)
        plt.ylabel('Rsd')
        plt.xlabel('Step')
        plt.savefig("Rsd.jpg")
        with open('Rsd.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(self.list_steps)):
                writer.writerow([i, self.list_Rsd[i]])
        print("success")

    def steps(self, type_name):
        with open(type_name + '_steps.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for i_p in range(len(self.list_steps)):
                writer.writerow([self.list_x[i_p], self.list_y[i_p]])
        print("success")


if __name__ == '__main__':
    env = Env()
    env.reset()
    i = 0
    total = 0
    while True:
        env.render()
        if i < 30:
            num = 10
        elif i < 60:
            num = 0
        else:
            num = 2
        i += 1
        time.sleep(0.1)
        obs, reward, done, message = env.step(num)

        print("obs={},reward={},done={},message={}".format(obs, reward, done, message))  # the reward is one step !
        total = total + reward
        if done:
            env.render()
            break

    print(total)
