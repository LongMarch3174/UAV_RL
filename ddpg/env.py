import gym
import numpy as np
from scipy.spatial import distance
import re

"""
球坐标系(r, θ, φ) --> 笛卡尔坐标系(x, y, z) 
定义域 0<=θ<=π, 0<=φ<2π
x=r*sin(θ)cos(φ)
y=r*sin(θ)sin(φ)
z=r*cos(θ)

欺骗方案
"""

P_S = 10
P_E = 10
P_U = 1000
SIGMA2 = DELTA2 = 1e-26
OMEGA = 1e-3
EPSILON = 1
AIOT = 1

MAX_DISTANCE = 20

H_MAX = 100
H_MIN = 50
X_MAX = 3000
X_MIN = 0
Y_MAX = 1000
Y_MIN = 0
T = 200

np.random.seed(1437)


class UAVEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # 定义环境参数
        self.state_num = 9
        self.action_num = 4
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]),
                                           shape=(self.action_num,))  # 定义动作空间
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_num,))  # 定义状态空间的维度

        self.X = X_MAX - X_MIN
        self.Y = Y_MAX - Y_MIN
        self.Z = H_MAX - H_MIN

        self.start_point = (0, 1000, 75)  # 起点坐标
        self.end_point = (3000, 1000, 100)  # 终点坐标
        self.S = (1500, 0, 0)  # S 位置
        self.D = (2000, 0, 0)  # D 位置
        self.time_interval = 1  # 时间间隔

        self.rho = 0
        self.k = 0

        self.distance_to_end = distance.euclidean(self.start_point, self.end_point)
        self.last_d = distance.euclidean(self.start_point, self.end_point)
        self.current_step = 0  # 当前时间步
        self.time = T
        self.current_position = self.start_point  # 当前无人机位置

        """self.d_SD = self._get_d_sd()  # d_SD距离
        self.d_SU = self._get_d_su()  # d_SU距离
        self.d_UD = self._get_d_ud()  # d_UD距离

        self.h_SU = self._get_h_su()  # h_SU
        self.h_UD = self._get_h_ud()  # h_UD
        self.h_SD = self._get_h_sd()  # h_SD"""

        self.Gamma_SD = self._get_gamma_sd()  # γ_SD
        self.Gamma_SU = self._get_gamma_su()  # γ_SU

        self.R_D = self._get_r_d()  # R_D
        self.R_U = self._get_r_u()  # R_U

        self.R_EAV = self._get_r_eav()  # R_EAV
        self.last_R_EAV = self.R_EAV

        self.h = self.current_position[2]
        self.constraints = 0
        self.k_bound = 1e16

        self.Viewer = None  # render

    def reset(self):
        self.current_step = 0
        self.current_position = self.start_point
        self.distance_to_end = distance.euclidean(self.start_point, self.end_point)
        self.last_d = distance.euclidean(self.start_point, self.end_point)

        self.Gamma_SD = self._get_gamma_sd()  # γ_SD
        self.Gamma_SU = self._get_gamma_su()  # γ_SU

        self.R_D = self._get_r_d()
        self.R_U = self._get_r_u()

        self.R_EAV = self._get_r_eav()  # R_EAV
        self.last_R_EAV = self.R_EAV

        self.h = self.current_position[2]
        self.constraints = 0
        self.rho = 0
        self.k = 0

        return self._get_observation()

    def _get_observation(self):
        # 获取当前状态向量
        state = np.zeros(self.state_num)
        state[0] = self.current_position[0] / 3000  # 当前位置 x
        state[1] = self.current_position[1] / 1000  # 当前位置 y
        state[2] = self.current_position[2] / 4000  # 当前位置 z
        state[3] = self.h / 4000
        state[4] = self.current_step / self.time  # 当前时间步
        state[5] = self.distance_to_end / 5000
        state[6] = self.R_D * 0.01
        state[7] = self.R_U * 0.01
        state[8] = self.R_EAV * 0.01

        return state

    def step(self, action):
        self.current_step += self.time_interval
        self._take_action(action)

        next_state = self._get_observation()

        done = self._is_done()

        reward = self._get_reward()

        info = {}

        # print(self.current_position)
        return next_state, reward, done, info

    def _take_action(self, action):
        self.last_d = self.distance_to_end
        self.last_R_EAV = self.R_EAV

        # 解析动作值
        action = (action + 1) / 2

        theta = np.clip(action[0], 0, 1)  # θ范围 [0, π]
        phi = np.clip(action[1], 0, 1)  # φ范围 [0, 2π]
        rho = np.clip(action[2], 0, 1)
        k = np.clip(action[3], 0, 1)

        theta = theta * np.pi
        phi = phi * 2 * np.pi
        rho = rho * 1
        k = k * self.k_bound

        # 根据立体角和行驶距离更新无人机位置
        delta_x = np.sin(theta) * np.cos(phi) * MAX_DISTANCE
        delta_y = np.sin(theta) * np.sin(phi) * MAX_DISTANCE
        delta_z = np.cos(theta) * MAX_DISTANCE
        self.current_position = (
            self.current_position[0] + delta_x,
            self.current_position[1] + delta_y,
            self.current_position[2] + delta_z
        )
        self.h = self.h + delta_z

        self.rho = rho
        self.k = k
        self.constraints = self._get_constraints()

        if self.current_position[2] > H_MAX:
            self.current_position = (
                self.current_position[0],
                self.current_position[1],
                H_MAX
            )
        elif self.current_position[2] < H_MIN:
            self.current_position = (
                self.current_position[0],
                self.current_position[1],
                H_MIN
            )
        else:
            self.h = self.current_position[2]

        # 更新UAV到终点的距离
        self.distance_to_end = distance.euclidean(self.current_position, self.end_point)

        self.Gamma_SD = self._get_gamma_sd()  # γ_SD
        self.Gamma_SU = self._get_gamma_su()  # γ_SU

        self.R_D = self._get_r_d()
        self.R_U = self._get_r_u()

        self.R_EAV = self._get_r_eav()

    def _get_reward(self):
        # 根据任务目标设计奖励函数
        # 最小化距离
        """if self.last_d > self.distance_to_end:
            dis_reward = 0
        else:
            dis_reward = 0"""
        dis_reward = -self.distance_to_end * 0.0010

        if self.R_D <= self.R_U:
            Reav_reward = self.R_D * 0.05
        else:
            Reav_reward = -1 + self.R_D * 0.02

        if not self._verify_constraints():
            constraint_reward = -0.5
        else:
            constraint_reward = 0

        if H_MIN <= self.h <= H_MAX:
            h_reward = 0
        else:
            h_reward = -np.abs(self.h - 75) * 0.005  # 假设75是理想高度

        alpha = 0.5  # 权重alpha用于平衡飞向终点目标
        beta = 0.1  # 权重beta用于平衡Reav目标
        gamma = 0.1
        delta = 0.3

        # print(self.Gamma_SD, self.Gamma_SU, self.R_D, self.R_U, self.k, self._get_h_su(), self._get_h_sd(), self._get_h_ud())
        # print(dis_reward, h_reward)

        reward = alpha * dis_reward + beta * Reav_reward + gamma * constraint_reward + delta * h_reward

        if self.distance_to_end < 10:
            done_reward = 40
            reward += done_reward

        if self.current_step >= self.time:
            time_reward = -40
            reward += time_reward

        return reward

    def _is_done(self):
        if self.distance_to_end < 10:
            print("complete")
            return True

        if self.current_step >= self.time:
            return True

        '''if not self._verify_constraints():
            return True'''

        '''if not H_MIN <= self.current_position[2] <= H_MAX:
            return True'''

        return False

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.Viewer is None:
            self.Viewer = rendering.Viewer(500, 500)
        '''target_s = rendering.make_circle(2)
        target_s.set_color(0, 0, 0)
        target_s_transform = rendering.Transform(translation=(self.xs + 250, self.ys + 250))
        target_s.add_attr(target_s_transform)
        self.Viewer.add_geom(target_s)
        target_d = rendering.make_circle(2)
        target_d.set_color(0, 0, 0)
        target_d_transform = rendering.Transform(translation=(self.xd + 250, self.yd + 250))
        target_d.add_attr(target_d_transform)
        self.Viewer.add_geom(target_d)'''
        target_end = rendering.make_circle(2)
        target_end.set_color(0, 0, 0)
        target_end_transform = rendering.Transform(translation=(self.end_point[0] + 250, self.end_point[1] + 250))
        target_end.add_attr(target_end_transform)
        self.Viewer.add_geom(target_end)
        target_uav = rendering.make_circle(1)
        target_uav.set_color(1, 0, 0)
        target_u_transform = rendering.Transform(
            translation=(self.current_position[0] + 250, self.current_position[1] + 250))
        target_uav.add_attr(target_u_transform)
        self.Viewer.add_geom(target_uav)
        return self.Viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.Viewer:
            self.Viewer.close()
            self.Viewer = None
        return

    def _get_d_sd(self):
        d_SD = distance.euclidean(self.S, self.D)
        return d_SD

    def _get_d_su(self):
        d_SU = distance.euclidean(self.S, self.current_position)
        return d_SU

    def _get_d_ud(self):
        d_UD = distance.euclidean(self.current_position, self.D)
        return d_UD

    def _get_h_su(self):
        h_SU = OMEGA * (self._get_d_su() ** -2)
        return h_SU

    def _get_h_ud(self):
        h_UD = OMEGA * (self._get_d_ud() ** -2)
        return h_UD

    def _get_h_sd(self):
        h_SD = OMEGA * (self._get_d_sd() ** -EPSILON) * AIOT
        return h_SD

    def _get_gamma_sd(self):
        numerator = (np.abs(
            self._get_h_sd() + self.k * np.sqrt(self.rho) * self._get_h_su() * self._get_h_ud()) ** 2) * P_S
        denominator = (1 + self.k ** 2 * self._get_h_ud() ** 2) * SIGMA2
        Gamma_SD = numerator / denominator
        return Gamma_SD

    def _get_gamma_su(self):
        Gamma_SU = ((1 - self.rho) * self._get_h_su() ** 2 * P_S) / SIGMA2
        return Gamma_SU

    def _get_r_d(self):
        R_D = np.log2(1 + self._get_gamma_sd())
        return R_D

    def _get_r_u(self):
        R_U = np.log2(1 + self._get_gamma_su())
        return R_U

    def _get_r_eav(self):
        if self.R_D <= self.R_U:
            R_EAV = self.R_D
        else:
            R_EAV = 0

        return R_EAV

    def _get_constraints(self):
        constraints = (abs(self.k)) ** 2 * (self.rho * (abs(self._get_h_su())) ** 2 * P_S + SIGMA2)
        return constraints

    def _verify_constraints(self):
        if ((abs(self.k)) ** 2 * (self.rho * (abs(self._get_h_su())) ** 2 * P_S + SIGMA2)) <= P_U:
            return True
        else:
            return False
