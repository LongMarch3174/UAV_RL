import gym
import numpy as np
import re

MAX_DISTANCE = 1
UAV_H = 10

P = P_S = P_U = 0.03
SIGMA_2 = 10**(-7.326)
BETA_0 = 10**(-3)
K = 2.2
RHO = 10**(-3)

np.random.seed(42)


def _get_gauss():
    num_samples = 10

    real_part = np.random.normal(loc=0, scale=1, size=num_samples)
    imag_part = np.random.normal(loc=0, scale=1, size=num_samples)

    complex_samples = real_part + 1j * imag_part
    mean_value = np.mean(complex_samples)

    file_path = "Gauss_h.txt"
    with open(file_path, 'w') as f:
        f.write("Complex Samples:\n")
        f.write(np.array2string(complex_samples, separator=', ') + "\n")
        f.write("Mean Value: {}\n".format(mean_value))

    return mean_value


def read_complex_from_file():
    file_path = "Gauss_h.txt"
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Mean Value:"):
                complex_str = line.split("Mean Value:")[1].strip()
                # Use regular expression to extract the complex number
                complex_match = re.match(r'\((.*?)([-+]\d+\.\d+j)\)', complex_str)
                if complex_match:
                    real_part = float(complex_match.group(1))
                    imag_part = float(complex_match.group(2).replace("j", ""))
                    complex_value = np.complex128(real_part + imag_part * 1j)
                    return complex_value
                else:
                    raise ValueError("Invalid complex number format in the file.")


class UAVEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # 定义环境参数
        self.state_num = 7
        self.action_num = 1
        self.action_space = gym.spaces.Box(low=0, high=2*np.pi, shape=(self.action_num,))  # 定义动作空间为连续角度值
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_num,))  # 定义状态空间的维度

        self.height = 100
        self.width = 100
        self.start_point = (10, 20)  # 起点坐标
        self.end_point = (90, 80)  # 终点坐标
        self.ST_position = (0, 50)  # ST 位置
        self.SD_position = (70, 100)  # SD 位置
        self.LM_position = (55, 0)  # LM 位置
        self.time_interval = 1  # 时间间隔

        self.distance_to_end = 200
        self.last_d = 200
        self.current_step = 0  # 当前时间步
        self.time = 200
        self.current_position = self.start_point  # 当前无人机位置

        self.gauss_h = _get_gauss()

        '''self.d_SL = self._get_d_sl()  # d_SL距离
        self.d_SD = self._get_d_sd()  # d_SD距离
        self.d_SR = self._get_d_sr()  # d_SR距离
        self.d_RD = self._get_d_rd()  # d_RD距离
        self.d_RL = self._get_d_rl()  # d_RL距离

        self.Phi_SR = self._get_phi_sr()  # ΦSR
        self.Phi_RD = self._get_phi_rd()  # ΦRD
        self.Phi_RL = self._get_phi_rl()  # ΦRL

        self.h_SL = self._get_h_sl()  # h_SL
        self.h_SD = self._get_h_sd()  # h_SD
        self.h_SR = self._get_h_sr()  # h_SR
        self.h_RD = self._get_h_rd()  # h_RD
        self.h_RL = self._get_h_rl()  # h_RL

        self.Gamma_SD = self._get_gamma_sd()  # γ_SD
        self.Gamma_SL = self._get_gamma_sl()  # γ_SL'''

        self.R_SD = self._get_r_sd()  # R_SD
        self.R_SL = self._get_r_sl()  # R_SL

        self.R_EAV = self._get_r_eav()  # R_EAV
        self.last_R_EAV = self.R_EAV

        self.Viewer = None  # render

    def reset(self):
        self.current_step = 0
        self.current_position = self.start_point
        self.last_d = 200

        self.R_SD = self._get_r_sd()  # R_SD
        self.R_SL = self._get_r_sl()  # R_SL

        self.R_EAV = self._get_r_eav()  # R_EAV
        self.last_R_EAV = self.R_EAV

        return self._get_observation()

    def _get_observation(self):
        # 获取当前状态向量
        state = np.zeros(self.state_num)

        state[0] = self.current_position[0] / self.width  # 当前位置 x
        state[1] = self.current_position[1] / self.height  # 当前位置 y
        state[2] = self.current_step / self.time  # 当前时间步
        state[3] = self.distance_to_end / 200
        state[4] = self.R_SD / 10
        state[5] = self.R_SL / 10
        state[6] = self.R_EAV / 10
        """state[7] = self.end_point[0] / self.width  # end 位置 x
        state[8] = self.end_point[1] / self.height  # end 位置 y
        state[9] = self.ST_position[0] / self.width  # ST 位置 x
        state[10] = self.ST_position[1] / self.height  # ST 位置 y
        state[11] = self.SD_position[0] / self.width  # SD 位置 x
        state[12] = self.SD_position[1] / self.height  # SD 位置 y
        state[13] = self.LM_position[0] / self.width  # LM 位置 x
        state[14] = self.LM_position[1] / self.height  # LM 位置 y"""

        return state

    def step(self, action):
        self.current_step += self.time_interval
        self._take_action(action)

        next_state = self._get_observation()

        done = self._is_done()

        reward = self._get_reward()

        info = {}

        # print(self.current_position)
        self.last_d = self.distance_to_end
        self.last_R_EAV = self.R_EAV

        return next_state, reward, done, info

    def _take_action(self, action):
        # 解析动作值
        angle = (np.clip(action[0], 0, np.pi * 2))  # 将动作值映射到方向角范围 [0, 2π]

        # 根据方向角和行驶距离更新无人机位置
        delta_x = np.cos(angle) * MAX_DISTANCE
        delta_y = np.sin(angle) * MAX_DISTANCE
        self.current_position = (self.current_position[0] + delta_x, self.current_position[1] + delta_y)

        # 更新UAV到终点的距离
        self.distance_to_end = np.sqrt(
            (self.current_position[0] - self.end_point[0]) ** 2 + (self.current_position[1] - self.end_point[1]) ** 2)

        # 更新可达率与窃听率
        self.R_SL = self._get_r_sl()
        self.R_SD = self._get_r_sd()

        self.R_EAV = self._get_r_eav()

    def _get_reward(self):
        # 根据任务目标设计奖励函数
        # 最小化距离
        if self.distance_to_end < self.last_d:
            dis_reward = -np.exp(-self.current_step * 0.01)
        else:
            dis_reward = -np.exp(self.current_step * 0.01)
        dis_reward += -(self.distance_to_end * 0.02)

        if self.R_SL >= self.R_SD:
            Reav_reward = self.R_SD*0.28
        else:
            Reav_reward = -self.R_SD*0.1

        '''alpha = 0.4  # 权重alpha用于平衡飞向终点目标
        beta = 0.6  # 权重beta用于平衡Reav目标

        reward = alpha * dis_reward + beta * Reav_reward'''
        reward = dis_reward + Reav_reward

        if self.distance_to_end < 2:
            done_reward = 3
            reward += done_reward

        if self.current_step >= self.time:
            time_reward = -10
            reward += time_reward

        return reward

    def _is_done(self):
        if self.distance_to_end < 2:
            print("complete")
            return True
        if self.current_step >= self.time:
            return True
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
        target_u_transform = rendering.Transform(translation=(self.current_position[0] + 250, self.current_position[1] + 250))
        target_uav.add_attr(target_u_transform)
        self.Viewer.add_geom(target_uav)
        return self.Viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.Viewer:
            self.Viewer.close()
            self.Viewer = None
        return

    def read_gauss(self):
        self.gauss_h = read_complex_from_file()

    def _get_d_sl(self):
        d_SL = np.sqrt(
            (self.ST_position[0] - self.LM_position[0]) ** 2 + (
                    self.ST_position[1] - self.LM_position[1]) ** 2)
        return d_SL

    def _get_d_sd(self):
        d_SD = np.sqrt(
            (self.ST_position[0] - self.SD_position[0]) ** 2 + (
                    self.ST_position[1] - self.SD_position[1]) ** 2)
        return d_SD

    def _get_d_sr(self):
        d_SR = np.sqrt(
            (self.current_position[0] - self.ST_position[0]) ** 2 + (
                    self.current_position[1] - self.ST_position[1]) ** 2 + UAV_H ** 2)
        return d_SR

    def _get_d_rd(self):
        d_RD = np.sqrt(
            (self.current_position[0] - self.SD_position[0]) ** 2 + (
                    self.current_position[1] - self.SD_position[1]) ** 2 + UAV_H ** 2)
        return d_RD

    def _get_d_rl(self):
        d_RL = np.sqrt(
            (self.current_position[0] - self.LM_position[0]) ** 2 + (
                    self.current_position[1] - self.LM_position[1]) ** 2 + UAV_H ** 2)
        return d_RL

    def _get_h_sl(self):
        h_SL = (np.sqrt(RHO * (self._get_d_sl()**(-K)))) * self.gauss_h
        return h_SL

    def _get_h_sd(self):
        h_SD = (np.sqrt(RHO * (self._get_d_sd()**(-K)))) * self.gauss_h
        return h_SD

    def _get_h_sr(self):
        h_SR = BETA_0 * (self._get_d_sr() ** (-2.5))
        return h_SR

    def _get_h_rd(self):
        h_RD = BETA_0 * (self._get_d_rd() ** (-2.5))
        return h_RD

    def _get_h_rl(self):
        h_RL = BETA_0 * (self._get_d_rl() ** (-2.5))
        return h_RL

    def _get_gamma_srd(self):
        Gamma_SR = (P_S * self._get_h_sr()) / SIGMA_2
        Gamma_RD = (P_U * self._get_h_rd()) / SIGMA_2
        Gamma_SRD = min(Gamma_SR, Gamma_RD)
        return Gamma_SRD

    def _get_gamma_srl(self):
        Gamma_SR = (P_S * self._get_h_sr()) / SIGMA_2
        Gamma_RL = (P_U * self._get_h_rl()) / SIGMA_2
        Gamma_SRL = min(Gamma_SR, Gamma_RL)
        return Gamma_SRL

    def _get_gamma_sd(self):
        Gamma_SD = (P * (np.abs(self._get_h_sd()))) / SIGMA_2
        return Gamma_SD, self._get_gamma_srd()

    def _get_gamma_sl(self):
        Gamma_SL = (P * (np.abs(self._get_h_sl()))) / SIGMA_2
        return Gamma_SL, self._get_gamma_srl()

    def _get_r_sd(self):
        Gamma_SD2, Gamma_SD3 = self._get_gamma_sd()
        R_SD = np.log2(1 + Gamma_SD2) + (1/2)*(np.log2(1 + Gamma_SD3))
        return R_SD

    def _get_r_sl(self):
        Gamma_SL2, Gamma_SL3 = self._get_gamma_sl()
        R_SL = np.log2(1 + Gamma_SL2) + (1/2)*(np.log2(1 + Gamma_SL3))
        return R_SL

    def _get_r_eav(self):
        if self.R_SL >= self.R_SD:
            R_EAV = self.R_SD
        else:
            R_EAV = 0

        return R_EAV
