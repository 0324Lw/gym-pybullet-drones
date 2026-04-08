import gymnasium as gym
import numpy as np
import collections
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# ==========================================
# 模块 1：Config (全局配置类)
# ==========================================
class Task1Config:
    """集中管理所有的超参数和环境配置"""
    
    # --- 环境与任务参数 ---
    TARGET_POS = np.array([0.0, 0.0, 1.0])  # 目标悬停位置 (x, y, z)
    MAX_STEPS = 500                         # 每回合最大步数
    STACK_SIZE = 4                          # 状态帧堆叠数量
    
    # --- Sim2Real 动作控制参数 ---
    ACTION_SCALE = 0.50                     # 网络输出[-1,1]映射到的实际微调比例 (±20%)
    EMA_ALPHA = 0.3                         # 指数加权移动平均系数 (越小越平滑，延迟越大)
    
    # --- 奖励函数超参数 (原始数值) ---
    R_STEP = -0.01                          # 每步固定时间惩罚
    R_HEIGHT_K = 5.0                        # 高度奖励的高斯分布敏感度系数
    R_HEIGHT_SCALE = 0.2                    # 高度奖励最大值
    R_ATT_PENALTY = -0.1                    # 姿态偏离惩罚系数
    R_SMOOTH_PENALTY = -0.02                # 动作突变惩罚系数
    
    # 终端极刑与奖励 (截断前设定的大额分值)
    R_CRASH = -10.0                         # 侧翻/坠毁
    R_DEVIATE = -10.0                       # 严重偏离目标
    R_SUCCESS = 10.0                        # 完美达成悬停100步
    
    # --- 任务终止阈值 ---
    MAX_Z_ERR = 0.5                         # 允许的最大高度误差 (m)
    MAX_RP_ANGLE = 0.4                      # 允许的最大滚转/俯仰角 (rad, 约23度)
    SUCCESS_STEPS_REQ = 300                 # 连续满足误差在0.1m内的步数，视为成功


# ==========================================
# 模块 2：Env (Gymnasium 接口环境类)
# ==========================================
class Task1Env(gym.Wrapper):
    """
    基于 HoverAviary 的强化学习高保真包装器。
    实现 4 帧堆叠 (44维)，EMA 动作平滑，以及 [-1, 1] 的单步奖励截断。
    """
    def __init__(self, env: HoverAviary):
        super().__init__(env)
        self.cfg = Task1Config()
        
        # 动作空间：纯粹的 [-1.0, 1.0]，代表对 4 个电机的调整趋势
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 状态空间：单帧 11 维 * 4 帧 = 44 维
        single_frame_dim = 11 
        self.obs_dim = single_frame_dim * self.cfg.STACK_SIZE
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # 内部状态变量
        self.frame_buffer = collections.deque(maxlen=self.cfg.STACK_SIZE)
        self.current_ema_action = np.zeros(4, dtype=np.float32) # 当前底层真实动作
        self.prev_nn_action = np.zeros(4, dtype=np.float32)     # 上一步网络的原始输出
        
        self.step_counter = 0
        self.stable_counter = 0 # 记录连续稳定悬停的步数

    def _get_11d_frame(self) -> np.ndarray:
        """从底层提取 11 维度的单帧状态"""
        # 从底层直接获取绝对坐标和欧拉角
        pos = self.unwrapped.pos[0]
        rpy = self.unwrapped.rpy[0]
        
        # 高度差感知
        dz = pos[2] - self.cfg.TARGET_POS[2]
        
        # 当前实际电机的伪转速 (EMA平滑后的控制量)
        rpms = self.current_ema_action
        
        # 拼接成 11 维向量: [x, y, z, roll, pitch, yaw, rpm1, rpm2, rpm3, rpm4, dz]
        frame = np.concatenate([pos, rpy, rpms, [dz]]).astype(np.float32)
        return frame

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        
        # 重置内部变量
        self.step_counter = 0
        self.stable_counter = 0
        self.current_ema_action = np.zeros(4, dtype=np.float32)
        self.prev_nn_action = np.zeros(4, dtype=np.float32)
        
        # 获取第一帧并填满帧队列
        initial_frame = self._get_11d_frame()
        for _ in range(self.cfg.STACK_SIZE):
            self.frame_buffer.append(initial_frame)
            
        stacked_obs = np.concatenate(list(self.frame_buffer))
        return stacked_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_counter += 1
        
        # 1. --- Sim2Real 动作平滑 (EMA) 与映射 ---
        # 限制网络动作范围，防止异常极值
        action_nn = np.clip(action, -1.0, 1.0)
        
        # 目标微调比例
        target_action = action_nn * self.cfg.ACTION_SCALE
        
        # EMA 底层滤波
        self.current_ema_action = (self.cfg.EMA_ALPHA * target_action) + \
                                  ((1.0 - self.cfg.EMA_ALPHA) * self.current_ema_action)
        
        # 将平滑后的动作喂给底层物理环境 (底层接受 [-1, 1]，代表相对悬停基准的偏置)
        action_env = np.expand_dims(self.current_ema_action, axis=0)
        _, _, terminated_base, truncated_base, info = self.env.step(action_env)
        
        # 2. --- 提取状态并更新帧队列 ---
        current_frame = self._get_11d_frame()
        self.frame_buffer.append(current_frame)
        stacked_obs = np.concatenate(list(self.frame_buffer))
        
        pos = current_frame[0:3]
        rpy = current_frame[3:6]
        dz = current_frame[10]
        
        # 3. --- 奖励函数结算 ---
        r_step = self.cfg.R_STEP
        r_height = self.cfg.R_HEIGHT_SCALE * np.exp(-self.cfg.R_HEIGHT_K * (dz ** 2))
        r_att = self.cfg.R_ATT_PENALTY * (abs(rpy[0]) + abs(rpy[1])) # 忽略 Yaw
        r_smooth = self.cfg.R_SMOOTH_PENALTY * np.sum(np.square(action_nn - self.prev_nn_action))
        
        raw_reward = r_step + r_height + r_att + r_smooth
        
        # 4. --- 终端条件与极刑判断 (完全重构) ---
        terminated = False
        truncated = False
        
        if self.step_counter >= self.cfg.MAX_STEPS:
            truncated = True
            
        # 坠毁判断：真实触地 或 姿态越界 (彻底抛弃 terminated_base，只看物理坐标)
        if pos[2] < 0.1 or max(abs(rpy[0]), abs(rpy[1])) > self.cfg.MAX_RP_ANGLE:
            raw_reward = self.cfg.R_CRASH
            terminated = True
            
        # 偏离判断：高度差过大 (冲天或者过度掉落)
        elif abs(dz) > self.cfg.MAX_Z_ERR:
            raw_reward = self.cfg.R_DEVIATE
            terminated = True
            
# 成功判断：连续保持在目标高度附近
        elif abs(dz) <= 0.1:
            self.stable_counter += 1
            if self.stable_counter >= self.cfg.SUCCESS_STEPS_REQ:
                # 【核心修复 1：时间补偿机制】
                # 把提前下班省下来的步数，按照满分(R_HEIGHT_SCALE)折算成奖金一次性发给它
                time_bonus = (self.cfg.MAX_STEPS - self.step_counter) * self.cfg.R_HEIGHT_SCALE
                raw_reward = self.cfg.R_SUCCESS + time_bonus
                terminated = True
        else:
            self.stable_counter = 0

        # 【核心修复 2：放宽截断范围】
        # 将原来的 [-1.0, 1.0] 扩大到 [-20.0, 150.0]
        # 保证坠毁的 -10 分和提前完成的百余分补偿能够真实传递给价值网络！
        clipped_reward = float(np.clip(raw_reward, -20.0, 150.0))
        
        # 5. --- 记录调试信息 ---
        info['task1_stats'] = {
            'r_step': float(r_step),
            'r_height': float(r_height),
            'r_att': float(r_att),
            'r_smooth': float(r_smooth),
            'r_raw_total': float(raw_reward),
            'r_clipped': clipped_reward,
            'pos_z': float(pos[2]),
            'stable_counter': int(self.stable_counter)
        }
        
        self.prev_nn_action = action_nn.copy()
        
        return stacked_obs, clipped_reward, terminated, truncated, info


# ==========================================
# 模块 3：Plot (绘图类接口)
# ==========================================
class Task1Plot:
    """提供通用的可视化方法，用于分析环境测试或训练的数据分布"""
    
    @staticmethod
    def plot_episode_stats(info_list: list, save_path: str = "episode_stats.png"):
        """
        传入一个回合中每一步产生的 info['task1_stats'] 列表，绘制状态变化曲线。
        """
        if not info_list:
            print("[Plot] 无数据可绘！")
            return
            
        steps = range(len(info_list))
        z_history = [info['pos_z'] for info in info_list]
        r_clipped = [info['r_clipped'] for info in info_list]
        r_smooth = [info['r_smooth'] for info in info_list]
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # 1. 高度轨迹图
        axs[0].plot(steps, z_history, label="Drone Z-Position", color='blue')
        axs[0].axhline(y=1.0, color='red', linestyle='--', label="Target (1.0m)")
        axs[0].axhspan(0.9, 1.1, color='green', alpha=0.2, label="Success Zone (±0.1m)")
        axs[0].set_title("Drone Altitude over Episode")
        axs[0].set_ylabel("Z (meters)")
        axs[0].legend()
        axs[0].grid(True)
        
        # 2. 奖励分布图
        axs[1].plot(steps, r_clipped, label="Clipped Total Reward", color='purple')
        axs[1].set_title("Step Reward (Clipped to [-1, 1])")
        axs[1].set_ylabel("Reward")
        axs[1].legend()
        axs[1].grid(True)
        
        # 3. 动作平滑度图
        axs[2].plot(steps, r_smooth, label="Smoothness Penalty", color='orange')
        axs[2].set_title("Action Smoothness Penalty (Closer to 0 is better)")
        axs[2].set_xlabel("Steps")
        axs[2].set_ylabel("Penalty")
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[Plot] 回合分析图表已保存至 {save_path}")
        plt.close()
