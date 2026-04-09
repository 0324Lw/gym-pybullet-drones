import gymnasium as gym
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import pybullet as p

# ==========================================
# 模块 1：Config (全局配置类)
# ==========================================
class Task2Config:
    """集中管理所有的超参数和环境配置"""
    
    # --- 频率与时间配置 ---
    CTRL_FREQ = 240                         
    RL_FREQ = 48                            
    SUBSTEPS = CTRL_FREQ // RL_FREQ         
    RL_DT = 1.0 / RL_FREQ                   
    MAX_STEPS = 800                         # 轨迹追踪最大步数限制
    
    # --- 状态空间特征配置 ---
    STACK_SIZE = 4                          
    LOOKAHEAD_STEPS = 5                     # 提取未来 5 个点的相对坐标
    LOOKAHEAD_INTERVAL = 10                 # 前瞻点之间的索引间隔 (代表未来的距离跨度)
    OBS_DIM_PER_FRAME = 25                  # (3欧拉角 + 4力矩比例 + 3当前相对 + 15前瞻相对)
    
    # --- Sim2Real 动力学与控制参数 ---
    EMA_ALPHA = 0.5                         # 动作平滑系数 (兼顾响应与防爆震)
    ACTION_SCALE = 0.5                      # 还原 Task1 接口！最大允许偏置悬停推力的 ±50%
    
    # --- 连续奖励超参数 ---
    R_SURVIVAL = 0.1                        # 生存奖励
    R_TRACK_SIGMA = 1.0                     # 高斯距离奖励的平滑系数
    R_VEL_COEF = 0.15                       # 切向速度奖励系数
    R_HEADING_COEF = 0.05                   # 切向机头朝向奖励系数
    R_SMOOTH_COEF = -0.05                   # 力矩突变惩罚系数
    
    # --- 离散终端事件与极大极小奖励 ---
    R_CRASH = -20.0                         
    R_DEVIATE = -20.0                       
    R_SUCCESS_BASE = 50.0                   # 基础完成奖励
    TIME_BONUS_COEF = 0.1                   # 竞速奖励系数 (提前完成的时间折算补偿)
    
    # --- 任务终止阈值 ---
    MAX_DEV_ERR = 2.0                       # 最大允许偏离距离 (m)
    MAX_RP_ANGLE = 1.05                     # 约 60度 (1.047 rad)，允许高动态追踪压弯

# ==========================================
# 模块 2：Env (Gymnasium 接口环境类)
# ==========================================
class Task2Env(gym.Wrapper):
    def __init__(self, env: HoverAviary):
        super().__init__(env)
        self.cfg = Task2Config()
        
        # 动作空间：[-1.0, 1.0]^4 (网络输出，后续映射为推力比例)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 状态空间：单帧 25 维 * 4 帧 = 100 维
        self.obs_dim = self.cfg.OBS_DIM_PER_FRAME * self.cfg.STACK_SIZE
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.frame_buffer = collections.deque(maxlen=self.cfg.STACK_SIZE)
        self.current_ema_action = np.zeros(4, dtype=np.float32) # 安全归一化接口
        self.prev_action = np.zeros(4, dtype=np.float32)        # 记录网络原始输出
        
        self.step_counter = 0
        self.target_idx = 0  # Path Following：当前追踪的轨迹点索引
        
        self.waypoints = []
        self.tangents = []

    def _generate_static_trajectory(self):
        """回合开始前生成一条完整静态 3D 轨迹 (平滑且 Z > 1.0m)"""
        num_points = self.cfg.MAX_STEPS * 2  # 预留足够长的路径点供“超前竞速”消费
        
        Ax = np.random.uniform(1.5, 3.0)
        Ay = np.random.uniform(1.5, 3.0)
        Az = np.random.uniform(0.1, 0.4)     
        
        fx = np.random.uniform(0.02, 0.08)
        fy = np.random.uniform(0.02, 0.08)
        fz = np.random.uniform(0.01, 0.05)
        
        dx = np.random.uniform(0, 2 * math.pi)
        dy = np.random.uniform(0, 2 * math.pi)
        dz = np.random.uniform(0, 2 * math.pi)
        
        self.waypoints = []
        self.tangents = []
        
        for i in range(num_points):
            t = i * self.cfg.RL_DT
            px = Ax * math.sin(2 * math.pi * fx * t + dx)
            py = Ay * math.sin(2 * math.pi * fy * t + dy)
            pz = 1.2 + Az * math.sin(2 * math.pi * fz * t + dz)  # 保证安全空域 Z>=0.8m
            self.waypoints.append(np.array([px, py, pz]))
            
            # 计算切线方向 (差分逼近，用于计算速度与朝向投影)
            if i > 0:
                dir_vec = self.waypoints[i] - self.waypoints[i-1]
                norm = np.linalg.norm(dir_vec)
                tan = dir_vec / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
                self.tangents.append(tan)
        
        self.tangents.insert(0, self.tangents[0]) 
        
        self.waypoints = np.array(self.waypoints)
        self.tangents = np.array(self.tangents)

    def _get_drone_state(self):
        """直连底层，获取绝对物理状态，防止缓存延迟Bug"""
        pos_tuple, quat_tuple = p.getBasePositionAndOrientation(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        vel_tuple, angular_tuple = p.getBaseVelocity(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        
        pos = np.array(pos_tuple)
        vel = np.array(vel_tuple)
        quat = np.array(quat_tuple)
        rpy = np.array(p.getEulerFromQuaternion(quat_tuple))
        
        # 提取无人机当前机头朝向向量 (机身 X 轴方向)
        rot_mat = np.array(p.getMatrixFromQuaternion(quat_tuple)).reshape(3, 3)
        heading_vec = rot_mat[:, 0] 
        
        return pos, vel, rpy, heading_vec

    def _update_target_idx(self, pos: np.ndarray):
        """动态寻的机制 (Path Following)：往前搜索最近点，不绑定时间，允许超前竞速"""
        search_range = 30  # 只往前搜索局部窗口，防止轨迹交叉时飞错圈
        end_idx = min(self.target_idx + search_range, len(self.waypoints))
        
        distances = np.linalg.norm(self.waypoints[self.target_idx:end_idx] - pos, axis=1)
        closest_local_idx = np.argmin(distances)
        self.target_idx += closest_local_idx

    def _get_100d_frame(self) -> np.ndarray:
        """构建彻底泛化的高维状态空间 (基于相对距离特征)"""
        pos, _, rpy, _ = self._get_drone_state()
        self._update_target_idx(pos)
        
        # 1. 最近轨迹点相对坐标 (3维)
        target_pos = self.waypoints[self.target_idx]
        rel_target_pos = target_pos - pos
        
        # 2. 前瞻点相对坐标 (15维)
        lookahead_coords = []
        for k in range(1, self.cfg.LOOKAHEAD_STEPS + 1):
            f_idx = min(self.target_idx + k * self.cfg.LOOKAHEAD_INTERVAL, len(self.waypoints) - 1)
            f_pos = self.waypoints[f_idx]
            lookahead_coords.append(f_pos - pos)
        lookahead_flat = np.concatenate(lookahead_coords)
        
        # 组合单帧：欧拉角(3) + 当前底层力矩比(4) + 目标相对(3) + 前瞻相对(15) = 25维
        frame = np.concatenate([rpy, self.current_ema_action, rel_target_pos, lookahead_flat]).astype(np.float32)
        return frame

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        
        self._generate_static_trajectory()
        self.step_counter = 0
        self.target_idx = 0
        
        start_pos = self.waypoints[0]
        
        # 强制重置到底层：位置设定为起点，速度/角速度全部清零
        p.resetBasePositionAndOrientation(
            self.unwrapped.DRONE_IDS[0],
            start_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.unwrapped.CLIENT
        )
        p.resetBaseVelocity(
            self.unwrapped.DRONE_IDS[0],
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.unwrapped.CLIENT
        )
        
        # 恢复悬停的干净状态
        self.current_ema_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        initial_frame = self._get_100d_frame()
        self.frame_buffer.clear()
        for _ in range(self.cfg.STACK_SIZE):
            self.frame_buffer.append(initial_frame)
            
        return np.concatenate(list(self.frame_buffer)), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_nn = np.clip(action, -1.0, 1.0)
        # 目标推力微调比例 (相对于悬停基准)
        
        target_action = action_nn * self.cfg.ACTION_SCALE
        
        info = {} 
        
        # 高频物理平滑下发
        for _ in range(self.cfg.SUBSTEPS):
            self.current_ema_action = (self.cfg.EMA_ALPHA * target_action) + \
                                      ((1.0 - self.cfg.EMA_ALPHA) * self.current_ema_action)
            action_env = np.expand_dims(self.current_ema_action, axis=0)
            _, _, _, _, info = self.env.step(action_env) 
                
        self.step_counter += 1
        
        # --- 状态与物理量获取 ---
        stacked_obs = np.concatenate(list(self.frame_buffer)) 
        pos, vel, rpy, heading_vec = self._get_drone_state()
        
        self._update_target_idx(pos)
        current_frame = self._get_100d_frame()
        self.frame_buffer.append(current_frame)
        
        # --- 奖励函数计算 ---
        target_pos = self.waypoints[self.target_idx]
        tangent_vec = self.tangents[self.target_idx]
        
        # 计算真实物理距离误差 
        dx = pos[0] - target_pos[0]
        dy = pos[1] - target_pos[1]
        dz = pos[2] - target_pos[2]
        dist_err = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 对 Z 轴误差施加 2.5 倍的放大权重！逼迫网络多给油门，克服重力掉高
        weighted_dist_err = math.sqrt(dx**2 + dy**2 + (2.5 * dz)**2)
        dist_err = float(np.linalg.norm(pos - target_pos))
        
        # 1. 生存与追踪基础分 (高斯分布)
        r_survival = self.cfg.R_SURVIVAL
        r_track = 0.5 * math.exp(-(weighted_dist_err**2) / (2 * self.cfg.R_TRACK_SIGMA**2)) - 0.1
        
        # 2. 向量投影分：速度沿着轨迹 / 机头沿着轨迹
        raw_v_align = float(np.dot(vel, tangent_vec))
        safe_v_align = float(np.clip(raw_v_align, -5.0, 5.0)) 
        
        r_vel = self.cfg.R_VEL_COEF * safe_v_align
        r_heading = self.cfg.R_HEADING_COEF * float(np.dot(heading_vec, tangent_vec))
        
        # 3. 控制平顺惩罚 
        r_smooth = self.cfg.R_SMOOTH_COEF * float(np.sum(np.square(action_nn - self.prev_action)))
        
        # 聚合与安全截断
        raw_continuous_reward = r_survival + r_track + r_vel + r_heading + r_smooth
        clipped_continuous_reward = float(np.clip(raw_continuous_reward, -1.0, 1.0))
        
        # --- 终端事件判定 ---
        terminal_reward = 0.0
        terminated = False
        truncated = False
        
        if dist_err > self.cfg.MAX_DEV_ERR:
            terminal_reward = self.cfg.R_DEVIATE
            terminated = True
        elif max(abs(rpy[0]), abs(rpy[1])) > self.cfg.MAX_RP_ANGLE or pos[2] < 0.05:
            terminal_reward = self.cfg.R_CRASH
            terminated = True
            
        # 到达终点 (提前跑完给予竞速奖励)
        elif self.target_idx >= len(self.waypoints) - self.cfg.LOOKAHEAD_STEPS - 1:
            time_bonus = max(0, self.cfg.MAX_STEPS - self.step_counter) * self.cfg.TIME_BONUS_COEF
            terminal_reward = self.cfg.R_SUCCESS_BASE + time_bonus
            truncated = True
            
        elif self.step_counter >= self.cfg.MAX_STEPS:
            truncated = True
            
        final_reward = clipped_continuous_reward + terminal_reward
        
        info['task2_stats'] = {
            'r_surv': r_survival,
            'r_track': r_track,
            'r_vel': r_vel,
            'r_heading': r_heading,
            'r_smooth': r_smooth,
            'r_cont_clipped': clipped_continuous_reward,
            'r_terminal': terminal_reward,
            'r_final_total': final_reward,
            'dist_err': dist_err,
            'completion_rate': (self.target_idx / len(self.waypoints)) * 100.0,
            'pos': pos.copy(),
            'target_pos': target_pos.copy()
        }
        
        self.prev_action = action_nn.copy()
        
        return np.concatenate(list(self.frame_buffer)), float(final_reward), terminated, truncated, info

# ==========================================
# 模块 3：Plot (绘图类接口)
# ==========================================
class Task2Plot:
    @staticmethod
    def plot_tracking_performance(info_list: list, save_path: str = "trajectory_tracking.png"):
        if not info_list:
            return
            
        steps = range(len(info_list))
        actual_pos = np.array([info['pos'] for info in info_list])
        target_pos = np.array([info['target_pos'] for info in info_list])
        errors = [info['dist_err'] for info in info_list]
        r_total = [info['r_final_total'] for info in info_list]
        
        fig = plt.figure(figsize=(15, 6))
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'r--', label='Target Path', alpha=0.7)
        ax1.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 'b-', label='Drone Flight Path', linewidth=2)
        ax1.set_title("3D Path Following Performance")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.legend()
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(steps, errors, 'g-', label='Tracking Error (m)')
        ax2.plot(steps, r_total, 'purple', label='Total Reward', alpha=0.5)
        ax2.axhline(y=Task2Config.MAX_DEV_ERR, color='red', linestyle=':', label='Death Threshold')
        ax2.set_title("Tracking Error & Step Reward over Time")
        ax2.set_xlabel("RL Steps")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[Plot] 追踪性能图表已保存至 {save_path}")
        plt.close()