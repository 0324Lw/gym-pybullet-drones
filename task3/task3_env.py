import gymnasium as gym
import numpy as np
import collections
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
import pybullet as p

from task3_world import Task3World, WorldConfig

# ==========================================
# 模块 1：Config 
# ==========================================
class Task3Config:
    """集中管理所有的超参数和环境配置"""
    
    # --- 频率与时间配置 ---
    CTRL_FREQ = 240                         
    RL_FREQ = 48                            
    SUBSTEPS = CTRL_FREQ // RL_FREQ         
    RL_DT = 1.0 / RL_FREQ                   
    MAX_STEPS = 1200                        
    
    # --- 状态空间特征配置 ---
    STACK_SIZE = 3                          
    OBS_DIM_PER_FRAME = 34                  
    
    # --- Sim2Real 动力学与控制参数---
    EMA_ALPHA = 0.5                         
    ACTION_SCALE = 0.50                     # 0.50 动力边界
    
    # --- 奖励函数超参数---
    R_STEP = -0.03                         # 轻微时间惩罚，逼迫移动
    
    # 定高与姿态约束
    R_HEIGHT_SCALE = 0.25                    
    R_HEIGHT_K = 5.0                        
    R_ATT_PENALTY = -0.1                    
    R_SMOOTH_PENALTY = -0.002                
    
    # 逼近奖励
    R_APPROACH = 0.5                       
    R_DIR = 0.10                            
    R_REPULSION_MAX = -0.4                  
    
    # --- 离散终端事件与极大极小奖励 ---
    R_CRASH = -50.0                         # 侧翻/坠地/撞毁 极刑
    R_DEVIATE = -50.0                       # 严重偏离高度 极刑
    R_SUCCESS_BASE = 50.0                   
    TIME_BONUS_COEF = 0.1                   
    
    # --- 安全判定阈值---
    SAFE_LIDAR_DIST = 0.25                  
    MAX_Z_ERR = 0.8                         # 高度偏离 0.8m 直接判死
    MAX_RP_ANGLE = 1.0                      # 倾角大于约50度直接判死
    
    SUCCESS_XY_TOLERANCE = 0.4              # 终点水平容忍度
    SUCCESS_Z_TOLERANCE = 0.4               # 终点高度容忍度

# ==========================================
# 模块 2：Env (Gymnasium 接口环境类)
# ==========================================
class Task3Env(gym.Wrapper):
    def __init__(self, env: HoverAviary):
        super().__init__(env)
        self.cfg = Task3Config()
        self.world = Task3World(self.unwrapped.CLIENT)
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.obs_dim = self.cfg.OBS_DIM_PER_FRAME * self.cfg.STACK_SIZE
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.frame_buffer = collections.deque(maxlen=self.cfg.STACK_SIZE)
        self.current_ema_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        self.step_counter = 0
        self.goal_pos = np.zeros(3)

    def set_curriculum(self, num_static: int, num_dynamic: int, max_sg_dist: float):
        self.world.cfg.NUM_STATIC_OBS = num_static
        self.world.cfg.NUM_DYNAMIC_OBS = num_dynamic
        self.world.cfg.MAX_SG_DIST = max_sg_dist
        self.world.cfg.MIN_SG_DIST = max_sg_dist * 0.7 

    def _get_drone_state(self):
        pos_tuple, quat_tuple = p.getBasePositionAndOrientation(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        vel_tuple, angular_tuple = p.getBaseVelocity(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        
        pos = np.array(pos_tuple)
        vel = np.array(vel_tuple)
        rpy = np.array(p.getEulerFromQuaternion(quat_tuple))
        
        rot_mat = np.array(p.getMatrixFromQuaternion(quat_tuple)).reshape(3, 3)
        heading_vec = rot_mat[:, 0] 
        
        return pos, vel, rpy, heading_vec

    def _get_34d_frame(self, pos, rpy) -> np.ndarray:
        rel_goal_vec = self.goal_pos - pos
        lidar_scan = self.world.get_lidar_scan(pos, rpy[2], self.unwrapped.DRONE_IDS[0])
        frame = np.concatenate([rpy, self.current_ema_action, rel_goal_vec, lidar_scan]).astype(np.float32)
        return frame

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        obs, info = self.env.reset(**kwargs)
        
        start_pos, self.goal_pos = self.world.reset_world()
        
        # 【基线强制】：锁死起终点高度为 1.0m
        start_pos[2] = 1.0
        self.goal_pos[2] = 1.0
        
        self.step_counter = 0
        
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
        
        self.current_ema_action = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        initial_frame = self._get_34d_frame(start_pos, np.zeros(3))
        self.frame_buffer.clear()
        for _ in range(self.cfg.STACK_SIZE):
            self.frame_buffer.append(initial_frame)
            
        return np.concatenate(list(self.frame_buffer)), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action_nn = np.clip(action, -1.0, 1.0)
        target_action = action_nn * self.cfg.ACTION_SCALE
        
        info = {}
        for _ in range(self.cfg.SUBSTEPS):
            self.current_ema_action = (self.cfg.EMA_ALPHA * target_action) + \
                                      ((1.0 - self.cfg.EMA_ALPHA) * self.current_ema_action)
            step_returns = self.env.step(np.expand_dims(self.current_ema_action, axis=0))
            info = step_returns[-1]
            self.world.step_dynamics()
                
        self.step_counter += 1
        
        pos, vel, rpy, heading_vec = self._get_drone_state()
        current_frame = self._get_34d_frame(pos, rpy)
        self.frame_buffer.append(current_frame)
        
        lidar_scan = current_frame[-WorldConfig.LIDAR_NUM_RAYS:]
        min_lidar_dist = float(np.min(lidar_scan) * WorldConfig.LIDAR_MAX_RANGE)
        
        # 导航距离与向量
        dist_xy = math.hypot(self.goal_pos[0] - pos[0], self.goal_pos[1] - pos[1])
        goal_dir_vec = self.goal_pos - pos
        goal_dir_norm = np.linalg.norm(goal_dir_vec)
        unit_goal_dir = goal_dir_vec / goal_dir_norm if goal_dir_norm > 1e-3 else np.array([1.0, 0.0, 0.0])
        
        dz = pos[2] - 1.0
        
        # 1. 基础惩罚与定高模块
        r_step = self.cfg.R_STEP 
        r_height = self.cfg.R_HEIGHT_SCALE * math.exp(-self.cfg.R_HEIGHT_K * (dz ** 2))
        r_att = self.cfg.R_ATT_PENALTY * (abs(rpy[0]) + abs(rpy[1]))
        r_smooth = self.cfg.R_SMOOTH_PENALTY * float(np.sum(np.square(action_nn - self.prev_action)))
        
        # 2. 距离逼近
        v_toward = float(np.dot(vel, unit_goal_dir))
        v_toward_safe = float(np.clip(v_toward, -5.0, 5.0))
        r_approach = self.cfg.R_APPROACH * v_toward_safe
            
        # 3. 动态方向引导
        v_xy_norm = math.hypot(vel[0], vel[1])
        cos_theta_xy = float(np.dot(heading_vec[:2], unit_goal_dir[:2]) / (np.linalg.norm(heading_vec[:2]) * np.linalg.norm(unit_goal_dir[:2]) + 1e-6))
        r_dir = self.cfg.R_DIR * v_xy_norm * cos_theta_xy
        
        # 4. 障碍物斥力奖励
        r_repulsion = 0.0
        if min_lidar_dist < 1.0:
            r_repulsion = self.cfg.R_REPULSION_MAX * math.exp(-1.5 * (min_lidar_dist - self.cfg.SAFE_LIDAR_DIST))
        
        raw_cont_reward = r_step + r_height + r_att + r_smooth + r_approach + r_dir + r_repulsion
        clipped_cont_reward = float(np.clip(raw_cont_reward, -1.0, 1.0))

        terminal_reward = 0.0
        terminated = False
        truncated = False
        reason = "ALIVE"
        
        if min_lidar_dist < self.cfg.SAFE_LIDAR_DIST:
            terminal_reward = self.cfg.R_CRASH
            terminated = True
            reason = "CRASH_LIDAR"
            
        # 触地判断：低于极刑
        elif pos[2] < 0.1:
            terminal_reward = self.cfg.R_CRASH
            terminated = True
            reason = "CRASH_FLOOR"
            
        # 严格姿态判断：大于直接极刑
        elif max(abs(rpy[0]), abs(rpy[1])) > self.cfg.MAX_RP_ANGLE:
            terminal_reward = self.cfg.R_CRASH
            terminated = True
            reason = "CRASH_FLIP"
            
        # 高度出界判断：偏离超过极刑
        elif abs(dz) > self.cfg.MAX_Z_ERR:
            terminal_reward = self.cfg.R_DEVIATE
            terminated = True
            reason = "CRASH_Z_DEVIATE"
            
        elif dist_xy < self.cfg.SUCCESS_XY_TOLERANCE and abs(dz) < self.cfg.SUCCESS_Z_TOLERANCE:
            time_bonus = max(0, self.cfg.MAX_STEPS - self.step_counter) * self.cfg.TIME_BONUS_COEF
            terminal_reward = self.cfg.R_SUCCESS_BASE + time_bonus
            truncated = True
            reason = "SUCCESS"
            
        elif self.step_counter >= self.cfg.MAX_STEPS:
            truncated = True
            reason = "TIMEOUT"
            
        final_reward = clipped_cont_reward + terminal_reward
        
        info['task3_stats'] = {
            'r_step': r_step, 'r_height': r_height, 'r_att': r_att, 'r_smooth': r_smooth,
            'r_approach': r_approach, 'r_dir': r_dir, 'r_repulsion': r_repulsion,
            'r_cont_clipped': clipped_cont_reward, 'r_terminal': terminal_reward, 'r_final_total': final_reward,
            'dist_xy': dist_xy, 'pos_z': float(pos[2]),
            'min_lidar': min_lidar_dist, 'reason': reason
        }
        
        self.prev_action = action_nn.copy()
        return np.concatenate(list(self.frame_buffer)), float(final_reward), terminated, truncated, info

# ==========================================
# 模块 3：Plot (数据可视化类)
# ==========================================
class Task3Plot:
    @staticmethod
    def plot_flight_data(info_list: list, save_path: str = "task3_flight_analysis.png"):
        if not info_list: return
        steps = range(len(info_list))
        dist_xy = [info['dist_xy'] for info in info_list]
        pos_z = [info['pos_z'] for info in info_list]
        r_total = [info['r_final_total'] for info in info_list]
        min_lidar = [info['min_lidar'] for info in info_list]
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axs[0].plot(steps, dist_xy, 'b-', label='XY Distance to Goal')
        axs[0].set_ylabel("Distance (m)")
        axs[0].set_title("Navigation Tracking")
        
        ax0_twin = axs[0].twinx()
        ax0_twin.axhline(y=1.0, color='g', linestyle='--', label='Target Z (1.0m)')
        ax0_twin.plot(steps, pos_z, 'g-', label='Actual Z')
        ax0_twin.set_ylabel("Height Z (m)", color='g')
        ax0_twin.tick_params(axis='y', labelcolor='g')
        
        lines_1, labels_1 = axs[0].get_legend_handles_labels()
        lines_2, labels_2 = ax0_twin.get_legend_handles_labels()
        axs[0].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        
        axs[1].plot(steps, min_lidar, 'purple', label='Min LiDAR Distance')
        axs[1].axhline(y=Task3Config.SAFE_LIDAR_DIST, color='r', linestyle=':', label='Crash Threshold (0.25m)')
        axs[1].set_ylabel("Obstacle Proximity (m)")
        axs[1].legend()
        
        axs[2].plot(steps, r_total, 'orange', label='Total Step Reward')
        axs[2].set_xlabel("RL Steps")
        axs[2].set_ylabel("Reward Value")
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()