import gymnasium as gym
import numpy as np
import collections
import math
import pybullet as p
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from scipy.interpolate import splprep, splev
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# 导入我们昨天编写的世界模型
from task4_world import Task4World, WorldConfig

# ==========================================
# 模块 1：Config (全局超参数集中配置)
# ==========================================
class Task4Config:
    """集中管理 Sim2Real 建模、奖励函数与强化学习的所有可调参数"""
    
    # --- 基础与频率配置 ---
    CTRL_FREQ = 240                         # 物理引擎步进频率 (Hz)
    RL_FREQ = 48                            # RL 决策频率 (Hz)
    SUBSTEPS = CTRL_FREQ // RL_FREQ         # 每次 RL action 对应的物理步数 (5步)
    MAX_STEPS = 500                         # 回合最大步数 (约 10.4 秒)
    
    # --- 状态空间与堆叠配置 ---
    STACK_SIZE = 3                          # 时序堆叠帧数 (解决部分可观测性)
    DEPTH_RES = (64, 64)                    # 深度图分辨率
    PROPRIO_DIM = 17                         # 本体感知维度
    PRIVILEGED_DIM = 27                     # Critic特权维度
    
    # --- Sim2Real: 执行器建模 (Actuator Dynamics) ---
    RPM_DEADZONE = 0.05                     # 动作死区: [-0.05, 0.05] 内的噪声动作被忽略
    MOTOR_ALPHA = 0.6                       # 一阶低通滤波系数: 模拟电机响应延迟
    ACTION_SCALE = 0.5                      # 动作映射: 输出 [-1, 1] 对应悬停转速的 ±50% 增量
    IDLE_RPM_RATIO = 0.2                    # 最小怠速保护: 防止空中停转翻车
    
    # --- Sim2Real: 域随机化 (Domain Randomization) ---
    DR_RANGE = 0.10                         # 物理参数随机化幅度: ±10%
    NOISE_IMU_STD = 0.02                    # 本体感知(欧拉角)高斯噪声标准差
    NOISE_DEPTH_PROB = 0.01                 # 深度图椒盐噪声概率
    NOISE_DEPTH_STD = 0.03                  # 深度图高斯模糊标准差
    
    # --- 奖励函数超参数 (解耦重构版) ---
    R_STEP = -0.25                          # [步数惩罚] 基础流血
    R_TRACK_V_SCALE = 0.20                  # [跟踪奖励] 速度在切线投影上的乘数
    R_TRACK_K = 2.0                         # [跟踪奖励] 高斯距离衰减的敏感度
    R_ALIGN_POSE = 0.15                     # [对齐奖励] 机头方向与门法向的点积乘数
    R_SMOOTH = -0.01                        # [平滑惩罚] 动作惩罚
    
    # --- 离散极刑与大奖 ---
    R_GATE_BASE = 15.0                      
    R_CRASH = -100.0                        
    R_TIMEOUT_PENALTY = -100.0              # 超时等同于坠毁
    R_SUCCESS = 200.0                       # [通关大奖] 穿过最后一道门
    
    # --- 安全与结算阈值 ---
    CRASH_Z_MIN = 0.2                       # 触地阈值
    CRASH_RP_MAX = 1.2                      # 极限侧翻阈值 (约 68 度，放宽以允许极速切弯)
    PASS_GATE_DIST = 0.8                    # 判定穿门时，距离门心的最大容忍距离 (米)


# ==========================================
# 模块 2：Env (非对称强化学习环境)
# ==========================================
class Task4Env(gym.Wrapper):
    def __init__(self, env: HoverAviary):
        super().__init__(env)
        self.cfg = Task4Config()
        self.world = Task4World(self.unwrapped.CLIENT)
        
        # 1. 动作空间: [-1, 1] 的 4 个转速增量
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 2. 非对称 Dict 状态空间设计
        self.observation_space = gym.spaces.Dict({
            # Actor & Critic 共享: 3帧堆叠的 64x64 深度图 (Channel First)
            "depth_img": gym.spaces.Box(low=0.0, high=1.0, shape=(self.cfg.STACK_SIZE, *self.cfg.DEPTH_RES), dtype=np.float32),
            # Actor & Critic 共享: 3帧堆叠的本体感知
            "proprioception": gym.spaces.Box(low=-10.0, high=10.0, shape=(self.cfg.STACK_SIZE * self.cfg.PROPRIO_DIM,), dtype=np.float32),
            # 仅供 Critic 开启上帝视角: 绝对坐标、前瞻路点等
            "critic_privileged": gym.spaces.Box(low=-10.0, high=10.0, shape=(self.cfg.PRIVILEGED_DIM,), dtype=np.float32)
        })
        
        # 3. 状态与物理缓冲队列
        self.depth_buffer = collections.deque(maxlen=self.cfg.STACK_SIZE)
        self.proprio_buffer = collections.deque(maxlen=self.cfg.STACK_SIZE)
        
        self.current_real_rpm = np.zeros(4, dtype=np.float32)
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        # 4. 任务状态机
        self.step_counter = 0
        self.current_target_gate_idx = 0
        self.passed_gate_flags = []
        self.dense_spline_points = None
        self.base_mass = self.unwrapped.M   # 记录原生质量，用于域随机化

    # ---------------------------------------------------------
    # 核心 1：域随机化与重置
    # ---------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs, info = self.env.reset(**kwargs)
        self.step_counter = 0
        self.current_target_gate_idx = 0
        self.passed_gate_flags = [False] * self.world.cfg.NUM_GATES
        
        # 1. 重置世界并获取门阵列
        start_pos, self.gate_poses = self.world.reset_world()
        self._generate_dense_spline()
        
        # 2. 域随机化 (Domain Randomization)
        mass_scale = np.random.uniform(1.0 - self.cfg.DR_RANGE, 1.0 + self.cfg.DR_RANGE)
        new_mass = self.base_mass * mass_scale
        p.changeDynamics(self.unwrapped.DRONE_IDS[0], -1, mass=new_mass, physicsClientId=self.unwrapped.CLIENT)
        
        # 3. 初始化物理状态
        p.resetBasePositionAndOrientation(
            self.unwrapped.DRONE_IDS[0],
            start_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.unwrapped.CLIENT
        )
        p.resetBaseVelocity(self.unwrapped.DRONE_IDS[0], [0,0,0], [0,0,0], physicsClientId=self.unwrapped.CLIENT)
        
        self.current_real_rpm = np.ones(4) * self.unwrapped.HOVER_RPM
        self.prev_action = np.zeros(4, dtype=np.float32)
        
        # 4. 初始化缓冲队列
        initial_depth, initial_proprio, initial_priv = self._get_dict_obs()
        self.depth_buffer.clear()
        self.proprio_buffer.clear()
        for _ in range(self.cfg.STACK_SIZE):
            self.depth_buffer.append(initial_depth[0]) # 取消第一维以便入队
            self.proprio_buffer.append(initial_proprio)
            
        return self._build_obs_dict(initial_priv), info

    # ---------------------------------------------------------
    # 核心 2：执行器建模与主步进
    # ---------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        # 1. 异常值过滤与裁剪
        action_nn = np.clip(action, -1.0, 1.0)
        
        # 2. 死区模拟 (Deadzone)
        active_action = np.where(np.abs(action_nn) < self.cfg.RPM_DEADZONE, 0.0, action_nn)
        
        # 3. 映射至目标 RPM
        target_rpm = self.unwrapped.HOVER_RPM * (1.0 + active_action * self.cfg.ACTION_SCALE)
        
        info = {}
        # 4. 物理子步循环 (应用一阶延迟滤波)
        for _ in range(self.cfg.SUBSTEPS):
            # 一阶低通滤波
            self.current_real_rpm = (self.cfg.MOTOR_ALPHA * target_rpm) + \
                                    ((1.0 - self.cfg.MOTOR_ALPHA) * self.current_real_rpm)
            
            # 物理饱和防停机裁剪
            min_rpm = self.unwrapped.HOVER_RPM * self.cfg.IDLE_RPM_RATIO
            max_rpm = self.unwrapped.HOVER_RPM * 2.0
            self.current_real_rpm = np.clip(self.current_real_rpm, min_rpm, max_rpm)
            
            # 覆写底层力矩计算 (黑客式接管)
            # 真实环境中，升力正比于 RPM 的平方
            forces = np.square(self.current_real_rpm) * self.unwrapped.KF
            torques = np.square(self.current_real_rpm) * self.unwrapped.KM
            
            # 通过 applyExternalForce 等底层 API 施加力 (或利用底层 step 传入 RPM)
            # 此处我们直接通过原有的 action 接口将真实 RPM 反向归一化后传给底层引擎
            sim_action = (self.current_real_rpm / self.unwrapped.MAX_RPM) * 2.0 - 1.0
            step_returns = self.env.step(np.expand_dims(sim_action, axis=0))
            info = step_returns[-1]
            
        self.step_counter += 1
        
        # 5. 获取观测与状态
        depth_img, proprio, priv = self._get_dict_obs()
        self.depth_buffer.append(depth_img[0])
        self.proprio_buffer.append(proprio)
        
        # 6. 计算奖励与终端判定 (详见独立方法)
        reward, terminated, truncated, step_info = self._compute_reward_and_done(action_nn)
        info['task4_stats'] = step_info
        
        self.prev_action = action_nn.copy()
        
        return self._build_obs_dict(priv), float(reward), terminated, truncated, info

    # ---------------------------------------------------------
    # 核心 3：状态解构与噪声注入
    # ---------------------------------------------------------
    def _get_dict_obs(self):
        pos, quat = p.getBasePositionAndOrientation(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        vel, ang_vel = p.getBaseVelocity(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        rpy = np.array(p.getEulerFromQuaternion(quat))
        
        #  获取当前目标门的绝对坐标
        gate_pos = self.gate_poses[min(self.current_target_gate_idx, self.world.cfg.NUM_GATES-1)]['pos']
        
        # 计算全局误差向量 (从无人机指向门)
        global_vec_to_gate = gate_pos - np.array(pos)
        
        # === 核心罗盘：将全局向量转换到机身局部坐标系 (Local Frame) ===
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # 旋转矩阵的转置即为逆矩阵，用于将世界坐标系向量转入机体坐标系
        local_vec_to_gate = rot_mat.T.dot(global_vec_to_gate)
        
        # 为相对坐标注入少量传感器估算噪声 (模拟真实 VIO 的漂移)
        local_vec_noisy = local_vec_to_gate + np.random.normal(0, 0.1, size=3)
        
        # --- 注入 IMU 和速度噪声 ---
        rpy_noisy = rpy + np.random.normal(0, self.cfg.NOISE_IMU_STD, size=3)
        quat_noisy = p.getQuaternionFromEuler(rpy_noisy)
        vel_noisy = np.array(vel) + np.random.normal(0, 0.05, size=3)
        ang_vel_noisy = np.array(ang_vel) + np.random.normal(0, 0.05, size=3)
        norm_rpm = self.current_real_rpm / self.unwrapped.MAX_RPM
        
        # === 全新本体感知拼接：加入罗盘信息 ===
        proprio = np.concatenate([
            quat_noisy, vel_noisy, ang_vel_noisy, norm_rpm, local_vec_noisy
        ]).astype(np.float32)
        proprio = np.clip(proprio, -10.0, 10.0) # 异常值裁剪
        
        # --- 深度图渲染与噪声注入 ---
        depth_img = self.world.get_depth_vision(np.array(pos), quat)
        if np.random.rand() < self.cfg.NOISE_DEPTH_PROB:
            # 椒盐噪声
            mask = np.random.rand(*depth_img.shape) < 0.05
            depth_img[mask] = np.random.choice([0.0, 1.0])
        depth_img += np.random.normal(0, self.cfg.NOISE_DEPTH_STD, size=depth_img.shape)
        depth_img = np.clip(depth_img, 0.0, 1.0).astype(np.float32)
        
        # --- Critic 特权上帝视角 ---
        gate = self.gate_poses[min(self.current_target_gate_idx, self.world.cfg.NUM_GATES-1)]
        target_pos = gate['pos']
        target_norm = gate['tangent']
        
        # 获取未来 5 个样条点的相对坐标
        spline_refs = []
        # 寻找最近的样条点索引
        dists = np.linalg.norm(self.dense_spline_points - np.array(pos), axis=1)
        nearest_idx = np.argmin(dists)
        for i in range(1, 6):
            idx = min(nearest_idx + i * 5, len(self.dense_spline_points) - 1)
            spline_refs.extend(self.dense_spline_points[idx] - np.array(pos))
            
        priv = np.concatenate([
            pos, vel, target_pos, target_norm, spline_refs
        ]).astype(np.float32)
        priv = np.clip(priv, -10.0, 10.0)
        
        return depth_img, proprio, priv

    def _build_obs_dict(self, priv_obs):
        return {
            "depth_img": np.array(self.depth_buffer, dtype=np.float32),
            "proprioception": np.concatenate(list(self.proprio_buffer), dtype=np.float32),
            "critic_privileged": priv_obs
        }

    # ---------------------------------------------------------
    # 核心 4：奖励引擎与状态机
    # ---------------------------------------------------------
    def _compute_reward_and_done(self, action_nn):
        pos, quat = p.getBasePositionAndOrientation(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        vel, _ = p.getBaseVelocity(self.unwrapped.DRONE_IDS[0], physicsClientId=self.unwrapped.CLIENT)
        pos = np.array(pos); vel = np.array(vel)
        rpy = np.array(p.getEulerFromQuaternion(quat))
        
        # 将四元数转换为旋转矩阵，提取无人机局部坐标轴在世界坐标系下的朝向
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        x_body_world = rot_mat[:, 0]  # 无人机正前方 (机头) 向量
        
        # 1. 基础惩罚与平滑惩罚
        r_step = self.cfg.R_STEP
        r_smooth = self.cfg.R_SMOOTH * float(np.sum(np.square(action_nn - self.prev_action)))
        
        # ==========================================
        # 2. 跟踪奖励 (高斯距离衰减 * 切向速度分量)
        # ==========================================
        # 找到无人机距离最近的样条点
        dists = np.linalg.norm(self.dense_spline_points - pos, axis=1)
        nearest_idx = np.argmin(dists)
        min_dist = dists[nearest_idx]
        
        # 计算样条曲线在当前位置的切线方向 (利用下一个点减去当前点)
        next_idx = min(nearest_idx + 1, len(self.dense_spline_points) - 1)
        if next_idx == nearest_idx:
            next_idx = max(nearest_idx - 1, 0) # 容错处理：到达终点时
            
        tangent_spline = self.dense_spline_points[next_idx] - self.dense_spline_points[nearest_idx]
        tangent_spline = tangent_spline / (np.linalg.norm(tangent_spline) + 1e-6)
        
        # 计算无人机速度在曲线切线上的投影分量 (跑得越快，分量越大)
        v_tangent = float(np.dot(vel, tangent_spline))
        v_tangent_clipped = np.clip(v_tangent, -5.0, 5.0) # 限制预期最大速度，防止瞬间梯度爆炸
        
        # 如果倒车 (分量为负)，给予额外惩罚；如果前进，通过高斯距离进行衰减
        if v_tangent_clipped < 0:
            r_track = self.cfg.R_TRACK_V_SCALE * v_tangent_clipped * 1.5 
        else:
            # 只有在距离样条线很近时，速度奖励才能拿满；偏离航线则拿不到速度分
            r_track = self.cfg.R_TRACK_V_SCALE * v_tangent_clipped * math.exp(-self.cfg.R_TRACK_K * (min_dist ** 2))
        
        # ==========================================
        # 3. 姿态对齐奖励 (机头姿态向量 * 目标门法向量)
        # ==========================================
        gate = self.gate_poses[min(self.current_target_gate_idx, self.world.cfg.NUM_GATES-1)]
        target_norm = gate['tangent'] # 目标门的法向量
        
        # 奖励无人机将其正前方 (机头) 对准门的开口方向
        pose_dot_prod = float(np.dot(x_body_world, target_norm))
        r_align = self.cfg.R_ALIGN_POSE * pose_dot_prod
            
        # 汇总连续奖励并强制裁剪在 [-1, 1] 以保证数值稳定
        raw_cont_reward = r_step + r_smooth + r_track + r_align
        clipped_cont_reward = float(np.clip(raw_cont_reward, -1.0, 1.0))
        
        # ==========================================
        # 4. 终端极刑与状态机
        # ==========================================
        terminal_reward = 0.0
        terminated = False
        truncated = False
        reason = "ALIVE"
        
        # a. 极刑：触地或过度侧翻
        if pos[2] < self.cfg.CRASH_Z_MIN or max(abs(rpy[0]), abs(rpy[1])) > self.cfg.CRASH_RP_MAX:
            terminal_reward = self.cfg.R_CRASH
            terminated = True
            reason = "CRASH_FLIP_OR_FLOOR"
            
        # b. 极刑：撞墙或偏离赛道过远 (>3m)
        elif abs(pos[1]) > (self.world.cfg.ARENA_WIDTH/2 - 0.2) or min_dist > 3.0:
            terminal_reward = self.cfg.R_CRASH
            terminated = True
            reason = "CRASH_WALL_OR_DEVIATE"
            
        # c. 穿门判定检测 (修复几何 Bug 版: X轴截断判定)
        if not terminated and self.current_target_gate_idx < self.world.cfg.NUM_GATES:
            gate_pos = gate['pos']
            vec_to_drone = pos - gate_pos
            
            # 当且仅当无人机真正飞过了门心所在的 X 坐标时，才进行投影核算
            if pos[0] > gate_pos[0]: 
                dist_to_center = np.linalg.norm(vec_to_drone - np.dot(vec_to_drone, target_norm) * target_norm)
                
                if dist_to_center < self.cfg.PASS_GATE_DIST:
                    # 完美穿过！发放阶梯大奖
                    terminal_reward += self.cfg.R_GATE_BASE * (self.current_target_gate_idx + 1)
                    self.passed_gate_flags[self.current_target_gate_idx] = True
                    self.current_target_gate_idx += 1
                    
                    if self.current_target_gate_idx >= self.world.cfg.NUM_GATES:
                        terminal_reward += self.cfg.R_SUCCESS
                        terminated = True
                        reason = "SUCCESS_ALL_GATES"
                else:
                    # 从门外绕过去，判定为撞击
                    terminal_reward = self.cfg.R_CRASH
                    terminated = True
                    reason = "CRASH_MISSED_GATE"
                    
        # d. 懦夫惩罚：超时等同于坠毁
        if not terminated and self.step_counter >= self.cfg.MAX_STEPS:
            truncated = True
            terminated = True
            reason = "TIMEOUT"
            terminal_reward = self.cfg.R_TIMEOUT_PENALTY

        total_reward = clipped_cont_reward + terminal_reward
        
        step_info = {
            'r_cont_clipped': clipped_cont_reward, 'r_track': r_track, 'r_align': r_align,
            'r_terminal': terminal_reward, 'total_reward': total_reward,
            'passed_gates': self.current_target_gate_idx, 'reason': reason
        }
        
        return total_reward, terminated, truncated, step_info

    def _generate_dense_spline(self):
        """生成密集样条参考点供奖励计算与上帝视角使用"""
        pts = [self.world.cfg.START_POS] + [g['pos'] for g in self.gate_poses]
        pts = np.array(pts)
        tck, _ = splprep([pts[:,0], pts[:,1], pts[:,2]], s=0, k=min(3, len(pts)-1))
        u_fine = np.linspace(0, 1, 200) # 生成 200 个密集参考点
        path = splev(u_fine, tck)
        self.dense_spline_points = np.vstack(path).T


# ==========================================
# 模块 3：Plot (训练监控数据绘图类)
# ==========================================
class Task4Plot:
    @staticmethod
    def plot_learning_curves(history_steps, history_rewards, history_success, history_gates, save_path="task4_curves.png"):
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        axs[0].plot(history_steps, history_rewards, 'b-', linewidth=2)
        axs[0].set_title('Episode Reward')
        axs[0].grid(True)
        
        axs[1].plot(history_steps, history_success, 'g-', linewidth=2)
        axs[1].set_title('Success Rate (%)')
        axs[1].grid(True)
        
        axs[2].plot(history_steps, history_gates, 'purple', linewidth=2)
        axs[2].set_title('Avg Gates Passed')
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()