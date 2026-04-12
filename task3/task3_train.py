import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import collections
from typing import Callable
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from task3_env import Task3Env, Task3Config

# ==========================================
# 1. 环境工厂函数
# ==========================================
def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
        env = Task3Env(base_env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

# ==========================================
# 2. 基础学习率衰减器
# ==========================================
def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

# ==========================================
# 3. 极简版训练监控器 (仅做数据记录与打印)
# ==========================================
class PhaseMonitor(BaseCallback):
    def __init__(self, check_freq: int, phase_name: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.phase_name = phase_name
        
        self.ep_rewards = collections.deque(maxlen=100)
        self.ep_lengths = collections.deque(maxlen=100)
        self.final_dists = collections.deque(maxlen=100)
        self.death_reasons = collections.deque(maxlen=100)
        self.action_buffer = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            self.action_buffer.append(actions)
            
        for info in self.locals.get("infos", []):
            if "episode" in info and "task3_stats" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                
                stats = info["task3_stats"]
                self.final_dists.append(stats.get('dist_xy', 0.0))
                self.death_reasons.append(stats.get('reason', 'UNKNOWN'))
                
        if self.num_timesteps % self.check_freq == 0:
            self._print_log()
            self.action_buffer = []
            
        return True

    def _print_log(self):
        if not self.ep_rewards: return
        mean_rew = np.mean(self.ep_rewards)
        mean_len = np.mean(self.ep_lengths)
        mean_dist = np.mean(self.final_dists)
        
        reason_counts = Counter(self.death_reasons)
        success_rate = (reason_counts.get("SUCCESS", 0) / len(self.death_reasons)) * 100.0
        reason_str = ", ".join([f"{k}: {v/len(self.death_reasons)*100:.0f}%" for k, v in reason_counts.items()])
        
        act_arr = np.array(self.action_buffer)
        act_mean = np.mean(act_arr) if len(act_arr) > 0 else 0.0
        act_std = np.std(act_arr) if len(act_arr) > 0 else 0.0
        
        # 实时获取学习率
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        
        print(f"\n[{self.num_timesteps:08d} 步 | {self.phase_name}] {'-'*30}")
        print(f"🌍 [指标] 奖: {mean_rew:5.1f} | 存活: {mean_len:5.1f} | 成功率: {success_rate:5.1f}% | 距终点: {mean_dist:5.1f}m")
        print(f"💀 [死因] {reason_str}")
        print(f"🧠 [PPO] 动作均值: {act_mean:.3f} (Std: {act_std:.3f}) | LR: {current_lr:.2e}")

# ==========================================
# 4. 主训练流程 (分阶段串行调用)
# ==========================================
def main():
    print("=" * 65)
    print("🚀 [Task 3] 定高避障平飞 - 串行分段训练版")
    print("=" * 65)
    
    NUM_CPU = 10
    SAVE_DIR = "./models/task3"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    set_random_seed(42)
    vec_env = SubprocVecEnv([make_env(i, seed=42) for i in range(NUM_CPU)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))
    ROLLOUT_SIZE = 2048
    BATCH_SIZE = 1024
    
    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4, 1e-5),  # 基础调度器，每次 learn 会自动重置进度
        n_steps=ROLLOUT_SIZE, batch_size=BATCH_SIZE, n_epochs=10,
        gamma=0.99, gae_lambda=0.95,
        ent_coef=0.01, max_grad_norm=0.5, target_kl=0.015,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"{SAVE_DIR}/tensorboard/", verbose=0
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=max(1_000_000 // NUM_CPU, 1), save_path=SAVE_DIR, name_prefix="ppo_obstacle")

    # ==========================================================
    # 🌟 第一阶段：无障碍基础导航 (1000万步)
    # ==========================================================
    print(f"\n[INFO] >>> 开启第一阶段：简单避障 <<<")
    vec_env.env_method("set_curriculum", num_static=5, num_dynamic=0, max_sg_dist=25.0)
    phase1_logger = PhaseMonitor(check_freq=ROLLOUT_SIZE * NUM_CPU, phase_name="阶段 1")
    
    model.learn(
        total_timesteps=20_000_000, 
        callback=[checkpoint_callback, phase1_logger], 
        reset_num_timesteps=True # 第一次设为 True，初始化全局步数
    )
    model.save(os.path.join(SAVE_DIR, "ppo_phase1_done.zip"))

    # ==========================================================
    # 🌟 第二阶段：静态障碍物避让 (1500万步)
    # ==========================================================
    print(f"\n[INFO] >>> 开启第二阶段：一般避障 <<<")
    vec_env.env_method("set_curriculum", num_static=10, num_dynamic=2, max_sg_dist=25.0)
    phase2_logger = PhaseMonitor(check_freq=ROLLOUT_SIZE * NUM_CPU, phase_name="阶段 2")
    
    model.learn(
        total_timesteps=20_000_000, 
        callback=[checkpoint_callback, phase2_logger], 
        reset_num_timesteps=False # 保持 False，让 Tensorboard 步数连续
    )
    model.save(os.path.join(SAVE_DIR, "ppo_phase2_done.zip"))

    # ==========================================================
    # 🌟 第三阶段：动态障碍物综合测试 (2500万步)
    # ==========================================================
    print(f"\n[INFO] >>> 开启第三阶段：复杂避障 <<<")
    vec_env.env_method("set_curriculum", num_static=25, num_dynamic=4, max_sg_dist=45.0)
    phase3_logger = PhaseMonitor(check_freq=ROLLOUT_SIZE * NUM_CPU, phase_name="阶段 3")
    
    model.learn(
        total_timesteps=20_000_000, 
        callback=[checkpoint_callback, phase3_logger], 
        reset_num_timesteps=False 
    )

    # 最终保存
    final_model_path = os.path.join(SAVE_DIR, "ppo_obstacle_final.zip")
    vec_norm_path = os.path.join(SAVE_DIR, "vec_normalize.pkl")
    model.save(final_model_path)
    vec_env.save(vec_norm_path)
    print(f"\n✅ 全部训练结束！终极模型已保存至: {final_model_path}")

if __name__ == "__main__":
    main()