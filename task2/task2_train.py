import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task2_env import Task2Env, Task2Config, Task2Plot

# ==========================================
# 1. 环境并行工厂函数
# ==========================================
def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        # 无头模式 (gui=False) 极速训练
        base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
        env = Task2Env(base_env)
        env = Monitor(env)  # 包装 Monitor 以记录 episode reward 和 length
        env.reset(seed=seed + rank)
        return env
    return _init

# ==========================================
# 2. 动态学习率调度器
# ==========================================
def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """学习率从 initial_value 线性衰减到 final_value，防止后期参数震荡"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

# ==========================================
# 3. 深度透视日志器 (接管环境与 PPO 底层指标)
# ==========================================
class DeepMonitorCallback(BaseCallback):
    def __init__(self, check_freq: int, max_steps: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.max_steps = max_steps
        
        # 环境数据缓冲
        self.ep_rewards = []
        self.ep_lengths = []
        self.completion_rates = []
        self.death_reasons = []
        self.action_buffer = []
        
        # 宏观绘图历史
        self.history_steps = []
        self.history_rewards = []
        self.history_completion = []

    def _on_step(self) -> bool:
        # 收集动作以计算均值和方差
        actions = self.locals.get("actions")
        if actions is not None:
            self.action_buffer.append(actions)

        # 抓取环境的 Info 字典
        for info in self.locals.get("infos", []):
            # 当回合结束时，Monitor 会写入 "episode"
            if "episode" in info and "task2_stats" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                
                stats = info["task2_stats"]
                self.completion_rates.append(stats.get('completion_rate', 0.0))
                
                # 诊断死因
                r_term = stats.get('r_terminal', 0.0)
                if r_term >= Task2Config.R_SUCCESS_BASE:
                    reason = "SUCCESS"
                elif r_term == Task2Config.R_DEVIATE:
                    reason = "DEVIATE"
                elif r_term == Task2Config.R_CRASH:
                    reason = "CRASH"
                else:
                    reason = "TIMEOUT"
                self.death_reasons.append(reason)

        # 定期输出日志
        if self.num_timesteps % self.check_freq == 0:
            self._print_and_record_log()
            # 清空动作缓冲，防止内存溢出
            self.action_buffer = [] 
                
        return True

    def _print_and_record_log(self):
        if not self.ep_rewards:
            return
            
        # 1. 计算环境统计学指标 (取最近 100 局)
        mean_rew = np.mean(self.ep_rewards[-100:])
        mean_len = np.mean(self.ep_lengths[-100:])
        mean_comp = np.mean(self.completion_rates[-100:])
        
        recent_reasons = self.death_reasons[-100:]
        reason_counts = Counter(recent_reasons)
        reason_str = ", ".join([f"{k}: {v/len(recent_reasons)*100:.0f}%" for k, v in reason_counts.items()])
        
        act_arr = np.array(self.action_buffer)
        act_mean = np.mean(act_arr) if len(act_arr) > 0 else 0.0
        act_std = np.std(act_arr) if len(act_arr) > 0 else 0.0
        
        # 2. 读取 SB3 Logger 中的 PPO 底层指标 (来自上一次更新)
        logger_dict = self.logger.name_to_value
        kl = logger_dict.get("train/approx_kl", 0.0)
        ent = logger_dict.get("train/entropy_loss", 0.0)
        p_loss = logger_dict.get("train/policy_gradient_loss", 0.0)
        v_loss = logger_dict.get("train/value_loss", 0.0)
        net_std = logger_dict.get("train/std", 0.0)
        
        # 3. 结构化控制台输出
        print(f"\n[{self.num_timesteps:08d} 步] -----------------------------------------")
        print(f"🌍 [环境指标] 平均奖: {mean_rew:6.1f} | 存活: {mean_len:5.1f} | 轨迹完成度: {mean_comp:5.1f}%")
        print(f"💀 [终局诊断] {reason_str}")
        print(f"🧠 [PPO 内核] KL散度: {kl:.4f} | 策略熵: {ent:.4f} | 网络Std: {net_std:.4f}")
        print(f"📉 [网络损失] Pi_Loss: {p_loss:.4f} | V_Loss: {v_loss:.4f}")
        print(f"🕹️ [动作输出] 均值: {act_mean:.3f} | 方差: {act_std:.3f}")
        
        # 记录用于宏观绘图
        self.history_steps.append(self.num_timesteps)
        self.history_rewards.append(mean_rew)
        self.history_completion.append(mean_comp)

    def plot_macro_trends(self, save_path="task2_macro_trend.png"):
        """绘制整个训练过程的宏观成长曲线"""
        if not self.history_steps:
            return
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history_steps, self.history_rewards, 'b-', linewidth=2)
        plt.xlabel('Timesteps')
        plt.ylabel('Episode Reward (Mean of 100)')
        plt.title('Learning Curve (Reward)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history_steps, self.history_completion, 'g-', linewidth=2)
        plt.axhline(y=100.0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Timesteps')
        plt.ylabel('Completion Rate (%)')
        plt.title('Trajectory Tracking Capability')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\n✅ 宏观训练趋势图已保存至: {save_path}")
        plt.close()

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    print("=" * 65)
    print("🚀 [Task 2] 动态航点追踪 - PPO 分布式训练启动")
    print("=" * 65)
    
    # --- 核心训练参数 ---
    NUM_CPU = 10                     # 开启 10 个并行环境
    TOTAL_TIMESTEPS = 15_000_000      
    SAVE_DIR = "./models/task2"
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_random_seed(42)
    
    # --- 实例化环境与归一化 ---
    vec_env = SubprocVecEnv([make_env(i, seed=42) for i in range(NUM_CPU)])
    # 【稳定性调优 1】：对 100 维观测值进行滑动均值归一化，极大加速深层 MLP 收敛
    # 注意：绝不归一化 reward，因为我们精心设计并 clip 了边界！
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # --- PPO 算法与网络配置 ---
    # 【稳定性调优 2】：深层网络结构，SB3 默认使用正交初始化 (Orthogonal Init)，非常适合此结构。
    policy_kwargs = dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]))
    
    # 每个环境每轮收集 2048 步，10个环境 = 20480 步/轮 (Rollout Size)
    ROLLOUT_SIZE = 2048 
    BATCH_SIZE = 1024
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(2e-4, 1e-5),   
        n_steps=ROLLOUT_SIZE,                
        batch_size=BATCH_SIZE,              
        n_epochs=10,                 
        gamma=0.99,                  
        gae_lambda=0.95,             
        ent_coef=0.01,               # 探索熵系数
        max_grad_norm=0.5,           # 【稳定性调优 3】：梯度裁剪，防止偶尔的极值导致梯度爆炸
        target_kl=0.015,             # 【稳定性调优 4】：安全锁！更新若超出 KL 阈值立即停止该 Epoch
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"{SAVE_DIR}/tensorboard/",
        verbose=0  # 关闭默认输出，全权交由我们的 DeepMonitorCallback 处理
    )
    
    print(f"[INFO] 架构加载完毕 | 网络维度: {policy_kwargs['net_arch']} | 计算设备: {model.device}")
    
    # --- 回调链配置 ---
    # 1. 检查点回调：每 50万 步保存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // NUM_CPU, 1), 
        save_path=SAVE_DIR,
        name_prefix="ppo_tracker"
    )
    
    # 2. 自定义透视日志：每次 Rollout 结束时打印 (20480 步)
    console_logger = DeepMonitorCallback(
        check_freq=ROLLOUT_SIZE * NUM_CPU, 
        max_steps=Task2Config.MAX_STEPS
    )
    
    # --- 开始训练 ---
    print(f"\n[INFO] 开始执行 {TOTAL_TIMESTEPS} 步训练，等待首次 Rollout 收集完毕...\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, console_logger],
        progress_bar=False 
    )
    
    # --- 持久化保存 ---
    final_model_path = os.path.join(SAVE_DIR, "ppo_tracker_final.zip")
    vec_norm_path = os.path.join(SAVE_DIR, "vec_normalize.pkl")
    model.save(final_model_path)
    vec_env.save(vec_norm_path)
    print(f"\n✅ 训练结束！模型已保存至: {final_model_path}")
    
    # --- 训练结束后自动化数据绘图 ---
    console_logger.plot_macro_trends(save_path=os.path.join(SAVE_DIR, "task2_macro_trend.png"))
    
    print("\n[INFO] 正在抽取 1 局进行微观性能评估 (Micro Performance)...")
    eval_base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    eval_env = Task2Env(eval_base_env)
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    
    # 冻结归一化层参数
    eval_vec_env = VecNormalize.load(vec_norm_path, eval_vec_env)
    eval_vec_env.training = False 
    eval_vec_env.norm_reward = False
    
    obs = eval_vec_env.reset()
    infos_list = []
    
    for _ in range(Task2Config.MAX_STEPS):
        # 评估模式：关闭探索噪声，输出最确定的动作
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_vec_env.step(action)
        infos_list.append(info[0]['task2_stats'])
        if done[0]:
            break
            
    plot_path = os.path.join(SAVE_DIR, "task2_micro_tracking.png")
    Task2Plot.plot_tracking_performance(infos_list, save_path=plot_path)
    eval_vec_env.close()

if __name__ == "__main__":
    main()