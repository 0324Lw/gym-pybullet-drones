import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task1_env import Task1Env, Task1Plot, Task1Config

# ==========================================
# 1. 环境工厂函数 (用于多进程并行)
# ==========================================
def make_env(rank: int, seed: int = 0) -> Callable:
    """
    生成独立环境的工厂函数
    :param rank: 进程编号，用于确保不同环境的随机种子不同
    """
    def _init():
        # 无头模式，初始抛飞高度 1.0m
        base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
        env = Task1Env(base_env)
        
        # 使用 Monitor 包装器，自动记录回合的总奖励和步数，供 SB3 和回调函数调用
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

# ==========================================
# 2. 动态学习率调度器
# ==========================================
def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """
    线性学习率衰减函数
    :param progress_remaining: 剩余进度 (从 1.0 下降到 0.0)
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

# ==========================================
# 3. 自定义控制台日志与监控回调
# ==========================================
class CustomConsoleLogger(BaseCallback):
    """自定义回调函数：聚合并行环境的数据，整齐输出训练信息"""
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_successes = []
        self.ep_z_heights = []
        
        # 用于记录历史数据绘制训练曲线
        self.history_steps = []
        self.history_rewards = []
        self.history_success = []

    def _on_step(self) -> bool:
        # 遍历当前所有并行环境返回的 infos 字典
        for info in self.locals.get("infos", []):
            if "task1_stats" in info:
                self.ep_z_heights.append(info["task1_stats"]["pos_z"])
            
            # 当且仅当一个环境回合结束时，Monitor 会写入 "episode" 字典
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                
                # 判定成功：如果终局原始奖励等于我们设定的 R_SUCCESS (10.0)，说明成功悬停
                is_success = 1.0 if info.get("task1_stats", {}).get("r_raw_total", 0) > 5.0 else 0.0
                self.ep_successes.append(is_success)

        # 达到输出频率时，计算均值并打印控制台
        if self.num_timesteps % self.check_freq == 0:
            if len(self.ep_rewards) > 0:
                mean_rew = np.mean(self.ep_rewards[-100:])  # 统计最近 100 局
                mean_len = np.mean(self.ep_lengths[-100:])
                success_rate = np.mean(self.ep_successes[-100:]) * 100
                mean_z = np.mean(self.ep_z_heights[-1000:]) if len(self.ep_z_heights) > 0 else 0
                
                print(f"[{self.num_timesteps:07d} 步] "
                      f"平均截断奖励: {mean_rew:7.2f} | "
                      f"平均生存步数: {mean_len:5.1f} | "
                      f"当前平均高度: {mean_z:5.3f}m | "
                      f"悬停成功率: {success_rate:5.1f}%")
                
                # 保存用于绘图的历史数据
                self.history_steps.append(self.num_timesteps)
                self.history_rewards.append(mean_rew)
                self.history_success.append(success_rate)
                
                # 清理高度缓存防止内存溢出
                self.ep_z_heights = self.ep_z_heights[-1000:]
                
        return True

    def plot_training_curves(self, save_path="training_curves.png"):
        """训练结束后绘制全局趋势图"""
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history_steps, self.history_rewards, label='Mean Reward (Last 100 eps)', color='blue')
        plt.xlabel('Timesteps')
        plt.ylabel('Clipped Reward')
        plt.title('Training Reward Trend')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history_steps, self.history_success, label='Success Rate (%)', color='green')
        plt.xlabel('Timesteps')
        plt.ylabel('Success Rate (%)')
        plt.title('Hover Success Rate Trend')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"\n✅ 训练宏观趋势图已保存至: {save_path}")
        plt.close()

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    print("=" * 60)
    print("🚀 [无人机悬停] PPO 分布式强化学习训练启动")
    print("=" * 60)
    
    # 基础设置
    NUM_CPU = 8                     # 并行环境数量
    TOTAL_TIMESTEPS = 3_000_000     # 总训练步数 (100万步作为基准)
    SAVE_DIR = "./models/task1"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 固定随机种子
    set_random_seed(42)
    
    # 创建并包装 8 进程并行环境
    vec_env = SubprocVecEnv([make_env(i, seed=42) for i in range(NUM_CPU)])
    
    # [核心调优] 挂载在线滑动均值归一化 (只归一化观测状态，不归一化奖励，因为奖励我们已经在 env 中截断过了)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 初始化 PPO 网络
    # Policy网络结构: 256x256，Value网络结构: 256x256
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4, 1e-5), # 线性学习率衰减
        n_steps=1024,                # 每个进程收集 1024 步后进行一次更新 (Batch Size = 8 * 1024 = 8192)
        batch_size=512,              # 梯度下降批次
        n_epochs=10,                 # 每次更新迭代 10 轮
        gamma=0.99,                  # 远期收益折扣
        gae_lambda=0.95,             # 优势函数平滑参数
        ent_coef=0.005,              # 鼓励初期探索的熵系数
        max_grad_norm=0.5,           # 梯度裁剪
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"{SAVE_DIR}/tensorboard/",
        verbose=0                    # 关闭 SB3 自带的冗长打印，使用我们的自定义回调
    )
    
    print(f"[INFO] 模型架构准备完毕，计算设备: {model.device}")
    
    # 定义回调函数链
    # 1. 模型持久化：每 200,000 步保存一次
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // NUM_CPU, 1), 
        save_path=SAVE_DIR,
        name_prefix="ppo_hover"
    )
    
    # 2. 自定义控制台输出：每 10240 步 (约 1 个 Batch 周期) 打印一次状态
    console_logger = CustomConsoleLogger(check_freq=10240)
    
    # 启动训练
    print(f"\n[INFO] 开始执行 {TOTAL_TIMESTEPS} 步训练任务...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, console_logger],
        progress_bar=False 
    )
    
    # 保存最终模型与环境归一化状态
    final_model_path = os.path.join(SAVE_DIR, "ppo_hover_final.zip")
    vec_norm_path = os.path.join(SAVE_DIR, "vec_normalize.pkl")
    model.save(final_model_path)
    vec_env.save(vec_norm_path)
    print(f"\n✅ 训练结束！最佳模型已保存至: {final_model_path}")
    print(f"✅ 状态归一化词典已保存至: {vec_norm_path} (测试时必须加载此文件！)")
    
    # 调用回调函数中的宏观趋势绘图
    console_logger.plot_training_curves(save_path="training_macro_trend.png")
    
    vec_env.close()

    # ==========================================
    # 5. 训练后自动测试并绘制单局微观表现图
    # ==========================================
    print("\n[INFO] 开始运行训练后评估，生成单回合微观测试图...")
    eval_base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    eval_env = Task1Env(eval_base_env)
    
    # 为了测试单机环境，需要包装成 DummyVecEnv 并加载刚才训练好的归一化参数
    from stable_baselines3.common.vec_env import DummyVecEnv
    eval_vec_env = DummyVecEnv([lambda: eval_env])
    eval_vec_env = VecNormalize.load(vec_norm_path, eval_vec_env)
    # 测试时严禁再次更新滑动均值
    eval_vec_env.training = False 
    eval_vec_env.norm_reward = False
    
    obs = eval_vec_env.reset()
    infos_list = []
    
    for _ in range(Task1Config.MAX_STEPS):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_vec_env.step(action)
        
        # SB3 的 VecEnv 会把原始 info 存在一个字典列表里
        real_info = info[0]
        infos_list.append(real_info['task1_stats'])
        if done[0]:
            break
            
    Task1Plot.plot_episode_stats(infos_list, save_path="trained_episode_eval.png")
    eval_vec_env.close()

if __name__ == "__main__":
    main()