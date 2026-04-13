import os
import torch
import torch.nn as nn
import numpy as np
import collections
from collections import Counter
from typing import Callable, Dict
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task4_env import Task4Env, Task4Plot

# ==========================================
# 1. 环境工厂函数
# ==========================================
def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        # 无头模式 (gui=False)，出生点匹配世界模型
        base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[-12.0, 0.0, 1.5]]))
        env = Task4Env(base_env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

# ==========================================
# 2. 学习率调度器
# ==========================================
def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

# ==========================================
# 3. 核心创新：非对称特征提取器 (Asymmetric AC)
# ==========================================
class AsymmetricFeaturesExtractor(BaseFeaturesExtractor):
    """
    定制的多输入特征提取器：
    - 使用 CNN 处理 64x64 深度图 (3帧堆叠)。
    - 使用 MLP 处理 本体感知。
    - 使用 MLP 处理 Critic的特权信息 (上帝视角坐标)。
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=1) # 临时特征维度，稍后覆写
        
        # --- 1. 深度图 CNN 提取器 ---
        n_input_channels = observation_space.spaces["depth_img"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 自动计算 CNN 展平后的维度
        with torch.no_grad():
            sample_tensor = torch.as_tensor(observation_space.spaces["depth_img"].sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]
        self.cnn_linear = nn.Sequential(nn.Linear(n_flatten, cnn_output_dim), nn.ReLU())

        # --- 2. 本体感知 MLP ---
        proprio_dim = observation_space.spaces["proprioception"].shape[0]
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        # --- 3. 上帝视角 (Privileged) MLP ---
        self.priv_dim = observation_space.spaces["critic_privileged"].shape[0]
        self.priv_mlp = nn.Sequential(
            nn.Linear(self.priv_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        # 总特征维度 = CNN(256) + 本体(64) + 特权(64) = 384
        self._features_dim = cnn_output_dim + 64 + 64
        
        # 核心致盲标记 (将在自定义 Policy 中被动态赋值)
        self.is_actor = False 

    def forward(self, observations) -> torch.Tensor:
        depth_feat = self.cnn_linear(self.cnn(observations["depth_img"]))
        proprio_feat = self.proprio_mlp(observations["proprioception"])

        if self.is_actor:
            # 将特权特征全部置为 0，Actor 的全连接层在这个维度上不会获得任何梯度更新，
            # 从而强制它只能从深度图和本体感知中学习控制策略。
            priv_feat = torch.zeros((depth_feat.shape[0], 64), device=depth_feat.device)
        else:
            # Critic 能够看到上帝视角的特权特征，用于精确计算 Value
            priv_feat = self.priv_mlp(observations["critic_privileged"])

        return torch.cat([depth_feat, proprio_feat, priv_feat], dim=1)

# 配合特征提取器的自定义策略类
class AsymmetricPolicy(MultiInputActorCriticPolicy):
    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # 强制将 Actor 的提取器标记为致盲模式，Critic 保持默认
        self.pi_features_extractor.is_actor = True
        self.vf_features_extractor.is_actor = False
class SaveVecNormalizeCallback(BaseCallback):
    """
    定期保存 VecNormalize 统计数据的回调函数
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(path)
        return True
# ==========================================
# 4. 深度监控器与日志回调
# ==========================================
class DeepMonitorCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        
        self.ep_rewards = collections.deque(maxlen=100)
        self.ep_lengths = collections.deque(maxlen=100)
        self.ep_gates = collections.deque(maxlen=100)
        self.death_reasons = collections.deque(maxlen=100)
        
        # 用于最终画图的全局历史
        self.history_steps = []
        self.history_rewards = []
        self.history_success = []
        self.history_gates = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info and "task4_stats" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                stats = info["task4_stats"]
                self.ep_gates.append(stats.get('passed_gates', 0))
                self.death_reasons.append(stats.get('reason', 'UNKNOWN'))
                
        if self.num_timesteps % self.check_freq == 0:
            self._print_and_record_log()
            
        return True

    def _print_and_record_log(self):
        if not self.ep_rewards: return
        mean_rew = np.mean(self.ep_rewards)
        mean_len = np.mean(self.ep_lengths)
        mean_gates = np.mean(self.ep_gates)
        
        reason_counts = Counter(self.death_reasons)
        success_rate = (reason_counts.get("SUCCESS_ALL_GATES", 0) / len(self.death_reasons)) * 100.0
        reason_str = ", ".join([f"{k}: {v/len(self.death_reasons)*100:.0f}%" for k, v in reason_counts.items()])
        
        # 读取 SB3 内部记录器的数据
        logger_dict = self.logger.name_to_value
        kl = logger_dict.get("train/approx_kl", 0.0)
        ent = logger_dict.get("train/entropy_loss", 0.0)
        v_loss = logger_dict.get("train/value_loss", 0.0)
        clip_frac = logger_dict.get("train/clip_fraction", 0.0)
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        
        print(f"\n[{self.num_timesteps:08d} 步] {'-'*45}")
        print(f"🌍 [环境指标] 奖: {mean_rew:6.1f} | 存活: {mean_len:5.1f}步 | 平均穿门: {mean_gates:3.1f}个 | 成功率: {success_rate:4.1f}%")
        print(f"💀 [死因占比] {reason_str}")
        print(f"🧠 [PPO内视] KL: {kl:.4f} | 熵: {ent:.4f} | V_Loss: {v_loss:6.2f} | 截断率: {clip_frac:.3f} | LR: {current_lr:.2e}")
        
        self.history_steps.append(self.num_timesteps)
        self.history_rewards.append(mean_rew)
        self.history_success.append(success_rate)
        self.history_gates.append(mean_gates)

# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    print("=" * 65)
    print("🚀 [Task 4] 极限穿梭训练启动")
    print("=" * 65)
    
    # 训练配置
    NUM_CPU = 10
    TOTAL_TIMESTEPS = 50_000_000 # 5000万步小目标
    SAVE_DIR = "./models/task4"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    set_random_seed(42)
    
    # 构建并行环境并进行观测值归一化 (不归一化奖励，避免影响阶梯奖励的设计)
    vec_env = SubprocVecEnv([make_env(i, seed=42) for i in range(NUM_CPU)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 网络架构与算法超参数
    policy_kwargs = dict(
        features_extractor_class=AsymmetricFeaturesExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        share_features_extractor=False, 
        net_arch=dict(pi=[256, 256], vf=[256, 256]) # 纯 MLP 骨干
    )
    
    ROLLOUT_SIZE = 2048 
    BATCH_SIZE = 1024
    
    # 初始化 PPO 模型
    model = PPO(
        AsymmetricPolicy, # 使用我们定制的策略类
        vec_env, 
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(3e-4, 1e-5), 
        n_steps=ROLLOUT_SIZE, 
        batch_size=BATCH_SIZE, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        ent_coef=0.01,           # 鼓励视觉初期的探索
        max_grad_norm=0.5,       # 梯度裁剪，稳定网络
        target_kl=0.015,         # KL 散度安全锁，防走火入魔
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=f"{SAVE_DIR}/tensorboard/", 
        verbose=0
    )
    
    # 回调配置
    checkpoint_callback = CheckpointCallback(save_freq=max(1_000_000 // NUM_CPU, 1), save_path=SAVE_DIR, name_prefix="ppo_vision")
    vec_norm_callback = SaveVecNormalizeCallback(save_freq=max(1_000_000 // NUM_CPU, 1), save_path=SAVE_DIR)
    console_logger = DeepMonitorCallback(check_freq=ROLLOUT_SIZE * NUM_CPU)
    
    print(f"\n[INFO] 正在将模型加载入 GPU，准备进行环境预热...")
    
    # 启动训练
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_callback, vec_norm_callback, console_logger], progress_bar=False)
    
    # 训练结束后的持久化与画图
    final_model_path = os.path.join(SAVE_DIR, "ppo_vision_final.zip")
    vec_norm_path = os.path.join(SAVE_DIR, "vec_normalize.pkl")
    model.save(final_model_path)
    vec_env.save(vec_norm_path)
    
    print(f"\n✅ 5000万步训练圆满结束！终极视觉模型已保存至: {final_model_path}")
    
    # 调用数据绘图接口
    plot_path = os.path.join(SAVE_DIR, "task4_learning_trend.png")
    Task4Plot.plot_learning_curves(
        console_logger.history_steps, 
        console_logger.history_rewards, 
        console_logger.history_success, 
        console_logger.history_gates, 
        save_path=plot_path
    )
    print(f"📈 训练曲线图已生成: {plot_path}")

if __name__ == "__main__":
    main()