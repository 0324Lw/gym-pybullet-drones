import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# 导入我们刚刚编写的 Task2 环境与配置
from task2_env import Task2Env, Task2Config

def run_extreme_tests():
    print("="*65)
    print("🚀 [Phase 1] 空间维度与核心交互 (Step/Reset) 健康体检")
    print("="*65)
    
    # 初始化无头模式环境
    base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    env = Task2Env(base_env)
    
    obs, info = env.reset()
    
    # 状态维度验证：(3欧拉角 + 4力矩 + 3当前相对 + 15前瞻相对) * 4帧 = 100维
    expected_obs_dim = Task2Config.OBS_DIM_PER_FRAME * Task2Config.STACK_SIZE
    print(f"✅ 状态空间检查 | 期望: ({expected_obs_dim},), 实际: {obs.shape} | 数据类型: {obs.dtype}")
    assert obs.shape == (expected_obs_dim,), f"状态空间维度错误！应为 {expected_obs_dim}"
    
    # 动作维度验证：4 个电机的推力偏移 [-1.0, 1.0]
    print(f"✅ 动作空间检查 | 期望: (4,), 实际: {env.action_space.shape} | 范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    assert env.action_space.shape == (4,), "动作空间维度错误！"
    
    # Step 交互验证
    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)
    print(f"✅ Step() 交互检查 | 奖励类型: {type(reward)}, 总奖励: {reward:.3f} | Info 字典抓取: 成功")


    print("\n" + "="*65)
    print("📸 [Phase 2] 静态轨迹生成器测试 (保存 5 张随机 3D 轨迹图)")
    print("="*65)
    
    os.makedirs("./test_plots", exist_ok=True)
    for i in range(5):
        env.reset()
        waypoints = env.waypoints
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 'b-', linewidth=2)
        ax.scatter(waypoints[0, 0], waypoints[0, 1], waypoints[0, 2], color='red', s=100, label='Start (t=0)')
        
        ax.set_title(f"Random Static Trajectory {i+1}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        
        save_path = f"./test_plots/sample_trajectory_{i+1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"  └─> 已保存随机轨迹图 {i+1}: {save_path}")


    print("\n" + "="*65)
    print("💥 [Phase 3] 终局事件 (Terminal Conditions) 强制触发测试")
    print("="*65)
    
    # 1. 强制侧翻 (Flip Crash)
    env.reset()
    print("[测试 1] 注入极端动作 [1, -1, 1, -1] 强制引发侧翻死亡翻滚...")
    for step in range(50):
        _, _, term, trunc, info = env.step(np.array([1.0, -1.0, 1.0, -1.0]))
        if term:
            print(f"  ✅ 侧翻触发成功! 步数: {step+1} | 判定分值: {info['task2_stats']['r_terminal']} (期望: {Task2Config.R_CRASH})")
            break

    # 2. 强制偏离 (Deviate)
    env.reset()
    print("[测试 2] 使用上帝之手将无人机瞬间传送至 10m 外，强制触发偏离极刑...")
    p.resetBasePositionAndOrientation(
        env.unwrapped.DRONE_IDS[0],
        [10.0, 10.0, 10.0],
        p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=env.unwrapped.CLIENT
    )
    # 调用 step 结算判定
    _, _, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
    if term:
        stat = info['task2_stats']
        print(f"  ✅ 偏离触发成功! 误差: {stat['dist_err']:.2f}m | 判定分值: {stat['r_terminal']} (期望: {Task2Config.R_DEVIATE})")

# 3. 强制完成任务 (Success)
    env.reset()
    print("[测试 3] 篡改游标索引，并将无人机传送到终点...")
    # 把目标设为终点
    env.target_idx = len(env.waypoints) - Task2Config.LOOKAHEAD_STEPS - 1
    end_pos = env.waypoints[env.target_idx]
    
    # 【修复】把无人机也传送到终点位置，误差清零！
    p.resetBasePositionAndOrientation(
        env.unwrapped.DRONE_IDS[0],
        end_pos,
        p.getQuaternionFromEuler([0, 0, 0]),
        physicsClientId=env.unwrapped.CLIENT
    )
    
    _, _, term, trunc, info = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
    if trunc:
        print(f"  ✅ 完成任务触发成功! 提前通关奖励 | 判定分值: {info['task2_stats']['r_terminal']:.1f} (期望: > {Task2Config.R_SUCCESS_BASE})")

    print("\n" + "="*65)
    print("🎲 [Phase 4] 随机策略 5000 步采样与 Pandas 数值体检")
    print("="*65)
    
    env.reset()
    stats_list = []
    
    print("⏳ 正在采集 5000 步高频物理交互数据，请稍候...")
    for _ in range(5000):
        action = env.action_space.sample()
        _, _, term, trunc, info = env.step(action)
        stats_list.append(info['task2_stats'])
        
        if term or trunc:
            env.reset()
            
    # 使用 Pandas 分析数据
    df = pd.DataFrame(stats_list)
    
    # 提取我们关心的奖励与指标列
    reward_cols = [
        'r_surv', 'r_track', 'r_vel', 'r_heading', 'r_smooth', 
        'r_cont_clipped', 'r_terminal', 'r_final_total', 'dist_err'
    ]
    df_rewards = df[reward_cols]
    
    # 计算统计特征
    desc = df_rewards.describe().T
    desc['var'] = df_rewards.var()
    
    # 重新排列我们要展示的列顺序
    display_cols = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    result_df = desc[display_cols]
    
    print("📊 Task 2 奖励组件统计数据看板 (5000步随机采样):")
    print("-" * 80)
    print(result_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("-" * 80)
    
    print("\n[环境健康诊断指南]")
    print("1. r_cont_clipped 必须被严格钉在 [-1.0, 1.0] 内部。")
    print("2. r_terminal 的极小值应为 -20.0，极大值可能为空（随机策略跑不到终点）。")
    print("3. r_track (高斯距离) 的最大值应接近正值，证明网络只要靠得近就能吃到肉。")
    
    env.close()

if __name__ == "__main__":
    run_extreme_tests()