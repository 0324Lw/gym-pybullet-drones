import os
import time
import math
import numpy as np
import pandas as pd
import pybullet as p
import gymnasium as gym

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task4_env import Task4Env

def print_header(title):
    print("\n" + "="*70)
    print(f"🚀 [TEST] {title}")
    print("="*70)

def main():
    print_header("初始化 Task4 极限测试环境")
    # 初始化底层环境 (无 GUI 模式，静默极速测试)
    base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[-12.0, 0.0, 1.5]]))
    env = Task4Env(base_env)
    
    # =================================================================
    # 🧪 测试 1：状态空间与动作空间维度验证
    # =================================================================
    print_header("1. 空间维度与类型检查")
    obs, info = env.reset()
    
    print("🎯 动作空间 (Action Space):")
    print(f"   ├─ 类型: {type(env.action_space)}")
    print(f"   ├─ 形状: {env.action_space.shape}")
    print(f"   └─ 范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    assert env.action_space.shape == (4,), "动作空间维度错误！"

    print("\n🧠 状态空间 (Observation Space / Dict):")
    for key, space in env.observation_space.spaces.items():
        print(f"   ├─ [{key}] 形状: {space.shape}, 类型: {space.dtype}")
        assert key in obs, f"初始观测缺失键值: {key}"
        assert obs[key].shape == space.shape, f"观测 {key} 形状不匹配！"
    
    # 检查深度图的具体数值范围
    depth_img = obs['depth_img']
    print(f"\n📸 深度图检查 (当前帧最大值): {np.max(depth_img):.3f}, 最小值: {np.min(depth_img):.3f}")
    assert 0.0 <= np.min(depth_img) and np.max(depth_img) <= 1.0, "深度图未正确归一化！"
    print("✅ 空间维度测试通过！")

    # =================================================================
    # 🧪 测试 2：核心交互步进与起步验证
    # =================================================================
    print_header("2. 核心 step() 与平稳起步测试")
    print("⏳ 执行 10 步悬停指令 (动作 [0,0,0,0])...")
    for i in range(10):
        # 0动作意味着维持悬停 RPM
        next_obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
        assert not terminated and not truncated, "正常悬停不应触发终止！"
        assert 'task4_stats' in info, "Info 字典缺少 task4_stats 组件！"
    
    stats = info['task4_stats']
    print("✅ 交互正常！当前单步返回的 Info 数据：")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"   ├─ {k}: {v:.4f}")
        else:
            print(f"   ├─ {k}: {v}")

    # =================================================================
    # 🧪 测试 3：终局事件与极端物理注入测试
    # =================================================================
    print_header("3. 终局事件 (Terminal Events) 精准注入验证")
    
    # [测试 3.1] 侧翻极刑
    env.reset()
    print("⚠️ 正在注入【姿态侧翻】(Roll > 1.2 rad)...")
    # 强行覆写底层物理姿态
    flip_quat = p.getQuaternionFromEuler([1.5, 0, 0])
    p.resetBasePositionAndOrientation(env.unwrapped.DRONE_IDS[0], env.unwrapped.pos[0], flip_quat, physicsClientId=env.unwrapped.CLIENT)
    _, _, terminated, _, info = env.step(np.array([0,0,0,0]))
    print(f"   └─ 触发结果: Terminated={terminated}, Reason={info['task4_stats']['reason']}")
    assert terminated and info['task4_stats']['reason'] == "CRASH_FLIP_OR_FLOOR", "侧翻极刑未正确触发！"

    # [测试 3.2] 偏离极刑
    env.reset()
    print("⚠️ 正在注入【航线偏离】(Y坐标超界)...")
    deviate_pos = [0, 15.0, 2.0] # 远远偏离 10m 宽的场地
    p.resetBasePositionAndOrientation(env.unwrapped.DRONE_IDS[0], deviate_pos, p.getQuaternionFromEuler([0,0,0]), physicsClientId=env.unwrapped.CLIENT)
    _, _, terminated, _, info = env.step(np.array([0,0,0,0]))
    print(f"   └─ 触发结果: Terminated={terminated}, Reason={info['task4_stats']['reason']}")
    assert terminated and info['task4_stats']['reason'] == "CRASH_WALL_OR_DEVIATE", "偏离极刑未正确触发！"

    # [测试 3.3] 强制通关结算
    env.reset()
    print("🏆 正在注入【完美通关】(瞬移穿越所有门)...")
    num_gates = env.world.cfg.NUM_GATES
    for i in range(num_gates):
        gate = env.gate_poses[i]
        # 瞬移到门所在平面的正前方 0.1 米处 (越过截面)
        pass_pos = gate['pos'] + gate['tangent'] * 0.1
        # 强行给予一个朝向门法向的速度，满足对齐和过门条件
        p.resetBasePositionAndOrientation(env.unwrapped.DRONE_IDS[0], pass_pos, gate['quat'], physicsClientId=env.unwrapped.CLIENT)
        p.resetBaseVelocity(env.unwrapped.DRONE_IDS[0], gate['tangent'] * 5.0, [0,0,0], physicsClientId=env.unwrapped.CLIENT)
        
        _, _, terminated, _, info = env.step(np.array([0.5, 0.5, 0.5, 0.5]))
        reason = info['task4_stats']['reason']
        print(f"   ├─ 穿过门 {i+1}/{num_gates} | 当前进度: {info['task4_stats']['passed_gates']} | 状态: {reason}")
        
    assert terminated and reason == "SUCCESS_ALL_GATES", "通关事件未正确触发！"
    print("✅ 所有极端终局测试完美通过！")

    # =================================================================
    # 🧪 测试 4：随机策略压力测试与 Pandas 数据分析
    # =================================================================
    print_header("4. 执行 5000 步随机压力测试与分布分析")
    env.reset()
    
    collected_stats = []
    total_steps = 20000
    
    start_time = time.time()
    for step in range(total_steps):
        # 产生随机动作 (模拟毫无纪律的初始策略)
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        
        # 记录数值类型的奖励组件
        stats_dict = info['task4_stats']
        numeric_stats = {k: v for k, v in stats_dict.items() if isinstance(v, (int, float))}
        collected_stats.append(numeric_stats)
        
        # 遇死则重生，保持测试不中断
        if terminated or truncated:
            env.reset()
            
    fps = total_steps / (time.time() - start_time)
    print(f"✅ 压测完成！平均仿真速度: {fps:.1f} RL Steps/sec (物理底层约为 {fps * env.cfg.SUBSTEPS:.1f} Hz)")
    
    print_header("📊 奖励组件数值分布体检报告 (Pandas Profiling)")
    # 使用 Pandas 创建 DataFrame 并计算统计指标
    df = pd.DataFrame(collected_stats)
    
    # 提取我们关心的连续奖励和总奖励
    target_cols = ['r_cont_clipped', 'r_track', 'r_align', 'total_reward']
    df_target = df[target_cols]
    
    # 计算均值、方差、最小、25%、中位数、75%、最大值
    summary = df_target.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary['var'] = df_target.var()
    
    # 格式化重排并打印
    output_df = summary[['mean', 'var', 'min', '25%', '50%', '75%', 'max']]
    print(output_df.to_string(float_format=lambda x: f"{x:8.3f}"))
    
    print("\n🧐 质量校验提示:")
    print("  1. r_cont_clipped 的 max 和 min 是否被成功截断在 [-1.0, 1.0]？")
    print("  2. total_reward 的 max 是否预留了单局大奖空间，min 是否保留了 -100 的极刑？")
    print("  3. 观察 variance 判断各种奖励的波动幅度是否合理。")
    print("="*70)

if __name__ == "__main__":
    main()