import numpy as np
import pandas as pd
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task1_env import Task1Env, Task1Config

def run_sanity_checks():
    print("="*60)
    print("🚀 [Phase 1] 环境接口与空间维度极限测试")
    print("="*60)
    
    # 初始化环境 (关闭GUI加速测试)
    base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    env = Task1Env(base_env)
    
    obs, info = env.reset()
    
    print(f"✅ 状态空间检查 | 期望: (44,), 实际: {obs.shape} | 数据类型: {obs.dtype}")
    assert obs.shape == (44,), "状态空间维度错误！"
    
    print(f"✅ 动作空间检查 | 期望: (4,), 实际: {env.action_space.shape} | 范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    assert env.action_space.shape == (4,), "动作空间维度错误！"
    
    # 单步试运行
    action = env.action_space.sample()
    next_obs, reward, term, trunc, info = env.step(action)
    print(f"✅ Step() 交互检查 | 奖励类型: {type(reward)}, 截断值: {reward:.3f} | Info 字典抓取: 成功")
    assert isinstance(reward, float), "奖励必须是 float 类型"
    assert -1.0 <= reward <= 1.0, "单步奖励未正确截断在 [-1, 1] 之间！"


    print("\n" + "="*60)
    print("💥 [Phase 2] 终局事件 (Terminal Conditions) 强制触发测试")
    print("="*60)
    
    # 1. 测试侧翻坠毁 (Crash)
    env.reset()
    print("[测试] 注入极端扭矩动作 [1, -1, 1, -1] 强制翻车...")
    for i in range(100):
        _, _, term, _, info = env.step(np.array([1.0, -1.0, 1.0, -1.0]))
        if term:
            print(f"✅ 侧翻触发成功! 步数: {i+1} | 判定分值: {info['task1_stats']['r_raw_total']} (期望: {Task1Config.R_CRASH})")
            break

    # 2. 测试偏离坠毁 (Deviation)
    env.reset()
    print("[测试] 注入关停电机动作 [-1, -1, -1, -1] 强制自由落体...")
    for i in range(100):
        _, _, term, _, info = env.step(np.array([-1.0, -1.0, -1.0, -1.0]))
        if term:
            print(f"✅ 偏离触发成功! 步数: {i+1} | 高度: {info['task1_stats']['pos_z']:.2f}m | 判定分值: {info['task1_stats']['r_raw_total']} (期望: {Task1Config.R_DEVIATE})")
            break

    # 3. 测试完成任务 (Success)
    env.reset()
    print("[测试] 篡改稳定计数器强制模拟悬停完成...")
    env.stable_counter = Task1Config.SUCCESS_STEPS_REQ - 1 # 修改计数器到临界值
    # 注入0动作（经过EMA和比例映射，相当于保持当前悬停推力）
    _, _, term, _, info = env.step(np.array([0.0, 0.0, 0.0, 0.0]))
    if term:
        print(f"✅ 完成任务触发成功! 稳定步数: {info['task1_stats']['stable_counter']} | 判定分值: {info['task1_stats']['r_raw_total']} (期望: {Task1Config.R_SUCCESS})")


    print("\n" + "="*60)
    print("🎲 [Phase 3 & 4] 随机策略 5000 步采样与 Pandas 统计分析")
    print("="*60)
    
    env.reset()
    stats_list = []
    
    print("⏳ 正在采集 5000 步随机交互数据，请稍候...")
    for _ in range(5000):
        action = env.action_space.sample()
        _, _, term, trunc, info = env.step(action)
        stats_list.append(info['task1_stats'])
        
        if term or trunc:
            env.reset()
            
    # 使用 Pandas 分析数据
    df = pd.DataFrame(stats_list)
    
    # 提取我们关心的奖励组件列
    reward_cols = ['r_step', 'r_height', 'r_att', 'r_smooth', 'r_raw_total', 'r_clipped']
    df_rewards = df[reward_cols]
    
    # 计算统计特征
    # df.describe() 已经包含了 count, mean, std, min, 25%, 50%, 75%, max
    # 我们额外需要加入方差 (variance)
    desc = df_rewards.describe().T
    desc['var'] = df_rewards.var()
    
    # 重新排列我们要展示的列顺序
    display_cols = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']
    result_df = desc[display_cols]
    
    print("📊 奖励组件统计数据看板 (5000步):")
    print("-" * 75)
    # 格式化输出，保留四位小数
    print(result_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("-" * 75)
    
    print("\n[诊断分析指南]")
    print("1. r_raw_total 的 min 和 max 是否击中了我们设定的极刑 (-10) 和 奖励 (10)？")
    print("2. r_clipped 的 min 和 max 是否严格保持在 [-1.0, 1.0] 内？")
    print("3. r_smooth 的 mean 是否符合预期 (因为是随机动作，这个负值方差应该比较大)？")
    
    env.close()

if __name__ == "__main__":
    run_sanity_checks()