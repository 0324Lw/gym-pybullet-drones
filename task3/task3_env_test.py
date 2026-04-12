import numpy as np
import pandas as pd
import pybullet as p
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from task3_env import Task3Env, Task3Config

def test_curriculum_stages():
    print("=" * 70)
    print("🎓 [Task 3] 课程学习三阶段 - 奖励分布极限压力测试 (5000步/阶段)")
    print("=" * 70)

    # 初始化无头模式环境
    base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 0.3]]))
    env = Task3Env(base_env)

    # 定义三个课程阶段的障碍物配置: (阶段名, 静态数量, 动态数量)
    stages = [
        ("阶段 1: 空旷领空 (起降训练)", 0, 0,10),
        ("阶段 2: 静态密林 (基础避障)", 15, 0,30),
        ("阶段 3: 王牌空战 (动态拦截)", 30, 4,45)
    ]

    # 需要提取的核心奖励与状态字段
    reward_cols = [
        'r_step', 'r_z', 'r_approach', 'r_dir', 'r_smooth', 'r_repulsion','r_att',
        'r_cont_clipped', 'r_terminal', 'r_final_total', 
        'dist_xy', 'pos_z', 'min_lidar'
    ]
    display_cols = ['mean', 'var', 'min', '25%', '50%', '75%', 'max']

    for stage_idx, (stage_name, num_static, num_dynamic,num_t) in enumerate(stages):
        print(f"\n\n🚀 正在切换至 【{stage_name}】...")
        print(f"   └─> 注入障碍物: 静态 {num_static} 个, 动态 {num_dynamic} 个")
        
        # 调用我们在 env 中预留的修改接口
        env.set_curriculum(num_static, num_dynamic,num_t)
        env.reset()
        
        stats_list = []
        death_reasons = []
        
        # 采集 5000 步
        for _ in range(5000):
            action = env.action_space.sample()
            _, _, term, trunc, info = env.step(action)
            
            stats = info['task3_stats']
            stats_list.append(stats)
            
            if term or trunc:
                death_reasons.append(stats['reason'])
                env.reset()

        # Pandas 数据分析
        df = pd.DataFrame(stats_list)
        df_rewards = df[reward_cols]
        
        desc = df_rewards.describe().T
        desc['var'] = df_rewards.var()
        result_df = desc[display_cols]
        
        # 统计死因
        reason_counts = pd.Series(death_reasons).value_counts(normalize=True) * 100
        reason_str = ", ".join([f"{k}: {v:.1f}%" for k, v in reason_counts.items()])

        print("-" * 85)
        print(result_df.to_string(float_format=lambda x: f"{x:.4f}"))
        print("-" * 85)
        print(f"💀 [死因分布]: {reason_str}")

    env.close()
    print("\n✅ 所有阶段体检完毕！")

if __name__ == "__main__":
    test_curriculum_stages()