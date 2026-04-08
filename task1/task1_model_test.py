import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task1_env import Task1Env, Task1Plot, Task1Config

def main():
    print("=" * 60)
    print("🚁 [测试] 高保真无人机 PPO 模型 10 秒连续悬停测试")
    print("=" * 60)
    
    MODEL_DIR = "./models/task1"
    MODEL_PATH = os.path.join(MODEL_DIR, "ppo_hover_final.zip")
    VEC_NORM_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    
    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_NORM_PATH):
        raise FileNotFoundError(f"找不到模型或归一化文件，请检查 {MODEL_DIR} 目录。")

    # ==========================================
    # 👑 核心覆写：为 10 秒连续测试解除封印
    # ==========================================
    # 10秒对应的物理步数 (通常 HoverAviary 的底层控制频率 CTRL_FREQ 为 240Hz)
    # 10s * 240Hz = 2400 steps
    TEST_DURATION_SEC = 10
    TEST_STEPS = TEST_DURATION_SEC * 240 
    
    # 动态修改 Config，防止提前截断或因为“太成功”而被强行重置
    Task1Config.MAX_STEPS = TEST_STEPS + 100 
    Task1Config.SUCCESS_STEPS_REQ = TEST_STEPS + 100

    print("[INFO] 正在加载物理仿真环境 (GUI模式)...")
    base_env = HoverAviary(gui=True, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    env = Task1Env(base_env)
    
    # 包装为 DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    print("[INFO] 正在加载状态归一化器 (VecNormalize)...")
    # 加载训练时的归一化参数，并严禁测试时更新均值
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    print("[INFO] 正在加载 PPO 模型权重...")
    model = PPO.load(MODEL_PATH, env=vec_env)

    obs = vec_env.reset()
    infos_list = []
    
    print(f"\n🚀 起飞！开始进行 {TEST_DURATION_SEC} 秒实机演示...")
    
    # 记录物理引擎的实际频率，用于控制视觉渲染的休眠时间
    # 保证我们在屏幕上看到的 10 秒，就是物理流逝的 10 秒
    ctrl_freq = base_env.CTRL_FREQ
    sleep_time = 1.0 / ctrl_freq

    for step in range(TEST_STEPS):
        # deterministic=True 意味着关闭网络的高斯噪声，输出策略的绝对均值（最优解）
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = vec_env.step(action)
        
        # 抓取单机环境真实的 info 字典
        real_info = info[0]['task1_stats']
        infos_list.append(real_info)
        
        # 实时打印高度，每隔大概一秒打印一次
        if step % ctrl_freq == 0:
            current_sec = step // ctrl_freq
            pos_z = real_info['pos_z']
            print(f"⏱️ 飞行时间: {current_sec:02d}s / {TEST_DURATION_SEC}s | 当前高度: {pos_z:.4f}m")
            
        # 视觉同步延时
        time.sleep(sleep_time)
        
        # 如果意外炸机（跌落或侧翻），及时结束
        if done[0]:
            print("\n💥 [警告] 无人机发生严重失控或坠毁，测试提前终止！")
            break

    print(f"\n✅ {TEST_DURATION_SEC} 秒演示结束，正在生成性能分析图表...")
    vec_env.close()
    
    # 绘制这 10 秒的高清微观性能图
    plot_path = os.path.join(MODEL_DIR, "eval_10sec_performance.png")
    Task1Plot.plot_episode_stats(infos_list, save_path=plot_path)

if __name__ == "__main__":
    main()