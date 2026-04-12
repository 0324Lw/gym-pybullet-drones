import os
import torch
import numpy as np
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

from task3_env import Task3Env, Task3Config

# ==========================================
# 配置参数
# ==========================================
MODEL_PATH = "./models/task3/ppo_obstacle_final.zip"
VEC_NORM_PATH = "./models/task3/vec_normalize.pkl"
OUTPUT_DIR = "./models/task3/gifs"

TARGET_SUCCESS_EPISODES = 1   # 保存多少个成功回合的 GIF
FRAME_SKIP = 3                # 每隔 3 个物理步截取一帧 (减少内存占用)
GIF_FPS = 10                  # GIF 播放帧率 (10 FPS 配合抽帧，实现慢动作回放)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 辅助渲染函数
# ==========================================
def get_3d_camera_frame(raw_env, width=400, height=400):
    """获取 PyBullet 3D 摄像机跟随视角的画面"""
    client = raw_env.unwrapped.CLIENT
    pos = raw_env.unwrapped.pos[0]
    
    # 第三人称跟随视角：位于无人机后方 4m，斜向上俯视
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=pos,
        distance=4.0,
        yaw=45,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=float(width)/height, nearVal=0.1, farVal=100.0, physicsClientId=client
    )
    
    img_arr = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client
    )
    # img_arr[2] 是 RGBA 图像，取出 RGB 通道
    rgb = img_arr[2][:, :, :3]
    return rgb

def get_current_obstacles(raw_env):
    """实时获取所有障碍物的最新位置（支持动态障碍物）"""
    obs_list = []
    client = raw_env.unwrapped.CLIENT
    
    # 静态障碍物 (灰色)
    for i, uid in enumerate(raw_env.world.static_obs_ids):
        pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=client)
        r = raw_env.world._circles_record[i][2] # 从记录中提取半径
        obs_list.append((pos[0], pos[1], r, 'gray'))
        
    # 动态障碍物 (红色)
    offset = len(raw_env.world.static_obs_ids)
    for i, uid in enumerate(raw_env.world.dynamic_obs_ids):
        pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=client)
        r = raw_env.world._circles_record[offset + i][2]
        obs_list.append((pos[0], pos[1], r, 'red'))
        
    return obs_list

def create_2d_plot_frame(trajectory, obstacles, start_pos, goal_pos, current_pos):
    """使用 Matplotlib 绘制 2D 俯视动态轨迹图"""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f4f8') # 浅色背景

    # 绘制障碍物
    for (x, y, r, color) in obstacles:
        circle = plt.Circle((x, y), r, color=color, alpha=0.7)
        ax.add_patch(circle)

    # 绘制起终点
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label='Goal')

    # 绘制历史轨迹
    if len(trajectory) > 0:
        traj_arr = np.array(trajectory)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, alpha=0.6, label='Trajectory')

    # 绘制无人机当前位置
    ax.plot(current_pos[0], current_pos[1], 'bo', markersize=8)

    plt.legend(loc='upper right')
    plt.title("2D Navigation & Obstacle Avoidance", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 将 Matplotlib 图像转换为 numpy 数组
    fig.canvas.draw()
    # 直接提取 RGBA 数组并切片取前 3 个通道 (RGB)
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig) # 防止内存泄漏
    
    return img

# ==========================================
# 主测试流程
# ==========================================
def main():
    print("=" * 60)
    print("🎬 [Task 3] 模型渲染与测试启动 (仅捕获成功回合)")
    print("=" * 60)

    # 1. 初始化测试环境 (加载最难的阶段 3：30静态+4动态，距离40m)
    eval_base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[0.0, 0.0, 1.0]]))
    eval_env = Task3Env(eval_base_env)
    
    # 包装为 SB3 兼容的向量化环境
    vec_env = DummyVecEnv([lambda: eval_env])
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)
    vec_env.training = False     # 关闭测试时的均值更新
    vec_env.norm_reward = False  # 关闭奖励归一化
    
    # 强制设置为阶段 3 难度
    vec_env.env_method("set_curriculum", num_static=30, num_dynamic=4, max_sg_dist=40.0)
    raw_env = vec_env.envs[0]    # 获取底层环境引用，用于提取绘图数据

    # 2. 加载模型
    model = PPO.load(MODEL_PATH, env=vec_env, device="cpu")
    print("✅ 模型与归一化字典加载成功！准备寻找成功的回合...")

    success_count = 0
    attempt_count = 0

    while success_count < TARGET_SUCCESS_EPISODES:
        attempt_count += 1
        obs = vec_env.reset()
        
        frames_3d = []
        frames_2d = []
        trajectory = []
        
        start_pos = raw_env.world.start_pos.copy()
        goal_pos = raw_env.world.goal_pos.copy()
        
        done = False
        step_idx = 0
        final_reason = "UNKNOWN"

        print(f"\n▶️ 开始尝试第 {attempt_count} 个回合...")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_list = vec_env.step(action)
            
            done = done_arr[0]
            info = info_list[0]
            
            current_pos = raw_env.unwrapped.pos[0].copy()
            trajectory.append((current_pos[0], current_pos[1]))
            
            # 抽帧记录画面
            if step_idx % FRAME_SKIP == 0:
                # 获取 3D 画面
                img_3d = get_3d_camera_frame(raw_env)
                frames_3d.append(img_3d)
                
                # 获取最新的障碍物位置并绘制 2D 画面
                current_obs_data = get_current_obstacles(raw_env)
                img_2d = create_2d_plot_frame(trajectory, current_obs_data, start_pos, goal_pos, current_pos)
                frames_2d.append(img_2d)
                
            if done:
                final_reason = info.get('task3_stats', {}).get('reason', 'UNKNOWN')
                break
                
            step_idx += 1

        # 3. 判定回合结果
        if final_reason == "SUCCESS":
            success_count += 1
            print(f"🎉 回合 {attempt_count} 挑战成功！存活步数: {step_idx}。正在生成并保存 GIF...")
            
            # 保存 3D GIF
            path_3d = os.path.join(OUTPUT_DIR, f"success_{success_count}_3D_view.gif")
            imageio.mimsave(path_3d, frames_3d, fps=GIF_FPS)
            
            # 保存 2D GIF
            path_2d = os.path.join(OUTPUT_DIR, f"success_{success_count}_2D_radar.gif")
            imageio.mimsave(path_2d, frames_2d, fps=GIF_FPS)
            
            print(f"💾 已保存:\n  ├─ {path_3d}\n  └─ {path_2d}")
        else:
            print(f"💀 回合 {attempt_count} 挑战失败 (死因: {final_reason})。数据已丢弃，继续寻找...")

    print("\n✅ 所有测试任务圆满完成！")
    vec_env.close()

if __name__ == "__main__":
    main()