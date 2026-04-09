import os
import numpy as np
import pybullet as p
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.envs.HoverAviary import HoverAviary

# 请确保你的 task2_env.py 文件和当前脚本在同一目录下
from task2_env import Task2Env, Task2Config

def capture_camera_frame(env, target_pos, distance=7.0):
    """
    抓取 PyBullet 仿真器的当前帧画面
    相机自动对准轨迹中心点，固定俯视45度视角
    """
    client = env.unwrapped.CLIENT
    # 计算相机视角矩阵
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target_pos,
        distance=distance,
        yaw=45,
        pitch=-40,
        roll=0,
        upAxisIndex=2,
        physicsClientId=client
    )
    # 计算投影矩阵
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60,
        aspect=1.0,
        nearVal=0.1,
        farVal=100.0,
        physicsClientId=client
    )
    
    # 硬件OpenGL渲染，保证画质和线条可见性
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=600,
        height=600,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client
    )
    # 丢弃Alpha通道，返回纯RGB数组
    return rgbImg[:, :, :3]

def draw_static_trajectory(env, waypoints, color=[1, 0, 0], width=2.0):
    """
    在 PyBullet 场景中绘制红色目标轨迹
    【关键】GUI模式下，调试线会被相机正常捕获
    """
    client = env.unwrapped.CLIENT
    # 清除之前所有的调试线条/文字，避免画面残留
    p.removeAllUserDebugItems(physicsClientId=client)
    
    # 逐段绘制轨迹线
    for i in range(len(waypoints) - 1):
        p.addUserDebugLine(
            lineFromXYZ=waypoints[i],
            lineToXYZ=waypoints[i+1],
            lineColorRGB=color,
            lineWidth=width,
            physicsClientId=client
        )
    # 起点标记
    p.addUserDebugText(
        text="START",
        textPosition=waypoints[0] + np.array([0, 0, 0.2]),
        textColorRGB=[1, 0, 0],
        textSize=1.5,
        physicsClientId=client
    )
    # 终点标记
    p.addUserDebugText(
        text="END",
        textPosition=waypoints[-1] + np.array([0, 0, 0.2]),
        textColorRGB=[0, 1, 0],
        textSize=1.5,
        physicsClientId=client
    )

def main():
    print("=" * 60)
    print("🎬 [Task 2] 动态轨迹追踪模型可视化与 GIF 录制")
    print("=" * 60)
    
    # ===================== 配置项（可根据你的路径修改）=====================
    MODEL_PATH = "./models/task2/ppo_tracker_final.zip"
    VEC_NORM_PATH = "./models/task2/vec_normalize.pkl"
    OUTPUT_DIR = "./test_gifs"
    # 抽帧间隔：48Hz控制频率，每3步截一帧，最终GIF 16FPS
    CAPTURE_INTERVAL = 3
    GIF_FPS = 16
    # ========================================================================

    # 创建输出文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_NORM_PATH):
        print("❌ 错误：找不到模型文件或归一化文件！")
        print(f"请检查路径：\n模型: {MODEL_PATH}\n归一化文件: {VEC_NORM_PATH}")
        return

    # ===================== 1. 环境初始化（核心修复点）=====================
    # 【关键修复】gui=True 开启渲染，才能让相机捕获到轨迹线
    base_env = HoverAviary(
        gui=True,
        record=False,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]])
    )
    eval_env = Task2Env(base_env)
    client = eval_env.unwrapped.CLIENT

    # 【关键优化】隐藏PyBullet控制面板，只保留纯净3D窗口，不影响截图
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    # 可选优化：开启阴影，提升画面质感
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    # 强制开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client)

    # ===================== 2. 向量化环境与归一化加载 =====================
    vec_env = DummyVecEnv([lambda: eval_env])
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)
    # 测试模式关闭训练和奖励归一化
    vec_env.training = False
    vec_env.norm_reward = False

    # ===================== 3. 模型加载 =====================
    model = PPO.load(MODEL_PATH, device="cpu")
    print("✅ 模型与环境加载成功！准备开始录制...")

    # ===================== 4. 多回合录制 =====================
    total_episodes = 3
    for episode in range(total_episodes):
        print(f"\n🎥 正在录制第 {episode + 1}/{total_episodes} 个轨迹 GIF...")
        obs = vec_env.reset()
        
        # 获取环境实例与轨迹信息
        actual_env = vec_env.envs[0]
        client = actual_env.unwrapped.CLIENT
        waypoints = actual_env.waypoints

        # 轨迹合法性检查
        if len(waypoints) < 2:
            print("⚠️  警告：生成的轨迹点不足2个，跳过本轮")
            continue
        
        # 计算轨迹中心点，用于相机对焦
        center_pos = np.mean(waypoints, axis=0).tolist()
        # 绘制红色目标轨迹
        draw_static_trajectory(actual_env, waypoints, color=[1, 0, 0], width=2.0)

        frames = []
        # 初始化无人机位置，用于绘制蓝色飞行尾迹
        drone_state = actual_env._get_drone_state()
        prev_pos = drone_state[0].tolist()
        episode_done = False

        # 逐步仿真
        for step in range(Task2Config.MAX_STEPS):
            # 确定性预测，输出最优动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            # 获取无人机当前位置，绘制飞行尾迹
            curr_pos = actual_env._get_drone_state()[0].tolist()
            p.addUserDebugLine(
                lineFromXYZ=prev_pos,
                lineToXYZ=curr_pos,
                lineColorRGB=[0, 0.5, 1],  # 亮蓝色飞行轨迹
                lineWidth=3.0,
                physicsClientId=client
            )
            prev_pos = curr_pos

            # 抽帧录制
            if step % CAPTURE_INTERVAL == 0:
                frame = capture_camera_frame(actual_env, target_pos=center_pos)
                frames.append(frame)

            # 回合结束处理
            if done[0]:
                episode_done = True
                final_stats = info[0].get('task2_stats', {})
                success_flag = final_stats.get('r_terminal', 0) > 0
                reason = "✅ 任务成功" if success_flag else "❌ 任务失败"
                completion_rate = final_stats.get('completion_rate', 0.0)
                print(f"  └─> 回合结束 {reason} | 存活步数: {step+1} | 轨迹完成度: {completion_rate:.1f}%")
                # 捕获回合结束的最后一帧
                frame = capture_camera_frame(actual_env, target_pos=center_pos)
                frames.append(frame)
                break

        # 保存GIF
        if len(frames) > 0:
            gif_path = os.path.join(OUTPUT_DIR, f"trajectory_track_ep{episode+1}.gif")
            imageio.mimsave(gif_path, frames, fps=GIF_FPS, loop=0)
            print(f"✅ 第 {episode + 1} 轮 GIF 已保存至: {gif_path}")
        else:
            print("⚠️  本轮无有效帧，跳过GIF保存")

    # 环境关闭与资源释放
    vec_env.close()
    p.disconnect(physicsClientId=client)
    print("\n" + "=" * 60)
    print("🎉 全部录制完成！请到 test_gifs 文件夹查看结果")
    print("=" * 60)

if __name__ == "__main__":
    main()
