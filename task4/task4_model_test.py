import os
import torch
import numpy as np
import imageio
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from task4_env import Task4Env

# ==========================================
# 配置参数
# ==========================================
MODEL_PATH = "./models/task4/ppo_vision_34000000_steps.zip" # 请确保路径指向你最新的模型
VEC_NORM_PATH = "./models/task4/vec_normalize_34000000_steps.pkl"
OUTPUT_DIR = "./models/task4/gifs"

TEST_EPISODES = 5
FRAME_SKIP = 2     # 每 2 个物理步截取一帧
GIF_FPS = 24       # GIF 播放帧率

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 3D 摄像机渲染函数
# ==========================================
def get_3d_camera_frame(raw_env, width=400, height=400):
    client = raw_env.unwrapped.CLIENT
    pos = raw_env.unwrapped.pos[0]
    quat = raw_env.unwrapped.quat[0]
    
    # 获取机头朝向
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    forward_vec = rot_mat[:, 0]
    
    # 摄像机放在无人机后方 2m，斜上方 1m，看向机头前方
    cam_eye = pos - forward_vec * 2.0 + np.array([0, 0, 1.0])
    cam_target = pos + forward_vec * 1.0
    
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye,
        cameraTargetPosition=cam_target,
        cameraUpVector=[0, 0, 1],
        physicsClientId=client
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=75, aspect=float(width)/height, nearVal=0.1, farVal=100.0, physicsClientId=client
    )
    
    img_arr = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client
    )
    rgb = img_arr[2][:, :, :3]
    return rgb

# ==========================================
# 主测试流程
# ==========================================
def main():
    print("=" * 60)
    print("🎬 [Task 4] 模型渲染与行为诊断测试启动")
    print("=" * 60)

    # 1. 初始化环境
    eval_base_env = HoverAviary(gui=False, record=False, initial_xyzs=np.array([[-12.0, 0.0, 1.5]]))
    eval_env = Task4Env(eval_base_env)
    
    vec_env = DummyVecEnv([lambda: eval_env])
    vec_env = VecNormalize.load(VEC_NORM_PATH, vec_env)
    vec_env.training = False     # 关闭均值更新
    vec_env.norm_reward = False  # 关闭奖励归一化
    
    raw_env = vec_env.envs[0]

    # 2. 加载模型
    try:
        # 注意：需要传入 custom_objects 因为我们用了自定义的网络结构
        from task4_train import AsymmetricFeaturesExtractor, AsymmetricPolicy
        custom_objects = {
            "features_extractor_class": AsymmetricFeaturesExtractor,
            "policy_class": AsymmetricPolicy
        }
        model = PPO.load(MODEL_PATH, env=vec_env, device="cpu", custom_objects=custom_objects)
        print("✅ 模型与归一化字典加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径。错误信息: {e}")
        return

    # 3. 运行测试
    for ep in range(1, TEST_EPISODES + 1):
        obs = vec_env.reset()
        frames = []
        done = False
        step_idx = 0
        
        ep_track_reward = 0.0
        ep_align_reward = 0.0
        
        print(f"\n▶️ 开始录制回合 {ep}...")

        while not done:
            # 采用确定性策略 (deterministic=True) 观察网络最真实的意图
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info_list = vec_env.step(action)
            
            done = done_arr[0]
            info = info_list[0]
            stats = info.get('task4_stats', {})
            
            ep_track_reward += stats.get('r_track', 0)
            ep_align_reward += stats.get('r_align', 0)
            
            # 每 10 步打印一次调试信息，洞察内部奖励分配
            if step_idx % 10 == 0:
                print(f"   [Step {step_idx:03d}] Track奖: {stats.get('r_track',0):.3f} | Align奖: {stats.get('r_align',0):.3f} | 动作均值: {np.mean(action):.2f}")
            
            if step_idx % FRAME_SKIP == 0:
                frames.append(get_3d_camera_frame(raw_env))
                
            step_idx += 1
            
            if done:
                reason = stats.get('reason', 'UNKNOWN')
                passed_gates = stats.get('passed_gates', 0)
                print(f"💀 回合结束！存活: {step_idx}步 | 死因: {reason} | 穿门: {passed_gates}")
                print(f"📊 本局累计 -> 跟踪奖励: {ep_track_reward:.2f} | 对齐奖励: {ep_align_reward:.2f}")

        # 保存 GIF
        gif_path = os.path.join(OUTPUT_DIR, f"task4_diagnosis_ep{ep}.gif")
        imageio.mimsave(gif_path, frames, fps=GIF_FPS)
        print(f"💾 GIF 已保存至: {gif_path}")

    print("\n✅ 所有诊断测试完成！请前往文件夹查看生成的 GIF。")

if __name__ == "__main__":
    main()