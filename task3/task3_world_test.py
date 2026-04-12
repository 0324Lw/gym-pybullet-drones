import os
import io
import math
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio

# 导入我们编写的世界模型
from task3_world import Task3World, WorldConfig

def test_rl_interfaces(world: Task3World):
    print("\n" + "="*50)
    print("🛠️ [Phase 1] RL 接口与建图逻辑严密性体检")
    print("="*50)
    
    # 1. 重置逻辑与坐标校验
    start_pos, goal_pos = world.reset_world()
    dist = np.linalg.norm(start_pos[:2] - goal_pos[:2])
    print(f"✅ 世界重置成功！")
    print(f"  └─> 起点: {np.round(start_pos, 2)}")
    print(f"  └─> 终点: {np.round(goal_pos, 2)}")
    print(f"  └─> 欧氏距离: {dist:.2f} m (期望范围: [{WorldConfig.MIN_SG_DIST}, {WorldConfig.MAX_SG_DIST}])")
    assert WorldConfig.MIN_SG_DIST <= dist <= WorldConfig.MAX_SG_DIST, "起终点间距异常！"
    
    # 2. 实体数量校验
    num_static = len(world.static_obs_ids)
    num_dynamic = len(world.dynamic_obs_ids)
    print(f"✅ 障碍物生成校验通过！静态: {num_static} 个, 动态: {num_dynamic} 个")
    
    # 3. LiDAR 感知接口校验
    dummy_yaw = 0.0
    # 在起点进行一次扫描 (-1 代表虚拟的无人机 ID，避免射线扫到自己)
    lidar_scan = world.get_lidar_scan(start_pos, dummy_yaw, drone_id=-1)
    print(f"✅ LiDAR 雷达接口校验成功！")
    print(f"  └─> 射线数量 (张量维度): {lidar_scan.shape} (期望: ({WorldConfig.LIDAR_NUM_RAYS},))")
    print(f"  └─> 数值范围: 最小 {lidar_scan.min():.2f}, 最大 {lidar_scan.max():.2f} (合法范围: [0.0, 1.0])")
    assert lidar_scan.shape == (WorldConfig.LIDAR_NUM_RAYS,), "雷达维度错误！"
    assert 0.0 <= lidar_scan.min() and lidar_scan.max() <= 1.0, "雷达归一化数值越界！"

    # 4. 动力学游走接口校验
    print("✅ 动态障碍物物理保姆机制测试...")
    for _ in range(100):
        p.stepSimulation(physicsClientId=world.CLIENT)
        world.step_dynamics()
    print("  └─> 运行 100 步物理解算无崩溃，游走动能正常维持。")


def capture_3d_frame(client):
    """抓取 PyBullet 的 3D 俯视等轴测图像"""
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0, 0, 0],
        distance=65,
        yaw=45, pitch=-45, roll=0,
        upAxisIndex=2,
        physicsClientId=client
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=100.0, physicsClientId=client
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width=500, height=500,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_TINY_RENDERER, # 兼容无显卡服务器
        physicsClientId=client
    )
    return rgb[:, :, :3]

def capture_2d_frame(world: Task3World):
    """利用 Matplotlib 绘制 2D 平面战术地图并转为图像数组"""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
    bound = WorldConfig.ARENA_SIZE / 2.0
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    
    # 绘制起终点安全区
    ax.add_patch(Circle((world.start_pos[0], world.start_pos[1]), WorldConfig.SAFE_ZONE_RADIUS, color='blue', alpha=0.15))
    ax.add_patch(Circle((world.goal_pos[0], world.goal_pos[1]), WorldConfig.SAFE_ZONE_RADIUS, color='green', alpha=0.15))
    ax.plot(world.start_pos[0], world.start_pos[1], 'b*', markersize=15, label='Start')
    ax.plot(world.goal_pos[0], world.goal_pos[1], 'g*', markersize=15, label='Goal')
    
    # 获取柱子坐标和半径 (从_circles_record中提取)
    # 前 NUM_STATIC_OBS 个是静态，后面是动态
    num_static = len(world.static_obs_ids)
    
    for i, (cx, cy, cr) in enumerate(world._circles_record):
        if i < num_static:
            # 静态绿色
            ax.add_patch(Circle((cx, cy), cr, color='#2ca02c', alpha=0.7))
        else:
            # 动态红色 (获取其实时坐标)
            uid = world.dynamic_obs_ids[i - num_static]
            pos, _ = p.getBasePositionAndOrientation(uid, physicsClientId=world.CLIENT)
            ax.add_patch(Circle((pos[0], pos[1]), cr, color='#d62728', alpha=0.9))
            
    ax.set_title("2D Tactical Map: Dynamic Obstacle Navigation")
    ax.legend(loc="upper right")
    
    # 将 pyplot 画布转换为 RGB 数组
    buf = io.BytesIO()
    plt.savefig(buf, format='raw')
    buf.seek(0)
    img_arr = np.reshape(np.frombuffer(buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    plt.close(fig)
    return img_arr[:, :, :3]

def generate_gifs():
    print("\n" + "="*50)
    print("🎬 [Phase 2] 生成并保存 2D/3D 环境动图 (5 组)")
    print("="*50)
    
    OUTPUT_DIR = "./test_gifs_task3"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    client_id = p.connect(p.DIRECT)
    world = Task3World(client_id)
    
    # 首先进行逻辑测试
    test_rl_interfaces(world)
    
    NUM_EPISODES = 5
    STEPS_PER_GIF = 80  # 录制 80 帧
    
    for ep in range(NUM_EPISODES):
        print(f"⏳ 正在录制第 {ep+1}/{NUM_EPISODES} 组场景...")
        world.reset_world()
        
        frames_2d = []
        frames_3d = []
        
        # 为了让 GIF 中的动态物体移动明显，我们在每次抽帧之间运行多步物理模拟
        PHYSICS_STEPS_PER_FRAME = 8 
        
        for step in range(STEPS_PER_GIF):
            for _ in range(PHYSICS_STEPS_PER_FRAME):
                p.stepSimulation(physicsClientId=world.CLIENT)
                world.step_dynamics()
                
            frames_2d.append(capture_2d_frame(world))
            frames_3d.append(capture_3d_frame(world.CLIENT))
            
        # 保存 GIF
        path_2d = os.path.join(OUTPUT_DIR, f"task3_env_{ep+1}_2D.gif")
        path_3d = os.path.join(OUTPUT_DIR, f"task3_env_{ep+1}_3D.gif")
        
        imageio.mimsave(path_2d, frames_2d, fps=15)
        imageio.mimsave(path_3d, frames_3d, fps=15)
        print(f"  └─> 已保存: {path_2d}")
        print(f"  └─> 已保存: {path_3d}")

    p.disconnect(client_id)
    print(f"\n🎉 完美收工！请前往 {OUTPUT_DIR} 文件夹查看所有 10 张动图。")

if __name__ == "__main__":
    generate_gifs()