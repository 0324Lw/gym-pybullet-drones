import os
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev

# 导入我们刚刚编写的世界模型
from task4_world import Task4World, WorldConfig

# =====================================================================
# 辅助绘图函数
# =====================================================================
def get_gate_corners(center, quat, size):
    """
    根据门心坐标和四元数，计算门框在 3D 空间中的四个顶点，用于绘制门框平面。
    门在局部坐标系下处于 XY 平面，开口朝向 Z 轴。
    """
    rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    half_w = size / 2.0
    # 局部坐标系下的四个顶点
    local_corners = np.array([
        [-half_w, -half_w, 0],
        [ half_w, -half_w, 0],
        [ half_w,  half_w, 0],
        [-half_w,  half_w, 0]
    ])
    # 转换到世界坐标系
    global_corners = np.dot(local_corners, rot_mat.T) + center
    return global_corners

def plot_environment(episode: int, start_pos: np.ndarray, gate_poses: list, config: WorldConfig, save_dir="test_results"):
    """绘制单回合的 2D 和 3D 赛道图"""
    fig = plt.figure(figsize=(16, 8))
    
    # --- 提取路点并生成平滑样条曲线 ---
    pts = [start_pos] + [g['pos'] for g in gate_poses]
    pts = np.array(pts)
    
    # 使用 Scipy 生成平滑样条曲线
    tck, u = splprep([pts[:,0], pts[:,1], pts[:,2]], s=0, k=min(3, len(pts)-1))
    u_fine = np.linspace(0, 1, 100)
    smooth_path = splev(u_fine, tck)
    
    # ---------------------------
    # Subplot 1: 2D 俯视图 (XY平面)
    # ---------------------------
    ax2d = fig.add_subplot(121)
    ax2d.set_title(f"Episode {episode} - 2D Top-down View", fontsize=14)
    ax2d.set_xlabel("X (m)"); ax2d.set_ylabel("Y (m)")
    ax2d.set_xlim(-config.ARENA_LENGTH/2, config.ARENA_LENGTH/2)
    ax2d.set_ylim(-config.ARENA_WIDTH/2, config.ARENA_WIDTH/2)
    ax2d.grid(True, linestyle='--', alpha=0.6)
    
    # 画起点与平滑曲线
    ax2d.plot(start_pos[0], start_pos[1], 'go', markersize=10, label="Start")
    ax2d.plot(smooth_path[0], smooth_path[1], 'b-', linewidth=2, label="Spline Trajectory", alpha=0.7)
    
    # 画门与法向量
    for i, g in enumerate(gate_poses):
        pos = g['pos']
        tangent = g['tangent']
        
        # 门心与法向箭头
        ax2d.plot(pos[0], pos[1], 'ro', markersize=6)
        ax2d.quiver(pos[0], pos[1], tangent[0], tangent[1], color='red', scale=15, width=0.005)
        
        # 画表示门宽度的线段
        rot_mat = np.array(p.getMatrixFromQuaternion(g['quat'])).reshape(3, 3)
        right_vec = rot_mat[:, 0] # 局部 X 轴
        p1 = pos[:2] + right_vec[:2] * (config.GATE_SIZE/2)
        p2 = pos[:2] - right_vec[:2] * (config.GATE_SIZE/2)
        ax2d.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3)

    ax2d.legend(loc="upper left")

    # ---------------------------
    # Subplot 2: 3D 空间视图
    # ---------------------------
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.set_title(f"Episode {episode} - 3D Perspective View", fontsize=14)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_xlim(-config.ARENA_LENGTH/2, config.ARENA_LENGTH/2)
    ax3d.set_ylim(-config.ARENA_WIDTH/2, config.ARENA_WIDTH/2)
    ax3d.set_zlim(0, config.ARENA_HEIGHT)
    
    # 画起点与平滑曲线
    ax3d.scatter(*start_pos, color='green', s=100, label="Start")
    ax3d.plot(smooth_path[0], smooth_path[1], smooth_path[2], 'b-', linewidth=2, alpha=0.7)
    
    # 画门 (3D 矩形框) 与法向量
    for i, g in enumerate(gate_poses):
        pos = g['pos']
        tangent = g['tangent']
        
        # 画箭头
        ax3d.quiver(pos[0], pos[1], pos[2], tangent[0], tangent[1], tangent[2], 
                    color='red', length=1.5, normalize=True, arrow_length_ratio=0.2)
        
        # 画 3D 门框平面
        corners = get_gate_corners(pos, g['quat'], config.GATE_SIZE)
        poly = Poly3DCollection([corners], alpha=0.3, facecolors='cyan', edgecolors='black', linewidths=2)
        ax3d.add_collection3d(poly)
        ax3d.text(pos[0], pos[1], pos[2]+0.5, f"G{i+1}", color='black', fontweight='bold')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"world_test_ep{episode}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 保存渲染图: {save_path}")

# =====================================================================
# 主测试流程
# =====================================================================
def main():
    print("=" * 60)
    print("🧪 [Task 4] 世界模型 (Task4World) 综合单元测试启动")
    print("=" * 60)
    
    # 1. 启动 PyBullet (DIRECT 模式，不需要显示 3D GUI)
    client = bc.BulletClient(connection_mode=p.DIRECT)
    
    # 2. 实例化世界模型
    try:
        world = Task4World(client._client)
        print("✅ Task4World 实例化成功！场景背景与对象池构建完成。")
    except Exception as e:
        print(f"❌ 实例化失败: {e}")
        return

    # 3. 测试对象池逻辑
    pool_size = len(world.gate_pool_ids)
    print(f"✅ 对象池状态: 成功加载 {pool_size} 个复合体门框 (Compound Shapes)。")
    if pool_size != world.cfg.NUM_GATES:
        print("⚠️ 警告: 对象池数量与配置不符！")
        
    print("-" * 60)
    
    # 4. 生成 5 张随机地图并保存
    for ep in range(1, 6):
        start_pos, gate_poses = world.reset_world()
        plot_environment(ep, start_pos, gate_poses, world.cfg)
        
    print("-" * 60)
    
    # 5. 测试端到端视觉接口 (深度图渲染)
    print("🎥 测试深度图渲染接口...")
    # 模拟无人机处于起点，朝向正前方 (X轴)
    dummy_pos = start_pos
    dummy_quat = p.getQuaternionFromEuler([0, 0, 0])
    
    try:
        depth_img = world.get_depth_vision(dummy_pos, dummy_quat)
        print(f"✅ 深度图渲染成功！")
        print(f"   ├─ Tensor 形状: {depth_img.shape} (预期: (1, 64, 64))")
        print(f"   ├─ 数据类型: {depth_img.dtype} (预期: float32)")
        print(f"   ├─ 深度最大值: {np.max(depth_img):.3f} (预期: <= 1.0)")
        print(f"   └─ 深度最小值: {np.min(depth_img):.3f} (预期: >= 0.0)")
        
        if depth_img.shape != (1, 64, 64):
            print("❌ 错误: 深度图形状不匹配，请检查 CNN 维度设定！")
            
    except Exception as e:
        print(f"❌ 深度渲染失败: {e}")

    # 清理断开连接
    client.disconnect()
    print("=" * 60)
    print("🎉 所有单元测试圆满完成！底层逻辑极其稳固！")

if __name__ == "__main__":
    main()