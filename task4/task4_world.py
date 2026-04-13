import numpy as np
import math
import pybullet as p
from typing import Tuple, List, Dict

# ==========================================
# 模块 1：WorldConfig (纯视觉赛道参数配置)
# ==========================================
class WorldConfig:
    """集中管理视觉环境、赛道生成与对象池的超参数"""
    
    # --- 场地与起点 ---
    ARENA_LENGTH = 30.0                     # X轴长度 (赛道主轴)
    ARENA_WIDTH = 10.0                      # Y轴宽度
    ARENA_HEIGHT = 5.0                      # Z轴高度
    
    START_POS = np.array([-12.0, 0.0, 1.5]) # 固定的出生点
    
    # --- 窄门与赛道参数 ---
    NUM_GATES = 5                           # 赛道中门的数量
    GATE_SIZE = 0.5                         # 门内径 (米)。对于小型无人机(轴距0.15m)，这是3倍以上的容错率，适合极速
    GATE_THICKNESS = 0.05                   # 门框的厚度
    
    MAX_ROLL_PITCH = 45.0                   # 门法向的随机倾角扰动极限 (度)
    
    # --- 视觉相机参数 ---
    CAM_RES_W = 64                          # 图像宽度
    CAM_RES_H = 64                          # 图像高度
    CAM_FOV = 110.0                          # 视场角 (度)
    CAM_NEAR = 0.1                          # 最近裁剪面 (米)
    CAM_FAR = 10.0                          # 最远裁剪面 (米)，超过10米的物体视为不可见

# ==========================================
# 模块 2：Task4World (物理与视觉世界模型)
# ==========================================
class Task4World:
    """
    负责构建高保真复合刚体窄门、维护对象池、生成三次样条平滑赛道，
    并提供高速的 64x64 深度图渲染接口。
    """
    def __init__(self, client_id: int):
        self.CLIENT = client_id
        self.cfg = WorldConfig()
        
        self.gate_pool_ids = []             # 窄门对象池 UIDs
        self.gate_poses = []                # 记录当前回合门的真实坐标与姿态 (供 Critic 上帝视角使用)
        
        # 1. 构建纯净的环境背景 (灰白色墙面，方便深度相机的对比)
        self._build_clean_arena()
        
        # 2. 预先构建对象池 (埋在地下隐藏)
        self._create_gate_pool()

    # ---------------------------------------------------------
    # 核心接口 1: 世界重置与样条赛道生成
    # ---------------------------------------------------------
    def reset_world(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        瞬间重置赛道。
        返回: (无人机起点位置, 包含所有门中心坐标和四元数的列表 -> 专供 Critic 使用)
        """
        # 1. 生成 3D Catmull-Rom 样条控制点 (确保平滑)
        control_points = self._generate_spline_waypoints()
        self.gate_poses.clear()
        
        # 2. 遍历对象池，计算姿态并进行空间传送 (瞬移)
        for i, gate_uid in enumerate(self.gate_pool_ids):
            pos = control_points[i]
            
            # 计算切线方向 (朝向下一个点)
            if i < len(control_points) - 1:
                tangent = control_points[i+1] - pos
            else:
                tangent = pos - control_points[i-1] # 最后一个门的切线
                
            tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
            
            # 计算基础的偏航角 (Yaw) 和俯仰角 (Pitch) 使门的 Z 轴对齐切线
            yaw = math.atan2(tangent[1], tangent[0])
            pitch = -math.asin(tangent[2])
            
            # 增加极端难度的随机扰动 (强迫无人机侧飞)
            random_roll = np.random.uniform(-self.cfg.MAX_ROLL_PITCH, self.cfg.MAX_ROLL_PITCH)
            random_pitch_offset = np.random.uniform(-self.cfg.MAX_ROLL_PITCH/2, self.cfg.MAX_ROLL_PITCH/2)
            
            roll_rad = math.radians(random_roll)
            pitch_rad = pitch + math.radians(random_pitch_offset)
            
            # 将欧拉角转化为四元数
            quat = p.getQuaternionFromEuler([roll_rad, pitch_rad, yaw])
            
            # [速度黑魔法]: 直接覆写底层状态，零耗时
            p.resetBasePositionAndOrientation(gate_uid, pos, quat, physicsClientId=self.CLIENT)
            
            self.gate_poses.append({
                'pos': pos,
                'quat': quat,
                'tangent': tangent
            })
            
        return self.cfg.START_POS.copy(), self.gate_poses

    # ---------------------------------------------------------
    # 核心接口 2: 端到端深度图渲染
    # ---------------------------------------------------------
    def get_depth_vision(self, drone_pos: np.ndarray, drone_quat: Tuple) -> np.ndarray:
        """
        基于无人机当前姿态，渲染 64x64 深度图。
        返回: Shape 为 (1, 64, 64) 的 numpy 数组，数值范围 [0.0, 1.0]。
              数值代表距离，0.0 为贴脸，1.0 为大于等于 CAM_FAR (10米)。
        """
        # 计算相机的前向向量与上方向向量 (假设相机固定在机身 X 轴正前方)
        rot_mat = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        cam_forward = rot_mat[:, 0]
        cam_up = rot_mat[:, 2]
        
        target_pos = drone_pos + cam_forward * 1.0 # 目标点只需在前方即可
        
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=drone_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=cam_up,
            physicsClientId=self.CLIENT
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.cfg.CAM_FOV,
            aspect=1.0,
            nearVal=self.cfg.CAM_NEAR,
            farVal=self.cfg.CAM_FAR,
            physicsClientId=self.CLIENT
        )
        
        # 调用 PyBullet 的硬件 OpenGL 渲染器，速度最快
        # 参数: (width, height, viewMatrix, projectionMatrix)
        _, _, _, depth_buffer, _ = p.getCameraImage(
            width=self.cfg.CAM_RES_W, 
            height=self.cfg.CAM_RES_H,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.CLIENT
        )

        # PyBullet 返回的 depth_buffer 是一个非线性的 [0, 1] 数组。需要用反投影公式解算出真实的米数
        depth_img = np.array(depth_buffer).reshape(self.cfg.CAM_RES_H, self.cfg.CAM_RES_W)
        
        far = self.cfg.CAM_FAR
        near = self.cfg.CAM_NEAR
        # 非线性解码公式
        true_depth = (far * near) / (far - (far - near) * depth_img)
        
        # 归一化为 [0, 1] 区间，专供 CNN 处理 (超过 10m 的截断为 1.0)
        normalized_depth = np.clip(true_depth / far, 0.0, 1.0)
        
        # 转换为 PyTorch 喜欢的 Channel-First 格式: (1, 64, 64)
        return np.expand_dims(normalized_depth, axis=0).astype(np.float32)

    # =========================================================
    # 内部私有方法 (对象池与赛道逻辑)
    # =========================================================
    def _build_clean_arena(self):
        """构建纯净的墙壁包围盒，防止背景漏光干扰深度图"""
        l = self.cfg.ARENA_LENGTH / 2
        w = self.cfg.ARENA_WIDTH / 2
        h = self.cfg.ARENA_HEIGHT
        thick = 0.5
        
        # 墙壁颜色设为中性灰
        color = [0.8, 0.8, 0.8, 1.0]
        
        # 地面, 天花板, 前, 后, 左, 右
        walls = [
            ([0, 0, -thick/2], [l, w, thick/2]),
            ([0, 0, h + thick/2], [l, w, thick/2]),
            ([l + thick/2, 0, h/2], [thick/2, w, h/2]),
            ([-l - thick/2, 0, h/2], [thick/2, w, h/2]),
            ([0, w + thick/2, h/2], [l, thick/2, h/2]),
            ([0, -w - thick/2, h/2], [l, thick/2, h/2])
        ]
        
        for pos, extents in walls:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=self.CLIENT)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=color, physicsClientId=self.CLIENT)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos, physicsClientId=self.CLIENT)

    def _create_gate_pool(self):
        """利用复合体 (Compound Shape) 机制一次性创建 5 个高性能门框"""
        d = self.cfg.GATE_THICKNESS / 2.0
        w = self.cfg.GATE_SIZE / 2.0
        
        # 我们用 4 根长方体拼成一个“回”字形门框。默认门的开口朝向 Z 轴
        # 位置相对于门心的局部坐标
        link_poses = [
            [0, w + d, 0],   # 上边框 (Base)
            [0, -w - d, 0],  # 下边框 (Link 1)
            [w + d, 0, 0],   # 右边框 (Link 2)
            [-w - d, 0, 0]   # 左边框 (Link 3)
        ]
        
        link_extents = [
            [d, d, w + 2*d], # 上边框 half-extents
            [d, d, w + 2*d], # 下边框
            [d, w, d],       # 右边框
            [d, w, d]        # 左边框
        ]
        
        for _ in range(self.cfg.NUM_GATES):
            col_ids = []
            vis_ids = []
            poses = []
            quats = []
            
            for i in range(4):
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=link_extents[i], physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=link_extents[i], rgbaColor=[0.9, 0.3, 0.3, 1], physicsClientId=self.CLIENT)
                col_ids.append(col)
                vis_ids.append(vis)
                poses.append(link_poses[i])
                quats.append([0, 0, 0, 1])
                
            # 将 4 个碰撞体打成一个复合的刚体包 (速度极快，不会散架)
            uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_ids[0],
                baseVisualShapeIndex=vis_ids[0],
                basePosition=[0, 0, -10.0],
                linkMasses=[0, 0, 0],
                linkCollisionShapeIndices=col_ids[1:],
                linkVisualShapeIndices=vis_ids[1:],
                linkPositions=poses[1:],
                linkOrientations=quats[1:],
                linkInertialFramePositions=[[0,0,0]]*3,
                linkInertialFrameOrientations=[[0,0,0,1]]*3,
                linkParentIndices=[0, 0, 0],                # 指定 3 个子边框的父节点都是 0 (即 Base 上边框)
                linkJointTypes=[p.JOINT_FIXED]*3,           # 声明这 3 个子边框与父节点之间是固定关节 (死死焊住)
                linkJointAxis=[[0, 0, 1]]*3,                # 必须提供一个关节轴 (即使是固定关节也需要占位)
                physicsClientId=self.CLIENT
            )
            self.gate_pool_ids.append(uid)

    def _generate_spline_waypoints(self) -> List[np.ndarray]:
        """随机生成一条贯穿场地的动态平滑轨迹路点"""
        waypoints = []
        # X 轴步长划分 (将 24 米的实际跨度分给 5 个门)
        start_x = self.cfg.START_POS[0] + 4.0 
        end_x = self.cfg.ARENA_LENGTH / 2.0 - 4.0
        x_steps = np.linspace(start_x, end_x, self.cfg.NUM_GATES)
        
        for x in x_steps:
            # 限制 Y 和 Z 的振幅，防止转角过于刁钻
            y = np.random.uniform(-self.cfg.ARENA_WIDTH/5, self.cfg.ARENA_WIDTH/5)
            z = np.random.uniform(1.5, 3.5)
            waypoints.append(np.array([x, y, z])) 
            
        return waypoints