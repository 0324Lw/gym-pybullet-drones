import pybullet as p
import numpy as np
import math
from typing import Tuple, List

# ==========================================
# 模块 1：WorldConfig (世界参数配置类)
# ==========================================
class WorldConfig:
    """集中管理物理地图、障碍物与传感器的所有超参数"""
    
    # --- 场地与起终点 ---
    ARENA_SIZE = 50.0                       # 场地边长 50x50 (米)
    WALL_HEIGHT = 3.0                       # 边界墙壁高度 (米)
    MIN_SG_DIST = 7.0                      # 起终点最小直线间距
    MAX_SG_DIST = 10.0                      # 起终点最大直线间距
    SAFE_ZONE_RADIUS = 5.0                  # 起终点绝对安全区半径 (无障碍物)
    Z_START_GOAL = 1.0                      # 起终点的 Z 轴高度
    
    # --- 静态障碍物 (柱林) ---
    NUM_STATIC_OBS = 0                     # 静态障碍物数量
    STATIC_RADIUS_MIN = 0.5                 # 最小半径
    STATIC_RADIUS_MAX = 1.5                 # 最大半径
    OBS_HEIGHT = 3.0                        # 所有障碍物高度 (彻底封死越顶逃课)
    MIN_OBS_GAP = 2.0                       # 障碍物之间的最小通行间距 (防死胡同)
    
    # --- 动态障碍物 (游走拦截者) ---
    NUM_DYNAMIC_OBS = 0                     # 动态障碍物数量
    DYNAMIC_RADIUS = 1.0                    # 动态障碍物半径
    DYNAMIC_MASS = 500.0                    # 极大质量，撞击时弹开无人机而非被撞飞
    DYNAMIC_SPEED = 1.5                     # 随机游走基准速度 (m/s)
    
    # --- LiDAR 感知配置 ---
    LIDAR_NUM_RAYS = 24                     # 激光雷达射线数量 (每 15 度一根)
    LIDAR_MAX_RANGE = 10.0                  # 雷达最大量程 (米)
    LIDAR_Z_OFFSET = 0.0                    # 相对于无人机质心的 Z 轴高度偏移

# ==========================================
# 模块 2：Task3World (物理与感知世界模型类)
# ==========================================
class Task3World:
    """
    负责动态构建物理障碍物、维护游走刚体动力学，并提供激光雷达感知接口。
    此类与强化学习逻辑解耦，只专注于提供真实的物理与传感器反馈。
    """
    def __init__(self, client_id: int):
        self.CLIENT = client_id
        self.cfg = WorldConfig()
        
        self.start_pos = np.zeros(3)
        self.goal_pos = np.zeros(3)
        
        # 记录生成的实体 ID，便于重置时销毁
        self.wall_ids = []
        self.static_obs_ids = []
        self.dynamic_obs_ids = []
        
        # 内部记录圆柱体位置和半径，用于防碰撞生成校验
        self._circles_record = [] 

    # ---------------------------------------------------------
    # 核心接口 1: 世界重置与动态建图
    # ---------------------------------------------------------
    def reset_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """清除旧世界，生成新迷宫，返回 (起点坐标, 终点坐标)"""
        self._clear_world()
        self._circles_record.clear()
        
        # 1. 构建竞技场边界墙
        self._build_boundaries()
        
        # 2. 生成起终点 (满足距离与边界约束)
        self._generate_start_goal()
        
        # 3. 撒点生成静态森林 (带有安全区和间距碰撞校验)
        self._generate_static_obstacles()
        
        # 4. 生成动态游走拦截者 (部署在起终点连线附近)
        self._generate_dynamic_obstacles()
        
        return self.start_pos.copy(), self.goal_pos.copy()

    # ---------------------------------------------------------
    # 核心接口 2: LiDAR 传感器模拟
    # ---------------------------------------------------------
    def get_lidar_scan(self, drone_pos: np.ndarray, drone_yaw: float, drone_id: int) -> np.ndarray:
        """
        发射 2D 激光射线检测障碍物。
        返回: 长度为 LIDAR_NUM_RAYS 的归一化 numpy 数组 [0.0 ~ 1.0]。1.0 表示畅通。
        """
        angles = np.linspace(0, 2 * math.pi, self.cfg.LIDAR_NUM_RAYS, endpoint=False) + drone_yaw
        ray_from = []
        ray_to = []
        
        # 为防止射线原点在无人机碰撞体内导致误判，设置微小的起始偏移
        offset_dist = 0.15 
        ray_length = self.cfg.LIDAR_MAX_RANGE - offset_dist
        
        for ang in angles:
            dx = math.cos(ang)
            dy = math.sin(ang)
            # 起点：略微偏离机身中心
            ray_from.append([
                drone_pos[0] + offset_dist * dx, 
                drone_pos[1] + offset_dist * dy, 
                drone_pos[2] + self.cfg.LIDAR_Z_OFFSET
            ])
            # 终点：最大量程
            ray_to.append([
                drone_pos[0] + self.cfg.LIDAR_MAX_RANGE * dx, 
                drone_pos[1] + self.cfg.LIDAR_MAX_RANGE * dy, 
                drone_pos[2] + self.cfg.LIDAR_Z_OFFSET
            ])

        # 底层射线批处理检测 (性能极高)
        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.CLIENT)
        
        normalized_distances = []
        for res in results:
            hit_id = res[0]
            hit_fraction = res[2] # 0.0 到 1.0 之间的碰撞比例
            
            # 如果撞到有效障碍物 (排除了虚无 -1 和自身无人机)
            if hit_id != -1 and hit_id != drone_id:
                # 换算为真实的相对于 LIDAR_MAX_RANGE 的归一化距离
                actual_dist = hit_fraction * ray_length + offset_dist
                norm_dist = actual_dist / self.cfg.LIDAR_MAX_RANGE
                normalized_distances.append(norm_dist)
            else:
                normalized_distances.append(1.0) # 未碰到，视野开阔
                
        return np.array(normalized_distances, dtype=np.float32)

    # ---------------------------------------------------------
    # 核心接口 3: 维持动态障碍物的游走动能
    # ---------------------------------------------------------
    def step_dynamics(self):
        """
        供外部 step() 调用的物理保姆函数。
        确保动态障碍物不会因为摩擦力停下，或发生翻滚/腾空。
        """
        for uid in self.dynamic_obs_ids:
            # 1. 速度维持 (防止卡死或衰减)
            vel, _ = p.getBaseVelocity(uid, physicsClientId=self.CLIENT)
            v_xy = np.array([vel[0], vel[1]])
            speed = np.linalg.norm(v_xy)
            
            # 如果速度大幅衰减 (发生碰撞后能量损失)，强行注入动能补满
            if speed < self.cfg.DYNAMIC_SPEED * 0.9:
                if speed > 0.05:
                    v_xy = (v_xy / speed) * self.cfg.DYNAMIC_SPEED
                else:
                    # 如果彻底卡死，给个随机方向的满速
                    ang = np.random.uniform(0, 2 * math.pi)
                    v_xy = np.array([math.cos(ang), math.sin(ang)]) * self.cfg.DYNAMIC_SPEED
                    
                p.resetBaseVelocity(
                    uid, 
                    linearVelocity=[v_xy[0], v_xy[1], 0.0], 
                    angularVelocity=[0.0, 0.0, 0.0], 
                    physicsClientId=self.CLIENT
                )
            
            # 2. 姿态锁定 (因为是弹性碰撞，可能导致柱子被撞歪或离地)
            pos, quat = p.getBasePositionAndOrientation(uid, physicsClientId=self.CLIENT)
            rpy = p.getEulerFromQuaternion(quat)
            # 只要倾角大于 0.01 或 Z 轴高度偏离，强行钉回地面垂直状态
            if abs(pos[2] - self.cfg.OBS_HEIGHT / 2) > 0.01 or max(abs(rpy[0]), abs(rpy[1])) > 0.01:
                p.resetBasePositionAndOrientation(
                    uid, 
                    [pos[0], pos[1], self.cfg.OBS_HEIGHT / 2], 
                    p.getQuaternionFromEuler([0, 0, 0]), 
                    physicsClientId=self.CLIENT
                )

    # =========================================================
    # 内部私有建图方法
    # =========================================================
    def _clear_world(self):
        """安全地清除物理世界中的对象，避免 C++ 抛出 Remove body failed 警告"""
        # 1. 获取当前物理引擎中所有真实存在的合法的 body ID
        num_bodies = p.getNumBodies(physicsClientId=self.CLIENT)
        valid_uids = [p.getBodyUniqueId(i, physicsClientId=self.CLIENT) for i in range(num_bodies)]
        
        # 2. 只有当记录的 ID 依然存活时，才执行针对性移除
        for uid in self.wall_ids + self.static_obs_ids + self.dynamic_obs_ids:
            if uid in valid_uids:
                try:
                    p.removeBody(uid, physicsClientId=self.CLIENT)
                except:
                    pass
                    
        # 3. 清空列表记录
        self.wall_ids.clear()
        self.static_obs_ids.clear()
        self.dynamic_obs_ids.clear()

    def _build_boundaries(self):
        """用 4 个长方体死死封住 50x50 区域"""
        half_l = self.cfg.ARENA_SIZE / 2.0
        h = self.cfg.WALL_HEIGHT
        thickness = 1.0
        
        # 北海南墙，东西双壁
        wall_data = [
            ([0, half_l + thickness/2, h/2], [half_l + thickness, thickness/2, h/2]),  # N
            ([0, -half_l - thickness/2, h/2], [half_l + thickness, thickness/2, h/2]), # S
            ([half_l + thickness/2, 0, h/2], [thickness/2, half_l + thickness, h/2]),  # E
            ([-half_l - thickness/2, 0, h/2], [thickness/2, half_l + thickness, h/2])  # W
        ]
        
        for pos, extents in wall_data:
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=self.CLIENT)
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[0.5, 0.5, 0.5, 0.8], physicsClientId=self.CLIENT)
            uid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=pos, physicsClientId=self.CLIENT)
            self.wall_ids.append(uid)

    def _generate_start_goal(self):
        """生成起终点，确保它们在地图内，且距离符合 [40, 50] 约束"""
        bound = self.cfg.ARENA_SIZE / 2.0 - self.cfg.SAFE_ZONE_RADIUS
        while True:
            sx, sy = np.random.uniform(-bound, bound, 2)
            gx, gy = np.random.uniform(-bound, bound, 2)
            dist = math.hypot(gx - sx, gy - sy)
            if self.cfg.MIN_SG_DIST <= dist <= self.cfg.MAX_SG_DIST:
                self.start_pos = np.array([sx, sy, self.cfg.Z_START_GOAL])
                self.goal_pos = np.array([gx, gy, self.cfg.Z_START_GOAL])
                break

    def _is_position_valid(self, x: float, y: float, radius: float) -> bool:
        """极严谨的几何防重叠校验器"""
        # 1. 不得侵入起点安全区
        if math.hypot(x - self.start_pos[0], y - self.start_pos[1]) < (self.cfg.SAFE_ZONE_RADIUS + radius): return False
        # 2. 不得侵入终点安全区
        if math.hypot(x - self.goal_pos[0], y - self.goal_pos[1]) < (self.cfg.SAFE_ZONE_RADIUS + radius): return False
        # 3. 不得与边界墙壁过于重合
        bound = self.cfg.ARENA_SIZE / 2.0
        if abs(x) + radius > bound or abs(y) + radius > bound: return False
        # 4. 不得与其他已经生成的柱子互相穿模或贴得太近
        for cx, cy, cr in self._circles_record:
            if math.hypot(x - cx, y - cy) < (radius + cr + self.cfg.MIN_OBS_GAP): return False
        return True

    def _create_cylinder(self, x, y, radius, is_dynamic=False) -> int:
        """底层 PyBullet 生成通用圆柱体的方法"""
        h = self.cfg.OBS_HEIGHT
        mass = self.cfg.DYNAMIC_MASS if is_dynamic else 0.0
        color = [0.8, 0.2, 0.2, 1] if is_dynamic else [0.2, 0.6, 0.2, 1] # 动态是红色，静态是绿色
        
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=h, physicsClientId=self.CLIENT)
        vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=h, rgbaColor=color, physicsClientId=self.CLIENT)
        uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x, y, h/2], physicsClientId=self.CLIENT)
        return uid

    def _generate_static_obstacles(self):
        """生成静态柱子森林"""
        bound = self.cfg.ARENA_SIZE / 2.0
        for _ in range(self.cfg.NUM_STATIC_OBS):
            attempts = 0
            while attempts < 100:
                x, y = np.random.uniform(-bound, bound, 2)
                r = np.random.uniform(self.cfg.STATIC_RADIUS_MIN, self.cfg.STATIC_RADIUS_MAX)
                if self._is_position_valid(x, y, r):
                    uid = self._create_cylinder(x, y, r, is_dynamic=False)
                    self.static_obs_ids.append(uid)
                    self._circles_record.append((x, y, r))
                    break
                attempts += 1

    def _generate_dynamic_obstacles(self):
        """生成在起终点连线附近随机游走的红柱杀手"""
        sx, sy = self.start_pos[0], self.start_pos[1]
        gx, gy = self.goal_pos[0], self.goal_pos[1]
        
        for _ in range(self.cfg.NUM_DYNAMIC_OBS):
            attempts = 0
            while attempts < 100:
                # 随机提取起终点连线上的一段比例 (避免贴着起终点生成)
                t = np.random.uniform(0.15, 0.85)
                # 附加垂直于连线的随机横向散布偏置 (-5m 到 5m)
                base_x = sx + t * (gx - sx) + np.random.uniform(-5.0, 5.0)
                base_y = sy + t * (gy - sy) + np.random.uniform(-5.0, 5.0)
                
                r = self.cfg.DYNAMIC_RADIUS
                if self._is_position_valid(base_x, base_y, r):
                    uid = self._create_cylinder(base_x, base_y, r, is_dynamic=True)
                    
                    # 赋予游走刚体的物理特效：0 摩擦力、绝对弹性反弹 (Restitution=1.0)
                    p.changeDynamics(
                        uid, -1, 
                        lateralFriction=0.0, 
                        spinningFriction=0.0, 
                        rollingFriction=0.0, 
                        restitution=1.0,
                        linearDamping=0.0,
                        angularDamping=0.0,
                        physicsClientId=self.CLIENT
                    )
                    
                    # 初始化出膛速度
                    angle = np.random.uniform(0, 2 * math.pi)
                    vx = self.cfg.DYNAMIC_SPEED * math.cos(angle)
                    vy = self.cfg.DYNAMIC_SPEED * math.sin(angle)
                    p.resetBaseVelocity(uid, linearVelocity=[vx, vy, 0], physicsClientId=self.CLIENT)
                    
                    self.dynamic_obs_ids.append(uid)
                    self._circles_record.append((base_x, base_y, r))
                    break
                attempts += 1