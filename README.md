# 🚁 基于多伦多大学开源的无人机模型gym-pybullet-drones的RL训练与控制

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![OS](https://img.shields.io/badge/os-Ubuntu_22.04-green)

本项目致力于使用深度强化学习（Deep Reinforcement Learning）实现四旋翼无人机的控制，并面向真实的物理样机进行 Sim2Real（仿真到现实）部署准备。

本项目基于多伦多大学开源的超高保真无人机仿真库 [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)，摒弃了传统的简易质点运动学模型，直接针对电机转速（RPM / PWM）和真实空气动力学（地效、阻力）进行端到端的 Actor-Critic 策略训练。

---

## 🛠️ 硬件与系统要求 (Hardware & OS)

* **操作系统**：Ubuntu 22.04 LTS (推荐，兼容原生 ROS 2 环境)
* **GPU 算力**：NVIDIA RTX 5060 或同等支持最新架构的显卡
* **底层驱动**：CUDA 12.x 或 13.x
* **环境管理**：Miniconda / Anaconda

---

## 🚀 基础准备：物理仿真环境配置

本章节从零构建绝对干净且兼容最新 GPU 架构的 C++ 物理引擎与深度学习环境。

### Step 1. 创建干净的虚拟环境

由于我们需要精确控制底层 C++ 编译与 Python 版本的对应关系，请务必新建 Python 3.10 的虚拟环境：

```bash
conda create -n rl_drones python=3.10 -y
conda activate rl_drones
```

### Step 2. 克隆基础框架
下载多伦多大学的开源物理仿真引擎核心库：

```bash
git clone [https://github.com/utiasDSL/gym-pybullet-drones.git](https://github.com/utiasDSL/gym-pybullet-drones.git)
cd gym-pybullet-drones
```

### Step 3. 核心依赖安装与 GPU 适配
⚠️ 注意：新架构显卡（如 RTX 50 乃至 40 系列）极易因默认拉取老旧 PyTorch 导致 CUDA 运行时无法调用 GPU。我们需要强制指定官方最新的 cu121/cu130 构建版本，并利用清华源加速其他依赖。

```bash
# 1. 升级构建工具，防止 PyBullet 的 C++ 编译报错
pip install --upgrade pip setuptools wheel -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# 2. 强制拉取适配最新驱动的 PyTorch 2.x
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 3. 以开发者模式安装无人机库及其附属依赖
pip install -e . -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)

# 4. (可选) 解决 Ubuntu 下 ROS 2 全局环境变量导致的 PyYAML 冲突
pip install pyyaml -i [https://pypi.tuna.tsinghua.edu.cn/simple](https://pypi.tuna.tsinghua.edu.cn/simple)
```

### Step 4. 环境与物理引擎测试
安装完成后，首先验证 GPU 算力是否成功激活：

```Bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('GPU Ready:', torch.cuda.is_available())"
```
(期望输出：GPU Ready: True)  
接着，运行原生 PID 控制器，测试 PyBullet 的 3D GUI 渲染以及 OpenGL 是否正常工作：
```Bash
python gym_pybullet_drones/examples/pid.py
```
(如果一切顺利，你将看到一个弹出的 3D 窗口，几架无人机正基于传统 PID 算法沿预设轨迹飞行。)

## ➡️ Task 1: 无人机保持在指定高度

本任务是迈向 Sim2Real 的第一步。我们将训练一个未经任何经典控制算法（如 PID）预设的纯神经网络，让无人机学会从空中稳稳悬停。

### 1. 任务描述
* **初始状态**：无人机将被直接“抛飞”在 1.0m 的目标高度（出生转速为 0）。
* **任务目标**：在重力导致下坠的瞬间，迅速调整 4 个电机的推力，使无人机姿态平稳，并精准悬停在目标高度（误差 $< 0.1\text{m}$）。
* **挑战**：必须保证动作输出平滑无抖动，否则在真机部署时会烧毁电调（ESC）或电机。

### 2. 代码拉取与运行指南

本任务的所有核心代码均位于 `task1` 文件夹下。你可以通过以下指令获取代码：

```bash
# 如果尚未克隆仓库，请先克隆整个项目
git clone [https://github.com/0324Lw/gym-pybullet-drones.git](https://github.com/0324Lw/gym-pybullet-drones.git)
cd gym-pybullet-drones/task1
```

#### 文件结构与说明

| 文件名 | 功能说明 |
|--------|----------|
| `task1_env.py` | 包含超参数配置（Config）、符合 Gym 接口标准的高保真环境类（Env）以及可视化类（Plot）。 |
| `test_task1_env.py` | 环境测试脚本。用于验证状态/动作维度、终局事件（坠毁/偏离）以及单步奖励截断是否生效。 |
| `task1_train.py` | PPO 主训练脚本。支持多进程并行加速、动态学习率、在线特征归一化及自动化模型存储。 |
| `task1_model_test.py` | 模型评估脚本。调用 3D GUI 界面，对训练好的模型进行 10 秒钟无干扰的连续悬停视觉测试。 |

运行训练代码：
```bash
python task1_train.py
```

### 3. 强化学习建模 (RL Modeling)
针对无人机的非线性物理特性，本任务进行了如下的 RL 三要素设计：

#### A. 状态空间设计 (Observation Space)
- **基础维度**：11维  
  包括绝对坐标 $(x,y,z)$、欧拉角 $(\text{Roll}, \text{Pitch}, \text{Yaw})$、4个电机的历史控制转速，以及高度误差 $(\Delta z)$。
- **核心技巧 1：帧堆叠 (Frame Stacking)**  
  将当前帧与过去 3 帧拼接，形成 44 维输入。这让原本无记忆的纯 MLP 网络能够隐式感知到无人机的速度、加速度以及控制系统的硬件延迟。
- **核心技巧 2：在线滑动归一化**  
  采用 VecNormalize 包装器，在训练时动态收集数据的均值和方差，将 44 维物理量拉平至标准正态分布，加速网络收敛。

#### B. 动作空间设计与平滑技巧 (Action Space & Smoothing)
- **基准映射**  
  网络不直接输出转速，而是输出 $[-1, 1]$ 连续值。环境底层将其映射为目标悬停转速的 $\pm 50\%$ 微调比例。
- **核心技巧：EMA 低通滤波 (Exponential Moving Average)**  
  工业级 Sim2Real 的必备。神经网络的探索噪声会导致输出高频跳变，因此在环境底层加入 EMA 滤波（平滑系数 $\alpha=0.3$）。这既能滤除可能烧毁真实电机的抖动，又保留了多旋翼足够的姿态响应带宽。

#### C. 奖励函数组成 (Reward Shaping)
为了避免的局部最优和梯度爆炸，本任务设计了严苛且密集的奖励地貌，并在单步进行了截断保护。
1. **步数惩罚 (固定负值)**：每多走一步扣除固定分，逼迫无人机尽快稳定，避免偷懒。
2. **高度奖励 (高斯分布)**：距离目标高度越近，奖励呈指数级暴增；偏离则迅速衰减。
3. **平稳奖励 (线性负值)**：根据 Roll 和 Pitch 的绝对值进行线性惩罚（忽略 Yaw），保持机身水平。
4. **动作变化惩罚 (平方项)**：计算当前帧与上一帧网络输出的差值平方，严厉惩罚突变，鼓励输出平顺。
5. **偏离极刑 (大额负值)**：偏离目标高度 $>0.5\text{m}$，给予大额惩罚并直接结束回合。
6. **侧翻极刑 (大额负值)**：倾角超过物理容忍极限（约 23 度），视为坠毁，直接结束回合。
7. **完美达成奖励 (大额正值+时间补偿)**：连续 100 步稳定在误差 0.1m 内，发放巨额固定奖金，还会将剩余未走完的步数折算成“时间补偿”一次性发放，彻底消除“摸鱼”漏洞。

---

### 4. 算法、网络结构与超参数
#### 核心算法
- **PPO (Proximal Policy Optimization)**  
  由于本任务奖励地貌包含了极端的大额惩罚与奖励，Off-policy 算法（如 SAC）极易出现 Q 值过高估计导致网络崩溃。PPO 的 Clip 机制（信任区域）是当前任务最稳妥的选择。

#### 网络架构
- 独立的两层全连接神经网络（MLP）
- Actor 与 Critic 结构均为 `[256, 256]`
- 适合处理一维拉平的低维状态向量

#### 关键超参数与稳定性调优
- **并行环境**：8 个 SubprocVecEnv，充分压榨多核 CPU 算力。
- **学习率调度**：采用 Linear Schedule，从 $3 \times 10^{-4}$ 线性衰减至 $1 \times 10^{-5}$，防止后期最优解震荡。
- **梯度裁剪**：`max_grad_norm = 0.5`，防范偶尔出现的坠毁极值导致梯度爆炸。
- **Batch Size**：512，每个环境收集 1024 步（总共 8192 步）更新一次。
