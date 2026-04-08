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

## 🚀 基础准备：物理仿真环境配置 (Phase 0)

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



