# Shooting Task Design

## 任务目标
Shooting（踢球）任务的目标是让机器人通过踢球动作，使球按照期望的方向和速度移动。

## 两种实现方式

### 1. 端到端方式 (End-to-End)
**特点**：
- 直接输出关节位置/速度动作
- 策略网络需要同时学习机器人移动和踢球技能
- 动作空间：关节位置/速度（与locomotion相同）

**适用场景**：
- 从头开始训练
- 需要学习复杂的踢球技巧
- 动作空间较大，训练时间较长

**接口设计**：
- `ActionsCfg`: 使用 `JointPositionActionCfg` 或 `JointVelocityActionCfg`
- `CommandsCfg`: 包含球的目标速度和方向命令
- `ObservationsCfg`: 包含机器人状态、球的状态、目标命令
- `RewardsCfg`: 奖励球的速度跟踪、方向跟踪、机器人稳定性等

### 2. 分层强化学习方式 (Hierarchical RL)
**特点**：
- 先训练velocity任务（机器人移动）
- 使用速度指令作为动作空间
- 上层策略输出速度指令，下层策略（预训练）执行移动

**适用场景**：
- 已有训练好的velocity策略
- 快速训练踢球技能
- 动作空间较小，训练效率高

**接口设计**：
- `ActionsCfg`: 使用 `VelocityActionCfg`（输出速度指令）
- `CommandsCfg`: 包含球的目标速度和方向命令
- `ObservationsCfg`: 包含机器人状态、球的状态、目标命令、当前速度指令
- `RewardsCfg`: 奖励球的速度跟踪、方向跟踪、速度指令的平滑性等

## 核心组件设计

### Commands (命令)
- **ball_target_velocity**: 球的目标速度命令
  - `lin_vel_x`, `lin_vel_y`: 水平方向速度
  - `lin_vel_z`: 垂直方向速度（可选，通常为0）
  - `direction`: 目标方向（可选，可以用速度方向表示）

### Observations (观测)
**Policy观测**：
- 机器人状态：base_lin_vel, base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel
- 球的状态：ball_pos_rel（相对机器人位置）, ball_vel_rel（相对机器人速度）, ball_vel_w（世界坐标系速度）
- 目标命令：ball_target_velocity
- 历史动作：last_action
- （分层方式）当前速度指令：current_velocity_command

**Critic观测**（特权信息）：
- 包含所有policy观测
- 额外的球物理属性（质量、摩擦系数等，用于domain randomization）

### Rewards (奖励)
**任务奖励**：
- `track_ball_vel_xy`: 跟踪球在xy平面的目标速度（指数奖励）
- `track_ball_vel_direction`: 跟踪球的目标方向（可选）
- `ball_velocity_magnitude`: 奖励球达到目标速度大小

**稳定性奖励**：
- `robot_alive`: 机器人保持站立
- `robot_stability`: 机器人姿态稳定性
- `base_height`: 保持合适的base高度

**动作惩罚**：
- `action_rate`: 动作变化率惩罚（平滑性）
- `joint_vel`: 关节速度惩罚
- `joint_acc`: 关节加速度惩罚

### Terminations (终止条件)
- `time_out`: 超时
- `robot_fall`: 机器人摔倒（base高度过低）
- `bad_orientation`: 机器人姿态异常
- `ball_out_of_bounds`: 球出界（可选）

### Events (事件)
**Reset事件**：
- `reset_robot`: 重置机器人位置和姿态
- `reset_ball`: 重置球的位置和速度
- `reset_robot_joints`: 重置关节状态

**Randomization事件**：
- `randomize_physics_material`: 随机化物理材质（摩擦系数、恢复系数）
- `randomize_ball_mass`: 随机化球的质量
- `randomize_robot_mass`: 随机化机器人质量

### Curriculum (课程学习)
- `ball_velocity_levels`: 逐步增加目标速度难度
- `ball_distance_levels`: 逐步增加球与机器人的初始距离
- `terrain_levels`: 逐步增加地形难度

## 文件结构

```
shooting/
├── __init__.py
├── DESIGN.md (本文档)
├── shooting_env_cfg.py (端到端版本配置)
├── hierarchical_shooting_env_cfg.py (分层版本配置)
└── mdp/ (可选，如果需要自定义MDP函数)
    ├── observations/
    ├── rewards/
    └── commands/
```

## 实现建议

1. **先实现端到端版本**，验证任务设计是否合理
2. **再实现分层版本**，复用velocity任务的预训练模型
3. **共享基础配置**：两个版本可以共享Observations、Rewards、Terminations等配置的大部分内容
4. **MDP函数扩展**：可能需要实现新的MDP函数用于球的状态观测和奖励计算

## 与locomotion的差异

1. **增加了球的状态**：需要观测球的位置、速度
2. **命令不同**：从机器人速度命令变为球的目标速度命令
3. **奖励重点不同**：重点奖励球的速度跟踪，而非机器人速度跟踪
4. **终止条件扩展**：可能需要考虑球的状态（出界等）

## 待实现的MDP函数

以下MDP函数需要自定义实现（在`mdp/`目录下）：

### Observations (观测函数)
- `ball_pos_rel(env, asset_name="ball")`: 球相对机器人的位置（在机器人body frame中）
- `ball_vel_rel(env, asset_name="ball")`: 球相对机器人的速度（在机器人body frame中）
- `ball_vel_w(env, asset_name="ball")`: 球在世界坐标系中的速度

### Rewards (奖励函数)
- `track_ball_vel_xy_exp(env, command_name, std)`: 跟踪球在xy平面的目标速度（指数奖励）
- `track_ball_vel_direction(env, command_name)`: 跟踪球的目标方向
- `ball_velocity_magnitude(env, command_name)`: 奖励球达到目标速度大小
- `command_rate_l2(env, command_name)`: 命令变化率的L2惩罚（用于分层版本）

### Terminations (终止函数)
- `ball_out_of_bounds(env, max_distance)`: 球出界检测

### Events (事件函数)
- `reset_ball_state(env, position_range, velocity_range)`: 重置球的状态
- `randomize_ball_mass(env, mass_range)`: 随机化球的质量

### Actions (动作函数，仅分层版本)
- `VelocityActionCfg`: 速度指令动作配置（输出速度命令给低层策略）

这些函数可以参考IsaacLab的标准MDP函数实现模式，使用`SceneEntityCfg`来指定资产。
