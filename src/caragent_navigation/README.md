# caragent_navigation

`caragent_navigation` 是 CarAgent 的 Nav2 导航配置与目标代理包。它负责将地图定位结果、激光雷达局部感知和目标位姿组织为可执行路径，并根据实车机械特性提供 left-only 旋转接管策略。

## 模块定位

| 组成 | 当前职责 |
| --- | --- |
| Nav2 launch | 启动 planner、controller、behavior、costmap、BT navigator 等 Nav2 组件 |
| `nav2_params.yaml` | 配置路径规划、局部控制、代价地图和到达判定 |
| `left_only_goal_proxy.py` | RViz/外部目标代理，执行左向预对齐、Nav2 平移、左向终对齐 |
| RViz 配置 | 提供导航调试视图 |

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `launch/caragent_nav2.launch.py` | Nav2 启动入口，接收地图、端口、速度和 left-only 参数 |
| `config/nav2_params.yaml` | Nav2 全局规划、局部控制、代价地图和行为树参数 |
| `caragent_navigation/left_only_goal_proxy.py` | 左向旋转代理节点，订阅 `/caragent/left_only_goal` |
| `rviz/caragent_nav.rviz` | 导航调试布局 |

## 数据输入与输出

输入：

- `/map`：静态或定位模式下发布的占据栅格地图。
- `/scan`：2D 激光雷达扫描，用于局部 costmap 和安全检查。
- `/odom` / TF：机器人当前位置和姿态。
- `NavigateToPose` goal 或 `/caragent/left_only_goal`。

输出：

- `/cmd_vel`：发送给 `caragent_stm32_driver` 的速度指令。
- Nav2 action feedback/result：被 Agent controller 或 UI 用于状态展示。

## left-only 旋转接管

当前实车满意的导航策略是可选 left-only takeover：

1. 目标下发前，根据目标方向执行左向预对齐。
2. 平移段交给 Nav2 执行。
3. 到达目标附近后，执行左向终对齐到目标 yaw。

该策略不改变 Nav2 的全局规划能力，只在旋转阶段规避实车右转不稳定问题。

## 与其他模块关系

- `caragent_bringup` 在 `navigation` 模式下 include 本包 launch。
- `caragent_agent.controller.nav2.Nav2Controller` 通过 Nav2 action 或 `/cmd_vel` 接管策略调度导航。
- `caragent_stm32_driver` 接收 `/cmd_vel` 并下发到底盘。
- `caragent_memory` 中的关键帧位姿最终被转化为 Nav2 目标。

## 已实现能力

- Nav2 路径规划、局部避障和目标执行。
- 可选 map server，适配定位模式下已有 `/map` 的运行方式。
- left-only goal proxy 支持预对齐、路径方向估计、终对齐、安全半径检查和 Nav2 final-spin fallback。
- 支持 Dashboard 配置最大线速度、角速度和 left-only 开关。

## 边界说明

- Nav2 处理的是地图坐标和障碍物，不直接理解自然语言目标。
- 当前复杂动态避障不是当前主线，重点描述室内语义目标定位与基础导航执行闭环。
