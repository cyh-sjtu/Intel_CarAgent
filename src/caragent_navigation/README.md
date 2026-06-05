# caragent_navigation

Nav2 导航 bringup 包：全局路径规划 + 局部避障 + 速度指令生成。

## 组成

Nav2 是一个完整的 ROS2 导航栈，本包负责启动和配置它：

| 模块 | 功能 |
|------|------|
| Global Planner (Navfn) | 在静态地图上规划全局路径 |
| Local Planner (Regulated Pure Pursuit) | 基于局部代价地图生成实时速度指令 |
| Global Costmap | 基于静态地图的全局障碍物层 |
| Local Costmap | 基于实时 LiDAR 的动态避障层 |
| Behavior Tree | 导航行为编排：规划 → 控制 → 恢复 |
| Map Server (可选) | 加载 yaml 栅格地图（默认关闭，定位模式自带 /map） |

## 目录

| 路径 | 说明 |
|------|------|
| [launch/caragent_nav2.launch.py](launch/caragent_nav2.launch.py) | Nav2 启动文件 |
| [config/nav2_params.yaml](config/nav2_params.yaml) | Nav2 全部参数（planner、controller、costmap、BT） |
| [rviz/caragent_nav.rviz](rviz/caragent_nav.rviz) | 导航专用 RViz 配置 |

## 关键参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `max_linear_vel` | 0.30 m/s | 室内导航速度上限 |
| `max_angular_vel` | 0.85 rad/s | 最大转向速度 |
| `footprint` | 0.55×0.52 m | 车体投影，含前后左右各 5 cm 保护范围 |
| `inflation_radius` | 0.24 m local / 0.36 m global | 障碍物膨胀 |
| `goal_tolerance_xy` | 0.12 m | 到达判定距离 |
| `goal_tolerance_yaw` | 0.20 rad | 到达判定角度 |

## 依赖

- `caragent_bringup` — 定位（localization 模式）提供 /map 和 TF
- `nav2_bringup` — Nav2 核心启动
- `slam_toolbox` — 地图发布
