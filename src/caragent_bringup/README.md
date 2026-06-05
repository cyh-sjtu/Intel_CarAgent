# caragent_bringup

硬件 bringup 包：SLAM 建图、地图定位、全链路导航的 launch 文件与 slam_toolbox 参数配置。

## 目录

| 路径 | 说明 |
|------|------|
| [launch/](launch/) | 启动文件：SLAM、定位、统一入口 |
| [config/](config/) | slam_toolbox 参数（建图 / 定位模式） |
| [rviz/](rviz/) | RViz2 可视化配置文件 |

## 启动入口

```bash
# 统一入口（推荐）
ros2 launch caragent_bringup caragent_full.launch.py mode:=slam          # 建图
ros2 launch caragent_bringup caragent_full.launch.py mode:=localization  # 定位
ros2 launch caragent_bringup caragent_full.launch.py mode:=navigation    # 导航
```

## 依赖

- `caragent_stm32_driver` — STM32 串口里程计与 cmd_vel 转发
- `caragent_description` — URDF 机器人模型 + robot_state_publisher
- `caragent_navigation` — Nav2 导航（navigation 模式）
- `slam_toolbox` — 建图与定位
- `sllidar_ros2` — 激光雷达驱动
