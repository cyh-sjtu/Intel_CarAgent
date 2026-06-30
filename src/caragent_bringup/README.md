# caragent_bringup

`caragent_bringup` 是整车 ROS2 启动编排包，负责把雷达、STM32 串口、机器人模型、slam_toolbox、Nav2 和双目相机按不同运行模式组合起来。它是系统从单模块能力走向整车闭环的 launch 层。

## 模块定位

| 运行模式 | 入口 | 当前用途 |
| --- | --- | --- |
| `slam` | `caragent_full.launch.py mode:=slam` | 启动雷达、STM32、URDF、slam_toolbox 建图与 RViz |
| `localization` | `caragent_full.launch.py mode:=localization` | 加载已有地图并进行定位，提供 `/map` 和 TF |
| `navigation` | `caragent_full.launch.py mode:=navigation` | 在定位基础上启动 Nav2，可选 left-only 目标代理 |
| keyframe collection 支撑 | 被 `caragent_memory` launch include | 为关键帧采集提供定位、雷达、STM32 和地图上下文 |

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `launch/caragent_full.launch.py` | 统一入口，根据 `mode` 选择 SLAM、定位或导航链路 |
| `launch/rplidar_c1_slam.launch.py` | 建图模式，包含雷达、STM32、robot_state_publisher、slam_toolbox |
| `launch/rplidar_c1_localization.launch.py` | 定位模式，加载序列化地图和定位参数 |
| `config/` | slam_toolbox 建图/定位参数 |
| `rviz/` | 建图、定位、导航和调试用 RViz 配置 |

## 数据输入与输出

输入：

- `laser_port`：SLAMTEC C1 雷达串口。
- `stm32_port`：STM32 串口。
- `map_file_name` / `map_yaml_file`：定位和导航使用的地图文件。
- 相机参数：`camera_device`、`camera_width`、`camera_height`、`camera_left_width`、`camera_right_width`、`camera_fps`。

输出：

- `/scan`：雷达扫描。
- `/odom` 与 `odom -> base_link` TF：来自 STM32 driver。
- `/map`：建图或定位结果。
- Nav2 所需 TF、地图和代价地图上下文。

## 与其他模块关系

- 依赖 `caragent_stm32_driver` 将 STM32 里程计接入 ROS2。
- 依赖 `sllidar_ros2-main` 提供 `/scan`。
- 依赖 `caragent_description` 发布机器人模型。
- 在 `navigation` 模式下 include `caragent_navigation/launch/caragent_nav2.launch.py`。
- 可选 include `caragent_vision/launch/huibo_stereo_camera.launch.py`，为关键帧和 Agent 当前画面提供图像话题。

## 已实现能力

- 使用统一 launch 参数切换建图、定位和导航模式。
- 将相机分辨率参数贯通到整车启动流程。
- 支持 Nav2 map server 由启动参数控制，适配定位链路已发布 `/map` 的情况。
- 支持 left-only 目标代理开关，便于在机械右转受限时使用左向旋转预对齐和终对齐策略。

## 注意事项

- `navigation` 模式需要已有地图和稳定定位。
- 相机默认发布 raw split 图像；高分辨率下不应默认套用旧标定文件进行 rectification。
- 实车测试前需确认 Agent 配置中的 `navigation.simulation_mode: false`。
