# caragent_stm32_driver

`caragent_stm32_driver` 是 ROS2 与 STM32 底盘固件之间的串口桥接节点。它把 STM32 上报的 `ODOM` 遥测解析为 ROS2 `/odom` 和 `odom -> base_link` TF，同时将 Nav2 输出的 `/cmd_vel` 转换为固件可解析的 `CMD` ASCII 指令。

## 模块定位

| 层级 | 当前职责 |
| --- | --- |
| ROS 输入 | 订阅 `/cmd_vel`，可选启用底盘控制 |
| 串口输出 | 发送 `CMD,<linear_mmps>,<angular_mradps>` |
| 串口输入 | 读取 STM32 周期上报的 `ODOM,...` |
| ROS 输出 | 发布 `/odom`，广播 `odom -> base_link` TF |

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `caragent_stm32_driver/stm32_driver_node.py` | 串口连接、`ODOM` 解析、`/cmd_vel` 转发、TF 发布 |
| `setup.py` | 注册 `stm32_driver_node` console script |
| `package.xml` | 声明 `rclpy`、`nav_msgs`、`geometry_msgs`、`tf2_ros`、`python3-serial` 等依赖 |

## 数据输入与输出

输入：

- `/cmd_vel` (`geometry_msgs/Twist`)：Nav2 或调试工具输出的速度指令。
- STM32 串口行：`ODOM,...`、`PCDBG,...`、`RCDBG,...` 等遥测/调试信息。

输出：

- `/odom` (`nav_msgs/Odometry`)：机器人里程计。
- `odom -> base_link` TF：供 slam_toolbox、Nav2、关键帧采集使用。
- 串口 `CMD,<v_mmps>,<w_mradps>`：发送给 STM32 固件。

## 核心参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `stm32_port` | `/dev/ttyUSB0` | STM32 串口设备 |
| `baud_rate` | `115200` | 串口波特率 |
| `enable_cmd_vel` | `false` | 是否订阅并下发 `/cmd_vel` |
| `cmd_send_rate_hz` | `20.0` | 串口速度指令发送频率 |
| `cmd_timeout_sec` | `0.3` | 超时后发送停车 |
| `max_linear_mps` | `0.12` | driver 层线速度限幅 |
| `max_angular_radps` | `3.5` | driver 层角速度限幅 |
| `base_link_yaw_offset_deg` | `180.0` | 固件坐标到 ROS 坐标的偏航修正 |
| `zero_odom_on_start` | `true` | 启动后建立本地里程计原点 |

## 与其他模块关系

- 上接 `caragent_navigation` / Nav2 的 `/cmd_vel`。
- 下接 STM32 固件的 `pc_cmd_user` 和 `odom_user`。
- 输出的 `/odom` 与 TF 被 slam_toolbox、Nav2、关键帧采集和 Agent 当前状态读取使用。
- 在 `caragent_bringup` 的 SLAM、定位和导航模式中被启动。

## 已实现能力

- 自动连接和重连串口。
- 解析 STM32 上报的 `ODOM` 行，提取位置、姿态、速度和调试字段。
- 发布 `/odom`，并以固定频率广播 TF。
- 将 `/cmd_vel` 限幅后转换为 mm/s 和 mrad/s 单位的 `CMD` 行。
- 当 `/cmd_vel` 超时或串口异常时停止机器人。

## 边界说明

- driver 只做协议桥接和基本保护，不负责路径规划。
- `/odom` 来源于底盘里程计与固件融合结果，地图级定位由 slam_toolbox / Nav2 进一步处理。
