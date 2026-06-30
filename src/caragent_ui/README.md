# caragent_ui

`caragent_ui` 是 CarAgent 的 Dashboard Web Console，用于在开发板上统一启动、停止和查看 ROS2 流程。它面向实车调试和比赛演示准备，不是最终面向视障用户的无障碍 App。

## 模块定位

Dashboard 负责把多条命令行流程封装为浏览器操作：

- 建图、定位、导航、关键帧采集、Agent、相机测试、物体深度测试。
- 地图、关键帧 session、感知输出和日志浏览。
- 串口、相机设备、地图文件和运行参数选择。
- 进入 Agent UI，完成自然语言/参考图片语义导航演示。

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `caragent_ui/dashboard_node.py` | HTTP server、流程管理、API 路由和 ManagedProcess |
| `caragent_ui/static/dashboard.html` | Dashboard 前端页面 |
| `setup.py` | 注册 `dashboard_node` console script |

## Dashboard 管理的流程

| kind | 实际命令/功能 |
| --- | --- |
| `slam` | `caragent_bringup caragent_full.launch.py mode:=slam` |
| `navigation` | `caragent_bringup caragent_full.launch.py mode:=navigation` |
| `keyframe_record` | `caragent_memory caragent_keyframe_collect.launch.py` |
| `agent` | `caragent_agent caragent_agent.launch.py` |
| `camera_test` | `caragent_vision huibo_stereo_camera.launch.py` |
| `object_approach_live` | `caragent_agent object_approach_live_test` |
| `stereo_calib_*` | 双目标定采集与计算 |
| `lidar_camera_*` | LiDAR-相机外参采集与计算 |

## Agent UI 分工

Dashboard 主页面与 Agent UI 是两个不同入口：

- Dashboard：管理 ROS2 进程、地图、相机、关键帧、测试工具和日志。
- Agent UI：由 `caragent_agent/scripts/demo_ui/async_agent_web_demo.py` 提供，默认端口 `8123`，负责聊天式任务输入、图片上传、run memory 展示和 session checkpoint 选择。

Dashboard 可作为 Agent UI 的入口页，但具体 Resume / New session、自然语言输入和参考图片导航在 Agent UI 内完成。

## 数据输入与输出

输入：

- 浏览器 POST API 请求。
- 地图、数据集、串口、相机和运行参数。

输出：

- 启动/停止的 ROS2 子进程。
- Dashboard 状态 JSON、日志、地图列表、关键帧 session 列表和感知结果列表。

## 已实现能力

- 管理多类 ROS2 进程，并缓存日志。
- 扫描可用串口和相机设备。
- 根据 UI 参数动态拼接 launch 命令。
- 支持高分辨率相机参数传递。
- 管理关键帧筛选、可视化、语义标注和节点读取流程。
- 支持 left-only 接管测试相关参数。

## 边界说明

- Dashboard 不是最终视障辅助 App。
- UI 只负责进程管理和交互展示，不直接承担 SLAM、Nav2、感知或 Agent 推理。
