# caragent_agent

`caragent_agent` 是 CarAgent 的语义任务编排与感知工具包。它基于 LangGraph 构建异步多智能体 Agent，将自然语言目标或参考图片输入转化为关键帧检索、视觉复核、Nav2 导航、到达确认和物体精细定位等结构化机器人任务。

## 模块定位

当前 Agent 不是独立聊天应用，而是连接语义需求与真实机器人执行的任务层：

```text
用户目标
  → Agent 意图理解与任务规划
  → 关键帧语义记忆 / 当前画面 / 感知工具 / 导航工具
  → Nav2Controller
  → ROS2/Nav2 实车执行
  → 到达反馈、当前画面确认、run memory 记录
```

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `launch/caragent_agent.launch.py` | ROS2 Agent 启动入口 |
| `agent_ros_node.py` | ROS2 节点包装，接收命令、发布响应和图片 |
| `agents/async_agent/async_agent_interface.py` | Agent 类、LLM 构建、工具注册、LangGraph 创建 |
| `agents/async_agent/async_agent_graph.py` | ingest/orchestrate/plan/execute/background 节点图组装 |
| `controller/nav2/nav2_controller.py` | Nav2 Action 适配器、当前状态、当前画面、旋转接管 |
| `scripts/demo_ui/async_agent_web_demo.py` | Agent Web UI、session checkpoint、run memory 恢复 |
| `config/config.yaml` | LLM、scene memory、navigation、agent、路径和日志配置 |

## Agent 工具体系

| 工具类别 | 代表工具 | 当前职责 |
| --- | --- | --- |
| 语义检索 | `KeywordSearchTool`、`RequirementSearchTool`、`QueryMemoryTool` | 查询关键帧语义、需求匹配、会话记忆 |
| 参考图片 | `AttachedImageAnalyzerTool`、`AttachedImageKeyframeMatcherTool`、`AttachedImageObjectResolverTool` | 图片描述、候选关键帧匹配、图片指示目标解析 |
| 当前状态 | `Get_Current_State_Tool`、`CaptureCurrentViewTool` | 获取位置、状态和当前画面 |
| 导航执行 | `NavigationTool`、`NavigationToPositionTool` | 关键帧导航和地图坐标导航 |
| 物体定位 | `ApproachObjectInCurrentViewTool` | 当前画面目标检测、分割、深度估计和靠近点生成 |
| 场景信息 | `GetKeyFrameNodesInfoTool` | 读取关键帧节点信息 |

工具以结构化 schema 暴露给 LangGraph 节点，执行结果会被压缩为可记录、可复核的 tool evidence。

## 感知管线

| 子模块 | 职责 |
| --- | --- |
| `perception/grounding` | GroundingDINO 开放词汇检测，包含 OpenVINO 转换与推理入口 |
| `perception/sam` | EfficientSAM box-prompted 分割，支持 OpenVINO 推理 |
| `perception/depth` | Depth Anything V2 等深度工具 |
| `perception/fusion` | 双目深度、LiDAR/视觉融合、物体靠近目标点生成 |

这些工具主要服务到达后的目标确认和精细级空间定位，不替代 Nav2 的全局路径规划。

## 导航控制

`Nav2Controller` 订阅 `/odom`、`/scan`、左右目图像和 costmap，调用 `NavigateToPose` action 执行导航目标。它支持：

- 关键帧位姿或地图坐标目标下发。
- 到达容差判定和状态反馈。
- 当前画面采集，供到达确认和物体定位使用。
- left-only 旋转接管，与 `caragent_navigation` 的实车策略保持一致。
- simulation mode，用于流程回归；实车测试时配置为 `false`。

## 配置与运行时

| 配置区域 | 作用 |
| --- | --- |
| `scene_memory` | selected 数据集路径、OpenVINO CLIP text 设置 |
| `navigation` | Nav2 topic、相机 topic、dry run、simulation mode、旋转接管参数 |
| `agent` | 多智能体开关、目标解析、物体靠近深度后端、模型路由 |
| `paths` | 工作区、keyframes、models 和默认数据集路径 |
| `log_dir` | Agent 运行日志目录 |

`local_config.yaml` 可覆盖个人 API key、运行 profile 和本地路径，不写入公开仓库。

## 已实现能力

- LangGraph 异步多智能体工作流：ingest、orchestrate、plan、execute、background workers。
- 自然语言目标到关键帧检索和 Nav2 导航调度。
- 参考图片到候选关键帧匹配和 VLM 复核。
- 到达后当前画面采集与视觉确认。
- 当前画面物体检测、分割、双目/深度估计和靠近目标点生成。
- run memory 与 checkpoint 支持多轮会话、任务回放和演示恢复。

## 边界说明

- 当前 Agent 面向语义导航核心能力验证，尚未完成专门的视障无障碍语音交互闭环。
- 多模态大模型采用边云协同，不能描述为全离线系统。
- NPU 是 Intel 平台可扩展能力，未验证部署的模型不写成当前已实现。
