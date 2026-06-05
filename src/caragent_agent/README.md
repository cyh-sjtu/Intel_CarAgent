# caragent_agent

LangGraph 异步 Agent + 感知管线 + Nav2 导航控制。接收自然语言任务，通过多节点 DAG 协作完成语义导航。

**当前状态**：关键帧语义导航已实现（场景记忆 → 自然语言检索 → 关键帧导航），感知管线完成（GroundingDINO + EfficientSAM + Depth Anything V2 + LiDAR 融合 + 双目深度）。物体级别定位工具接入中。

## 架构

```text
用户输入（自然语言）
  → Orchestrate Node（意图理解、任务拆分、工具分配）
    → Plan Node（规划步骤序列，可中断等待用户确认）
      → Execute Node（执行单个步骤：tool call → 结果解析）
        → Background Workers（并行分析：图像理解、语义检索等）
          → User Response（汇总结果、生成自然语言回复）

Agent 状态在节点间通过 LangGraph StateGraph 流转，支持 checkpoint 持久化。
```

### 节点职责

| 节点 | 文件 | 职责 |
|------|------|------|
| Orchestrate | `orchestration/orchestrate_node.py` | 理解用户意图，拆分任务，路由到 plan/response |
| Ingest | `orchestration/ingest_node.py` | 输入预处理、上下文注入 |
| Plan | `planning/plan_node.py` | 生成执行计划（步骤 DAG），支持用户编辑 |
| Execute | `execution/execute_node.py` | 逐步执行：工具调用 → 等待结果 → 下一工具 |
| Background Workers | `execution/background.py` | 并行执行分析类工具（图像、检索等） |
| Response | `response/user_response.py` | 汇总执行结果，生成用户回复 |

## 包结构

```
caragent_agent/
  agents/
    async_agent/          — LangGraph 多节点异步 Agent
      orchestration/      — 编排、路由、运行时控制
      planning/           — 任务规划、计划编辑、prompt 构建
      execution/          — 步骤执行、工具结果处理、后台 worker
      memory/             — Agent 记忆导出、持久化
      runtime/            — 运行时类型、资源调度、控制台
      response/           — 用户回复生成
    tools/                — Agent 工具集
      navigation/         — Nav2 导航工具（关键帧导航、坐标导航）
      memory/             — 场景记忆检索工具
      search/             — 关键词/需求搜索工具
      objects/            — 物体实例查询工具
      analysis/           — 图像分析工具
      current/            — 当前状态查询工具
      base/               — 工具基类
    base/                 — Agent 接口基类
  perception/             — 感知推理管线
    grounding/            — GroundingDINO 开放词汇检测（OpenVINO + PyTorch）
    sam/                  — EfficientSAM 分割（OpenVINO，encoder GPU + decoder CPU）
    depth/                — Depth Anything V2 单目深度（OpenVINO + PyTorch）
    fusion/               — LiDAR-相机深度融合 + 双目 SGBM
  controller/             — 导航控制器
    nav2/                 — Nav2 行为树动作客户端
  impression_graph/       — 场景印象图（关键帧 + 物体节点 + 空间关系）
  config/                 — 配置管理、运行时路径、设备 profile
  prompts/                — LLM prompt 模板（YAML）
  utils/                  — 工具函数（几何、LLM、导航、日志）
  scripts/                — 离线工具
    demo_ui/              — Web Demo（Gradio）
    annotate_keyframes.py — 关键帧语义标注
    draw_fused_topdown_intuitive.py  — 融合俯视图绘制
    draw_scan_diagnostics.py         — 扫描诊断图绘制
  third_party/            — 第三方参考实现
    from_langgraph/       — LangGraph ReAct Agent 参考
    from_vlm_grounder/    — VLM Grounder 参考
```

## 感知管线

详见各子包 README：

| 子包 | 文档 | 功能 |
|------|------|------|
| `perception/grounding/` | [README](caragent_agent/perception/grounding/README.md) | 开放词汇检测，模型转换，OpenVINO 推理 |
| `perception/sam/` | [README](caragent_agent/perception/sam/README.md) | Box-prompted 分割，encoder/decoder 分离 |
| `perception/depth/` | [README](caragent_agent/perception/depth/README.md) | 单目相对深度，模型转换，OpenVINO 推理 |
| `perception/fusion/` | [README](caragent_agent/perception/fusion/README.md) | LiDAR 投影 + 尺度拟合 + 边缘过滤 + 双目交叉验证 |

快速链接：
- [融合管线技术说明](caragent_agent/perception/fusion/OBJECT_LOCALIZATION_TECHNICAL_NOTE.md) — 坐标系、投影几何、标定、拟合策略
- [融合管线使用指南](caragent_agent/perception/fusion/README.md) — 全流程命令

## ROS2 接口

### 订阅话题

| 话题 | 消息类型 | 用途 |
|------|----------|------|
| `/odom` | `nav_msgs/Odometry` | 机器人里程计 |
| `/scan` | `sensor_msgs/LaserScan` | 激光雷达扫描 |
| `/stereo/left/image_raw` | `sensor_msgs/Image` | 左目图像 |
| `/stereo/right/image_raw` | `sensor_msgs/Image` | 右目图像 |
| `/tf` | `tf2_msgs/TFMessage` | 坐标变换 |
| `/map` | `nav_msgs/OccupancyGrid` | 占据栅格地图 |

### 发布话题

| 话题 | 消息类型 | 用途 |
|------|----------|------|
| `/goal_pose` | `geometry_msgs/PoseStamped` | Nav2 导航目标 |

### Action 客户端

- `navigate_to_pose` (Nav2) — 路径规划与执行

## 启动

```bash
# Agent ROS 节点
ros2 launch caragent_agent caragent_agent.launch.py

# Web Demo
ros2 run caragent_agent agent_web_demo
```

## 配置

LLM 配置和运行时参数通过 `config/*.yaml` 和 `caragent_agent/config/` 模块管理：

```python
from caragent_agent.config.config import config
# config.llm_model, config.agent_name, ...
```

## 依赖

- ROS2 Humble (rclpy, nav2_msgs, geometry_msgs, sensor_msgs, tf2)
- LangChain / LangGraph — Agent 框架
- OpenVINO 2024+ — 感知模型推理
- PyTorch — PyTorch 版感知脚本（可选）
- NetworkX — 图结构记忆
- Gradio — Web Demo UI
