# async_agent

`async_agent` 是 CarAgent 的 LangGraph 异步多智能体实现。它将一次用户请求拆解为输入整理、任务编排、计划生成、工具执行、后台分析和用户回复等节点，并在运行过程中记录计划、任务、导航、观测和工具证据。

## 模块定位

该模块承担 Agent 的“任务大脑”职责：

- 将自然语言或参考图片输入转化为结构化任务。
- 为每个任务选择合适工具，如关键帧检索、图片匹配、Nav2 导航、当前画面分析。
- 将工具结果写入 run memory，供后续任务、会话恢复和运行分析使用。
- 通过 checkpoint 支持中断恢复，适合实车演示。

## LangGraph 节点

| 节点 | 创建入口 | 职责 |
| --- | --- | --- |
| `ingest` | `create_ingest_node` | 读取用户输入，注入历史上下文和运行时状态 |
| `orchestrate` | `create_orchestrate_node` | 判断当前状态，决定规划、执行或结束 |
| `plan` | `create_plan_node` | 生成或编辑任务计划，维护任务依赖 |
| `execute` | `create_execute_node` | 调用工具，处理工具结果，推进任务状态 |
| `bg_worker_*` | `create_background_worker_node` | 并行执行适合后台运行的检索/分析任务 |

节点图由 `async_agent_graph.py:create_async_agent` 组装：

```text
ingest → orchestrate
          ├─ plan → orchestrate
          ├─ execute → orchestrate
          └─ END

plan / execute 可触发 background workers
background workers 通过 shared_background_results 回填结果
```

## 关键子目录

| 子目录 | 职责 |
| --- | --- |
| `orchestration/` | 编排节点、路由、运行时控制和输入处理 |
| `planning/` | 计划图、任务图、计划编辑、prompt 构造 |
| `execution/` | 工具调用、导航 action 分发、工具预算、结果规范化 |
| `memory/` | run memory 导出和快照支持 |
| `runtime/` | 类型定义、资源调度、referents、控制台和兼容元数据 |
| `target_resolution/` | 语义目标解析、session anchors、目标策略 |
| `response/` | 面向用户的回复生成 |

## 工具注册

`async_agent_interface.py` 在 `_setup_tools()` 中注册当前工具：

- 场景检索：`KeywordSearchTool`、`RequirementSearchTool`、`GetKeyFrameNodesInfoTool`、`QueryMemoryTool`。
- 图片输入：`AttachedImageAnalyzerTool`、`AttachedImageKeyframeMatcherTool`、`AttachedImageObjectResolverTool`、`HistoricalKeyframeObjectPreanalysisTool`。
- 导航与当前状态：`NavigationTool`、`NavigationToPositionTool`、`Get_Current_State_Tool`、`CaptureCurrentViewTool`、`CurrentImageAnalyzerTool`。
- 物体精细定位：`ApproachObjectInCurrentViewTool`。

后台 worker 只允许检索和记忆类工具，避免导航和当前状态工具在后台并发执行。

## Run Memory

run memory 将一次会话中的关键信息组织为简化表：

| scope | 内容 |
| --- | --- |
| `conversation` | 用户输入、Agent 回复和轮次信息 |
| `plan` | 计划、任务图和执行状态 |
| `task` | 任务类型、依赖、目标、工具证据和结果 |
| `navigation` | 导航目标、到达状态和相关任务 |
| `observation` | 当前画面、图片引用和观测结果 |

`QueryMemoryTool` 暴露 `summary_table`、`timeline` 和 `detail` 三种视图，便于 Agent 查询当前 session 历史，而不是读取庞大原始 trace。

## Checkpoint 与演示恢复

Agent Web UI 在 `async_agent_web_demo.py` 中维护 session checkpoint：

- 保存可见会话、输入/输出、任务状态和 run memory 快照。
- 支持 Resume 和 New session 选择。
- Resume 时可恢复历史记忆，同时清理已完成的运行时残留计划，避免重复执行旧任务。

该机制适合实车演示中的断电、重启或长 session 回放。

## 边界说明

- Agent 负责高层任务编排，不直接闭环控制电机。
- 后台 worker 不执行导航类工具，避免并发移动风险。
- 当前视障辅助主题主要借助该模块验证语义目标理解和导航执行能力，专用语音交互可在后续接入。
