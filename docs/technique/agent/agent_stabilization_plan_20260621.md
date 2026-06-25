# Agent 后台与物体级工具收束计划（2026-06-21）

本文档记录 2026-06-21 深夜测试与复盘后的次日优化计划。明日目标不是继续扩功能，而是把已经跑通的关键帧导航、物体级定位、后台预分析三条链路收束成稳定、可解释、可展示的版本。

## 0. 核心原则

- 不用关键词硬规则修单个测试样例。
- 不绕开 Agent 体系，不把 planner/executor 改成固定流程脚本。
- 优先修结构化信息流：task schema、background result、tool result、memory、executor context。
- 前台仍由 Agent 选择工具；工具可以更聪明地复用后台结果，但不能偷偷改变任务语义。
- 后台结果是同一任务的预分析证据：能复用就复用，不能复用也要清楚说明原因。
- 用户可见状态必须清楚，避免长时间无反馈看起来像卡死。

## 1. 今日关键结论

今天的进展是有效的，不是无用功：

- 纯双目策略已恢复：`stereo_primary_mono_guard` 仍保留单目诊断字段，但最终 `selected_source=stereo`。
- 后台物体预分析的根因已定位并修复：staging keyframe 的最终 JSON 只在 `final_ai_content` 中，旧解析器优先看工具结果，导致一直认为 `waiting_for_staging_keyframe`。
- 已补充真实日志形态的 audit：工具只返回候选，最终回答才给 `destination.keyframe_id`。
- 板上 audit 与 `colcon build --packages-select caragent_agent --symlink-install` 已通过。
- 最近 6 个 session 日志已拉回本地：`temp_board_data/agent_logs_20260621_014619/`。

最重要的系统认识：

- 关键帧层面的后台配合总体是通的。
- 物体级后台不是完全没跑，而是之前卡在 staging keyframe 解析与前后台时序。
- 新暴露的核心问题是：后台经常在前台已经开始 task 后才完成，当前 fast path 只在 task 开始时检查一次，容易错过复用窗口。

## 2. 明日 P0：物体级后台等待与复用闭环

### 目标

让 `historical_keyframe_then_live` 的物体级任务在前台调用物体工具时，优雅地等待并复用同 task 的后台预分析结果。

期望行为：

```text
前台进入 object-level resolver
仍按 Agent 选择调用 approach_object_in_current_view
工具/工具 wrapper 发现该 task 有后台物体预分析正在运行
等待后台完成，最多等待一个有上限的窗口
若后台成功产出 recommended_destination，直接返回 tool_result_v1
若后台失败或超时，再 fallback 到 live 当前视野物体定位
```

### 设计方向

优先采用工具交汇点方案，而不是 executor 外部硬拦截：

- 前台还是正常调用 `approach_object_in_current_view`。
- tool wrapper 或工具执行上下文读取：
  - `current_task.task_id`
  - `current_task.resolver_kind`
  - `current_task.preanalysis_policy`
  - `shared_background_results[task_id]`
- 如果满足结构化条件：
  - `resolver_kind == object_level`
  - `preanalysis_policy == historical_keyframe_then_live`
  - background result 为 `running` / `completed`
- 则进入等待/复用逻辑。

不使用任务描述关键词判断。

### 建议等待策略

第一版取稳定优先：

- 默认等待 `20-30s`，配置化。
- 每 `0.5-1s` 检查一次后台结果。
- 后台 completed + `recommended_destination`：直接复用。
- 后台 failed：记录原因，fallback live。
- 超时：记录 timeout，fallback live。
- 前台 live 已完成后，晚到的后台只进入 memory/reference，不覆盖 live 结果。

### 验收标准

典型任务：

```text
靠近电梯口左侧的灭火箱
```

理想日志：

```text
Historical object preanalysis on keyframe 33 for task 3
Execute/ObjectTool: waiting for background object preanalysis, timeout=25s
Historical object preanalysis completed for task 3
Execute/ObjectTool: reused background preanalysis for task 3
```

若后台超时，应看到：

```text
Execute/ObjectTool: background object preanalysis timeout; fallback to live perception
approach_object_in_current_view ...
```

## 3. 明日 P0：物体级工具状态可见化

### 目标

物体级定位耗时较长，必须让前台日志/UI 能看到执行阶段，避免用户以为系统卡住。

### 状态阶段

建议统一为以下用户可理解阶段：

- `object_preanalysis_waiting`：正在等待后台物体预分析。
- `object_preanalysis_reused`：已复用后台物体预分析结果。
- `object_snapshot`：正在采集当前画面。
- `object_label_plan`：正在生成目标候选/查询计划。
- `object_grounding`：正在选择目标框。
- `object_segmentation`：正在分割目标区域。
- `object_depth`：正在估计目标深度。
- `object_goal_planning`：正在规划靠近位置。
- `object_done`：物体定位完成。
- `object_failed`：物体定位失败。

### 实现方向

- 复用 `ObjectApproachPipeline` 已有 `emit_progress(stage, status, message, payload)`。
- 在 `approach_object_in_current_view` 工具层接入 progress callback。
- 状态写入 foreground log / tool trace compact evidence。
- 若 UI 已有 session/progress 面板，尽量复用现有通道；不为此新增复杂 UI 大模块。

### 验收标准

运行一次物体级定位时，日志不应只有工具开始/结束，而应能看到阶段推进：

```text
[object_tool] object_snapshot running
[object_tool] object_grounding selected
[object_tool] object_segmentation ok
[object_tool] object_depth ok
[object_tool] object_goal_planning ok
[object_tool] object_done ok
```

后台复用场景应能看到：

```text
[object_tool] object_preanalysis_waiting running
[object_tool] object_preanalysis_reused ok
```

## 4. 明日 P1：减少重复检索与重复感知

### 当前现象

- 前台有时已经拿到 background candidate pack，仍重复搜索关键帧。
- 已经观察过当前画面，后续 task 仍重复做相似 observation。
- object-level resolver 前有时多余地调用 `get_current_state`。

### 收束方向

- 不做固定任务类型捷径。
- 利用已有结构化上下文：
  - background reference
  - upstream task outputs
  - arrival context
  - current place context
  - compact memory evidence
- 在 executor prompt/tool prompt 中强调：
  - 已有候选时优先比较与确认，不重复检索。
  - 只有缺少必要证据或冲突时才重新搜索。
  - 物体级工具本身会读取当前状态，非必要不要先单独调用 `get_current_state`。

### 验收标准

关键帧 resolver 中，如果后台已有候选，应明显减少重复：

- 不应连续多次调用同一 search 工具。
- 如果调用超过预算，应有 runtime guidance。
- 最终仍由 Agent 推理选择，不硬编码候选。

## 5. 明日 P1：Arrival 日志清理

### 当前现象

导航成功后偶尔仍有重复 arrival / unmatched arrival 日志。功能影响不大，但影响演示观感。

### 目标

- 已消费的同 token arrival 不再产生前台噪声。
- 保留 debug log，但 UI/assistant result 不刷屏。
- 维持安全：不能误吞真正的新 arrival。

### 验收标准

一次导航完成后：

- 前台只出现一次明确到达结果。
- 后续重复 controller status 最多记录为 debug/ignored duplicate。

## 6. 明日 P1：纯双目策略复核

### 背景

今天已改回纯双目作为最终 selected depth。单目融合仍保留为诊断字段。

### 测试目标

用 2-3 个典型目标确认停车距离是否自然：

- 近处箱子/蓝色纸箱。
- 电梯口左侧灭火箱。
- 门或远处桌子。

关注字段：

- `mono_guard_selected_source == "stereo"`
- `mono_guard_reason == "stereo_primary"`
- `object_base_xyz_m`
- `object_map_xy_m`
- 实际停车距离。

若纯双目稳定，就不要再当天来回切策略；把单目融合留作报告中的探索与未来优化。

## 7. 暂缓事项

明天不要开这些大坑：

- 不重构整个前后台调度成 keyframe job 黑板。
- 不做完整 artifact resume pipeline。
- 不新增多种深度融合策略。
- 不大改 planner prompt 架构。
- 不为刁钻物体写特殊规则。

这些可以写入后续工作：

- 关键帧检索共享 job/黑板。
- 物体工具 `resume_from_preanalysis`。
- 历史 artifact 分阶段复用：bbox/mask/depth/goal。
- 多 session memory 或跨日长期记忆。

## 8. 明日推荐执行顺序

1. 先确认今天修复后的后台 staging 解析在新任务中生效。
2. 实现物体工具等待后台完成并复用结果。
3. 加物体级工具状态日志。
4. 板上构建并重启 agent/dashboard。
5. 测试“电梯口左侧灭火箱”。
6. 测试一个当前视野物体，确认不会错误等待后台。
7. 复核纯双目停车距离。
8. 若仍有时间，再清理 arrival 重复日志和重复检索提示。

## 9. 辅助工具：Agent Workflow 仿真模式

### 目标

减少实机测试成本，在开发板上电但小车不运动的情况下，验证 planner、executor、后台预分析、arrival watchdog 和前后台衔接。

### 设计

- 在 `Nav2Controller` 层提供 `navigation.simulation_mode`。
- 开启后不发送 Nav2 action，不发布 `/cmd_vel`。
- 导航工具仍正常调用 `go_to_keyframe` / `go_to_position`。
- 控制器等待配置的延时后，更新虚拟 `position/yaw`，并发出原有格式的 arrival message。
- 上层 Agent 不感知这是仿真，因此能真实验证“导航期间后台是否完成、到达后任务是否继续、前台是否复用后台结果”。

### 使用方式

在板上的 `local_config.yaml` 覆盖：

```yaml
navigation:
  simulation_mode: true
  simulation_navigation_delay_sec: 30.0
  simulation_navigation_delay_per_meter_sec: 0.0
  simulation_initial_position: [0.0, 0.0, 0.0]
  simulation_initial_yaw_deg: 0.0
```

注意：

- `simulation_mode` 默认关闭，不影响实车。
- 这个模式不模拟真实图像变化，因此“当前视野临时物体”暂不适合验证。
- 历史关键帧导航、历史关键帧物体预分析、后台/前台调度非常适合用它验证。

## 10. 报告素材提示

如果明天 P0 收束成功，报告中可以这样表达：

- 系统采用前后台协同 Agent 架构。
- 前台负责交互、任务执行与安全 fallback。
- 后台利用导航时间提前做关键帧/物体级预分析。
- 物体工具支持复用后台预分析结果，减少现场等待。
- 若后台失败或超时，系统自动 fallback 到 live perception，不阻塞任务。
- 工具阶段状态可见，增强可解释性与调试能力。

这会比单纯说“我们用了大模型规划”更有含金量。

## 11. 后续边界升级：ambiguous 候选澄清

语义导航工具如果返回 `ambiguous`，当前阶段不应直接导航，也不应让
executor 自行猜测。短期策略是 fail fast，并把候选关键帧、理由和
failure reason 写入 task result/memory。

后续可以把它升级成更智能的交互：

- 新增 `clarification_required` 任务结果/状态。
- `submit_task_result` 提交候选 keyframe、缩略图路径、简短理由和面向用户的问题。
- UI 显示 2-4 个候选图片/关键帧卡片，让用户选择。
- 用户选择后，runtime 恢复当前任务或插入一个短 navigation task。

这条线只处理真实不确定性，不作为普通导航的硬规则。目标是让机器人在
多个候选都合理时主动澄清，而不是盲目行动或简单报错。
