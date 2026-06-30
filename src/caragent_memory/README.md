# caragent_memory

`caragent_memory` 负责关键帧候选采集、图像质量评估、CLIP/DINOv2 特征生成和场景记忆构建。它将机器人经过的位置与视觉观测绑定起来，为 Agent 提供轻量化语义场景地图。

## 模块定位

当前系统没有构建稠密三维语义地图，而是记录带位姿的关键帧图像，并在离线阶段筛选为可检索的场景记忆。该路线计算负担较低，适合低成本移动平台和边缘 AI 部署。

## 关键入口文件

| 文件 | 作用 |
| --- | --- |
| `launch/caragent_keyframe_collect.launch.py` | 启动定位、相机和 keyframe recorder |
| `keyframe_recorder_node.py` | 在线记录图像、位姿、scan 摘要和质量指标 |
| `build_scene_memory.py` | 从一次录制 session 自动完成关键帧筛选、语义标注、chunk index 和统一 manifest |
| `select_keyframes.py` | 关键帧筛选与 selected 数据集基础结构生成 |
| `dataset.py` | 数据集结构、manifest 读写和记录解析 |
| `image_quality.py` | 清晰度、亮度、对比度等图像质量评估 |
| `openvino_clip.py` / `convert_clip_openvino.py` | CLIP OpenVINO 图像特征推理与转换 |
| `dinov2_encoder.py` | DINOv2 PyTorch 图像特征编码回退 |
| `dinov2_openvino.py` / `convert_dinov2_openvino.py` | DINOv2 OpenVINO 图像特征推理与转换 |
| `scan_summary.py` | LiDAR scan 摘要保存 |

## 数据集结构

在线采集候选帧：

```text
~/caragent_ws/keyframes/<session_name>/
├── raw/             side-by-side 原图
├── left/            左目图像
├── right/           右目图像
├── pose/            关键帧 map 位姿
├── scan/            scan 摘要
├── meta/            图像质量、TF 状态、采集原因
├── manifest.jsonl   候选帧索引
└── session.json     采集参数快照
```

场景记忆输出：

```text
selected/
├── selected_manifest.jsonl
├── rejected_manifest.jsonl
├── review.html
├── scene_memory_summary.json
├── embeddings/
└── constructed_memory/
    ├── keyframe_nodes/
    ├── keyframe_graph.json
    ├── scene_memory_manifest.json
    ├── semantic_chunk_index_records.json
    └── semantic_chunk_index_matrix.npy
```

## 采集规则

`keyframe_recorder_node` 订阅 `/stereo/image_raw`、`/scan`、`/odom` 和 TF，在满足以下条件时保存候选帧：

- 第一帧或手动触发 `/keyframe_recorder/capture_once`。
- 距离上次保存超过最小时间，且位移或航向变化超过阈值。
- 图像通过清晰度、亮度和对比度基本质量检查。
- TF 可解析为 `map -> base_link` 位姿。

默认采集阈值包括 `min_time_sec=1.5`、`min_distance_m=0.65`、`min_yaw_deg=30.0`。

## 特征与筛选

| 特征 | 用途 |
| --- | --- |
| CLIP image embedding | 图文语义检索、参考图片与语义 chunk 匹配 |
| DINOv2 image embedding | 图像间相似度、冗余去重、地点视觉相似性 |
| 位姿与 yaw | 保证空间覆盖，避免只按图像相似度筛选 |
| image quality | 过滤模糊、过暗、过曝或低对比度帧 |

DINOv2 是当前默认去重后端，并默认通过 OpenVINO NPU 生成视觉特征；PyTorch 后端保留为回退路径。CLIP OpenVINO 图像编码用于边缘端语义检索加速。

## 一键构建场景记忆

录制结束后推荐使用统一入口：

```bash
ros2 run caragent_memory build_scene_memory \
  --dataset ~/caragent_ws/keyframes/session_YYYYMMDD_HHMMSS \
  --clip-model ~/caragent_ws/models/clip-vit-base-patch32/image_encoder.xml \
  --dinov2-model ~/caragent_ws/models/dinov2 \
  --device GPU \
  --dinov2-device auto \
  --annotate auto \
  --chunk-index auto
```

`--annotate auto` 会在 `DASHSCOPE_API_KEY`、`DASHSCOPE_API_KEYS` 或本地忽略配置可用时自动调用 VLM 标注；没有 API key 时跳过语义阶段但仍保留完整 selected/keyframe_nodes 结构。团队多 key 建议写在 `caragent_agent/config/local_config.yaml` 的 `api_keys.dashscope` 列表中，`--annotation-batch-size` 仍表示总并发上限，标注工具会把同一批请求轮询分摊到可用 key。`--chunk-index auto` 会在存在语义描述时预计算 `semantic_chunk_index_*`，减少 Agent 首次查询延迟。

## 与其他模块关系

- 上游依赖 `caragent_bringup` 提供定位、雷达和相机。
- 输出的 selected 数据集由 `caragent_agent.impression_graph.SceneMemory` 加载。
- `caragent_agent` 的关键帧检索、参考图片匹配和导航工具依赖该数据集中的图像、位姿和特征。
- Dashboard 在“结束并生成场景记忆”时调用 `build_scene_memory`，手动“重建场景记忆”也走同一入口。

## 已实现能力

- 在线采集带 map 位姿的双目关键帧候选。
- 保存 scan 摘要和图像质量指标，便于后续审计。
- 使用 CLIP / DINOv2 特征进行筛选和冗余去除。
- 自动写出 `scene_memory_manifest.json`，将 keyframe nodes、graph、语义 chunk index 等产物收敛到一个完整模块。
- 支持预计算 semantic chunk index，减少 live Agent 首次查询延迟。

## 边界说明

- 当前关键帧记忆主要适用于相对稳定的室内空间。
- 动态环境变化可通过到达确认、局部搜索和周期性关键帧刷新扩展，但不作为当前已完全解决能力描述。
- 本包负责图像和特征数据准备，不直接调用 Nav2 或多模态大模型完成任务编排。
