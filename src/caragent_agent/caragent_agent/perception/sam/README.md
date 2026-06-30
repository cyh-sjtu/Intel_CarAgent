# EfficientSAM — Box-Prompted Segmentation

接收 GroundingDINO 检测框，生成物体 mask。使用 OpenVINO IR 推理，encoder/decoder 分离部署。

## 文件

| 文件 | 功能 |
|------|------|
| `convert_efficientsam_openvino.py` | 将 EfficientSAM 转换为 OpenVINO IR（encoder + decoder 分别导出） |
| `efficient_sam_openvino.py` | 库类：`EfficientSAMOpenVINO`，封装 encoder/decoder 推理 |
| `run_efficientsam_openvino.py` | CLI：从检测框生成 mask |

## 模型转换

```bash
python3 -m caragent_agent.perception.sam.convert_efficientsam_openvino \
  --output-dir ~/caragent_ws/models/efficient_sam_openvino
```

生成两个 IR 文件：
- `efficient_sam_vitt_encoder.xml` + `.bin` — 图像 encoder
- `efficient_sam_vitt_decoder.xml` + `.bin` — mask decoder

## 推理

```bash
python3 -m caragent_agent.perception.sam.run_efficientsam_openvino \
  --grounding-json outputs/000013_grounding_openvino.json \
  --label-query "door" \
  --encoder-xml ~/caragent_ws/models/efficient_sam_openvino/efficient_sam_vitt_encoder.xml \
  --decoder-xml ~/caragent_ws/models/efficient_sam_openvino/efficient_sam_vitt_decoder.xml \
  --device GPU \
  --decoder-device CPU \
  --output-dir outputs \
  --output-stem 000013
```

### 设备策略

**encoder 跑 GPU，decoder 必须跑 CPU。** decoder 在 GPU 上产生 NaN，是已知问题。

```bash
--device GPU          # 默认设备（encoder 使用）
--decoder-device CPU  # decoder 覆盖为 CPU
```

也可以用 `--encoder-device GPU --decoder-device CPU` 显式指定。

### 检测选择

`--label-query` 从 GroundingDINO 的多个检测中筛选目标。支持逗号分隔的多个关键词：

```bash
--label-query "door, elevator door"
```

匹配规则：检测的 label 包含任一关键词即可。如果没匹配到，回退到最高 score 的检测。

## 输出

- `*_mask_ov.png` — 二值 mask 图像
- `*_mask_overlay_ov.png` — 原图 + mask 叠加 + 检测框
- `*_segmentation_ov.json` — mask 路径、检测信息、mask 面积

```json
{
  "image": "/path/to/image.png",
  "mask_path": "/path/to/mask_ov.png",
  "mask_area_px": 12345,
  "source_detection": {
    "label": "door",
    "score": 0.45,
    "box": [320, 100, 480, 300]
  }
}
```

## 依赖

- OpenVINO 2024+
- PyTorch（仅模型转换时需要）
- PIL (Pillow)
