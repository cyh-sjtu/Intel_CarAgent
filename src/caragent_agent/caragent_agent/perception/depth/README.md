# Depth Anything V2 — Monocular Depth Estimation

单目相对深度估计，支持 OpenVINO IR 推理和 PyTorch 原生推理。

## 文件

| 文件 | 功能 |
|------|------|
| `convert_depth_anything_openvino.py` | 将 Hugging Face 模型转换为 OpenVINO IR |
| `run_depth_anything_openvino.py` | OpenVINO 推理（推荐，DK-2500 GPU 加速） |
| `run_depth_anything_v2.py` | PyTorch 原生推理（本地调试/对比） |

## 模型转换

```bash
python3 -m caragent_agent.perception.depth.convert_depth_anything_openvino \
  --model-id depth-anything/Depth-Anything-V2-Small-hf \
  --hf-endpoint https://hf-mirror.com \
  --output-dir ~/caragent_ws/models/depth_anything_v2_openvino
```

默认使用 `Depth-Anything-V2-Small-hf`，适合 DK-2500 的算力。

## OpenVINO 推理

```bash
python3 -m caragent_agent.perception.depth.run_depth_anything_openvino \
  --image ~/caragent_ws/keyframes/session_xxx/selected/left/000013.png \
  --model-dir ~/caragent_ws/models/depth_anything_v2_openvino \
  --device GPU \
  --output-dir ~/caragent_ws/perception_outputs/depth
```

输出：
- `*_depth.npy` — 相对深度数组 (H×W float32)
- `*_depth_color.png` — 彩色可视化

### 设备选择

- `--device GPU` — DK-2500 Arc GPU，推荐
- `--device CPU` — CPU 回退

## PyTorch 推理

用于本地有 GPU 时的调试和对比：

```bash
python3 -m caragent_agent.perception.depth.run_depth_anything_v2 \
  --image path/to/image.png \
  --output-dir outputs
```

## 输出说明

Depth Anything V2 输出**相对深度**（非米制）。数值越大表示越近，但具体映射因场景而异。米制尺度通过 `perception/fusion/project_scan_fit_monodepth.py` 中的 LiDAR 投影 + 曲线拟合获得。

## 依赖

- OpenVINO 2024+（OpenVINO 推理）
- PyTorch + Transformers（PyTorch 推理 + 模型转换）
