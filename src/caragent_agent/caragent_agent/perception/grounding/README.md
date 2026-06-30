# GroundingDINO — Open-Vocabulary Object Detection

开放词汇目标检测，支持 OpenVINO IR 推理（DK-2500）和 PyTorch 原生推理（本地调试）。

## 文件

| 文件 | 功能 |
|------|------|
| `convert_grounding_dino_openvino.py` | 将 GroundingDINO 转换为 OpenVINO IR |
| `grounding_dino_openvino.py` | 库类：`GroundingDINOOpenVINO`，封装推理 + 前后处理 |
| `run_grounding_dino_openvino.py` | CLI：OpenVINO 推理（推荐，DK-2500 GPU 加速） |
| `run_grounding_dino.py` | CLI：PyTorch 原生推理（本地调试/对比） |

## Install on DK-2500

```bash
python3 -m pip install "openvino>=2024.0" torch transformers pillow
```

## Convert

```bash
cd /home/car/caragent_ws/src/caragent_agent
python3 -m caragent_agent.perception.grounding.convert_grounding_dino_openvino \
  --repo-dir /home/car/caragent_ws/src/GroundingDINO \
  --hf-endpoint https://hf-mirror.com \
  --output-dir /home/car/caragent_ws/models/grounding_dino_openvino
```

If `GroundingDINO` lives under `/home/car/caragent_ws/src`, prevent colcon from
trying to build it as a ROS package:

```bash
touch /home/car/caragent_ws/src/GroundingDINO/COLCON_IGNORE
```

Why not convert the Hugging Face model directly? The Transformers
GroundingDINO implementation currently exports unsupported ATen ops in
OpenVINO (`cummax`, `cummin`, `isin`, `special_logit`). Use the official
GroundingDINO implementation path instead.

If the board cannot download the checkpoint, copy it manually and pass:

```bash
python3 -m caragent_agent.perception.grounding.convert_grounding_dino_openvino \
  --repo-dir /home/car/caragent_ws/src/GroundingDINO \
  --weights /home/car/caragent_ws/models/checkpoints/groundingdino_swint_ogc.pth \
  --output-dir /home/car/caragent_ws/models/grounding_dino_openvino
```

GroundingDINO also downloads `bert-base-uncased` for text encoding. If the board
cannot reach Hugging Face, either keep `--hf-endpoint https://hf-mirror.com` or
copy a local BERT directory and pass:

```bash
python3 -m caragent_agent.perception.grounding.convert_grounding_dino_openvino \
  --repo-dir /home/car/caragent_ws/src/GroundingDINO \
  --weights /home/car/caragent_ws/models/checkpoints/groundingdino_swint_ogc.pth \
  --bert-path /home/car/caragent_ws/models/bert-base-uncased \
  --local-files-only \
  --output-dir /home/car/caragent_ws/models/grounding_dino_openvino
```

Runtime inference also needs the Hugging Face GroundingDINO processor files for
image preprocessing and postprocessing. For offline use, copy
`IDEA-Research/grounding-dino-tiny` to the board, for example:

```bash
/home/car/caragent_ws/models/grounding-dino-tiny
```

Then pass that local directory to the runner:

```bash
python3 -m caragent_agent.perception.grounding.run_grounding_dino_openvino \
  --image /home/car/caragent_ws/keyframes/session_20260524_005910/selected/left/000123.png \
  --text "wooden round table . chair . door ." \
  --model-id /home/car/caragent_ws/models/grounding-dino-tiny \
  --device CPU \
  --output-dir /home/car/caragent_ws/perception_outputs/grounding_dino
```

## Run

```bash
cd /home/car/caragent_ws/src/caragent_agent
python3 -m caragent_agent.perception.grounding.run_grounding_dino_openvino \
  --image /home/car/caragent_ws/keyframes/session_20260524_005910/selected/left/000123.png \
  --text "wooden round table . chair . door ." \
  --device CPU \
  --output-dir /home/car/caragent_ws/perception_outputs/grounding_dino
```

The output schema matches the PyTorch prototype:

```json
{
  "detections": [
    {
      "label": "wooden round table",
      "score": 0.37,
      "box": [389.0, 74.0, 590.0, 273.0],
      "box_int": [389, 74, 590, 273]
    }
  ],
  "metadata": {
    "backend": "openvino",
    "device": "CPU"
  }
}
```

### 设备选择

- `--device GPU` — DK-2500 Arc GPU，GroundingDINO 支持 GPU 推理
- `--device CPU` — CPU 回退

## PyTorch 推理

用于本地有 GPU 时的调试和对比，不需要 OpenVINO 模型转换：

```bash
python3 -m caragent_agent.perception.grounding.run_grounding_dino \
  --image path/to/image.png \
  --text "door . chair . table ." \
  --output-dir outputs
```

输出格式与 OpenVINO 版一致。

## GroundingDINOOpenVINO 类

`grounding_dino_openvino.py` 提供可直接导入的推理类：

```python
from caragent_agent.perception.grounding.grounding_dino_openvino import GroundingDINOOpenVINO

detector = GroundingDINOOpenVINO(
    model_dir="~/caragent_ws/models/grounding_dino_openvino",
    model_id="~/caragent_ws/models/grounding-dino-tiny",
    device="GPU",
)
detections = detector.detect("path/to/image.png", "door . chair .")
```

## 依赖

- OpenVINO 2024+（OpenVINO 推理）
- PyTorch + Transformers + GroundingDINO（PyTorch 推理 + 模型转换）
- PIL (Pillow)
