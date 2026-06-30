# CarAgent - Indoor Semantic Navigation Robot for Visually Impaired Assistance

**2026 Intel Cup Undergraduate Electronic Design Contest — Embedded System Design Invitational**

CarAgent is a ROS2-based indoor semantic navigation robot designed to assist visually impaired individuals. It combines **open-vocabulary object detection** (GroundingDINO), **scene memory** (keyframe-based semantic indexing), and **Nav2 autonomous navigation** to understand natural-language navigation requests and guide users to their destinations.

## Features

- **Natural Language Navigation** — Users describe destinations in plain language (e.g. "take me to the water dispenser near the pillar"). The robot autonomously plans and executes the navigation.
- **Open-Vocabulary Perception** — GroundingDINO enables zero-shot detection of arbitrary object categories without retraining, significantly outperforming fixed-vocabulary detectors in real-world indoor environments.
- **Scene Memory** — Keyframe-based semantic indexing stores visual landmarks, text descriptions, and spatial relationships for efficient environment recall.
- **Left-Only Rotation Takeover** — A mechanical constraint-aware navigation strategy that handles the robot's turning radius by pre-aligning with left-only rotation before Nav2 plan execution.
- **Intel OpenVINO Optimization** — DINOv2 feature extraction runs on Intel NPU via OpenVINO for efficient onboard inference.
- **Web Dashboard** — Browser-based monitoring and control interface with real-time camera preview, navigation status, and keyframe management.

## Hardware

| Component | Detail |
|-----------|--------|
| Compute | Intel DK-2500 Developer Board |
| Camera | HuiBo Stereo Camera (3840×1200 @ 30fps) |
| LiDAR | SLAMTEC SLLiDAR |
| Chassis | Differential-drive mobile base, STM32 MCU control |
| Software | ROS2 Humble, Ubuntu 22.04 |

## Package Overview

| Package | Description |
|---------|-------------|
| `caragent_agent` | Async agent framework: planning, execution, tool routing, LLM integration, config |
| `caragent_bringup` | Full-system launch files and configuration |
| `caragent_description` | URDF robot model description |
| `caragent_memory` | Keyframe collection, scene memory indexing, keyframe selection |
| `caragent_navigation` | Nav2 launch, configuration, left-only goal proxy |
| `caragent_stm32_driver` | STM32 firmware communication node |
| `caragent_ui` | Web dashboard (FastAPI + HTML/CSS/JS) |
| `caragent_vision` | Stereo camera driver, calibration tools |
| `sllidar_ros2-main` | SLAMTEC LiDAR ROS2 driver |

## Quick Start

### Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+
- Intel OpenVINO Runtime (for NPU acceleration)

### Build

```bash
cd caragent_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Launch (Simulation)

```bash
ros2 launch caragent_bringup caragent_full.launch.py simulation_mode:=true
```

### Launch (Real Robot)

```bash
ros2 launch caragent_bringup caragent_full.launch.py simulation_mode:=false
```

Open the dashboard at `http://localhost:8234/demo`.

## Key Technologies

- **GroundingDINO**: Open-vocabulary object detection
- **Nav2**: ROS2 navigation stack (global/local planning, AMCL, costmaps)
- **OpenVINO + DINOv2**: NPU-accelerated visual feature extraction
- **LLM Integration**: Async agent framework with multi-model support (DeepSeek, DashScope, DouBao)

## Project Structure

```
src/
├── caragent_agent/       # Agent core (planning, execution, tools, config)
├── caragent_bringup/     # Launch files & system-level config
├── caragent_description/ # URDF model
├── caragent_memory/      # Keyframe collection & scene memory
├── caragent_navigation/  # Nav2 integration & left-only proxy
├── caragent_stm32_driver/# STM32 communication
├── caragent_ui/          # Web dashboard
├── caragent_vision/      # Stereo camera driver
└── sllidar_ros2-main/    # LiDAR driver
```

## License

This project is open-sourced as part of the 2026 Intel Cup Embedded System Design Invitational Contest.

## Acknowledgments

- 2026 Intel Cup Organizing Committee
- Intel DK-2500 Platform Support
- ROS2 & OpenVINO Communities
