# Autonomy Server - ADAS Pipeline

**Real-time monocular ADAS backend** for lane keeping and collision avoidance.  
**~30 FPS** inference on Intel Iris Xe GPU using OpenVINO fp16.
**~13 FPS** inference on Jetson Nano 128 core maxwell 4 GB shared RAM using Tensorrt fp32.
---

## Pipeline Architecture

```plain
┌────────────────────────────────────────────────────────────────────────────┐
│                           AUTONOMY PIPELINE                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   ┌─────────┐     ┌──────────────────┐     ┌──────────────────────────┐    │
│   │  Frame  │────►│  Letterbox 640   │────►│  OpenVINO YOLOPv2 (GPU)  │    │
│   │  Input  │     │  (Any Res→640²)  │     │  → Lane + Road Logits    │    │
│   └─────────┘     └──────────────────┘     └────────────┬─────────────┘    │
│                                                         │                  │
│                                                         ▼                  │
│   ┌──────────────────────┐     ┌──────────────────────────────────────┐    │
│   │  YOLO Object Detect  │     │  Unletterbox → Binary Masks (orig)   │    │
│   │  → Bounding Boxes    │     └────────────┬─────────────────────────┘    │
│   └──────────┬───────────┘                  │                              │
│              │                              ▼                              │
│              │                 ┌──────────────────────────────┐            │
│              │                 │  Morphological Cleaning      │            │
│              │                 │  (Close → Open)              │            │
│              │                 └────────────┬─────────────────┘            │
│              │                              │                              │
│              ▼                              ▼                              │
│   ┌──────────────────────┐     ┌──────────────────────────────┐            │
│   │  ObjectPerception    │     │  RoadPerception              │            │
│   │  - Trapezoid ROI     │     │  - Row-wise centerline       │            │
│   │  - Distance estimate │     │  - EMA temporal smoothing    │            │
│   │  - Brake force calc  │     └────────────┬─────────────────┘            │
│   └──────────┬───────────┘                  │                              │
│              │                              ▼                              │
│              │                 ┌───────────────────────────────┐           │
│              │                 │  CenterlineMPC                │           │
│              │                 │  - 9 precomputed trajectories │           │
│              │                 │  - Bicycle model simulation   │           │
│              │                 │  - Cost: center + GPS + smooth│           │
│              │                 └────────────┬──────────────────┘           │
│              │                              │                              │
│              ▼                              ▼                              │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                    CONTROL OUTPUT                                │     │
│   │     Steering Angle (rad) │ Trajectory Points │ Brake Force       │     │
│   └──────────────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Inference Layer (`src/inference/`)

| File | Class | Description |
|------|-------|-------------|
| `openvino_engine.py` | `InferenceEngine` | YOLOPv2 on OpenVINO with FP16, baked preprocessing (NHWC→NCHW, BGR→RGB, /255) |
| `object_engine.py` | `ObjectInferenceEngine` | YOLO26n for vehicle detection |
| `object_engine.py` | `ObjectPerception` | Trapezoid ROI filtering, distance estimation, brake force calculation |

**Key Optimizations:**

- PrePostProcessor bakes normalization into graph → zero CPU overhead
- Model caching enabled (`./model_cache/`)
- Latency-optimized compile hints

### 2. Perception Layer (`src/adas/perception/road/`)

| File | Function/Class | Description |
|------|----------------|-------------|
| `segmentation.py` | `clean_road_mask()` | Morphological close+open to fill holes, remove noise |
| `road_v2.py` | `RoadPerception` | Extracts centerline from road mask with temporal EMA |

**RoadPerception Algorithm:**

1. Sample bottom 60% of image at every 4th row
2. For each row, find left/right road boundaries via `argmax`
3. Apply EMA smoothing on geometry (α=0.25) for temporal stability
4. Output: List of (x, y) centerline points + confidence score

### 3. Control Layer (`src/adas/control/`)

| File | Class | Description |
|------|-------|-------------|
| `mpcv2.py` | `CenterlineMPC` | MPC-style steering controller with precomputed trajectory cache |

**CenterlineMPC Algorithm:**

1. **Precompute** 9 candidate trajectories at init (bicycle model, 40 steps each)
2. For each candidate:
   - Check if trajectory stays inside road mask (vectorized bound check)
   - Compute cost: `W_CENTER * deviation + W_GPS * bias + W_SMOOTH * jerk`
3. Select lowest-cost trajectory
4. Apply EMA smoothing on output steering (α=0.25)

**Cost Weights:**

- `W_CENTER = 1.0` — Follow centerline
- `W_GPS = 0.3` — Follow GPS bias
- `W_SMOOTH = 0.2` — Minimize steering jerk

---

## Project Structure

```plain
src/
├── main.py                      # FastAPI WebSocket server
├── api/
│   └── models.py                # Pydantic schemas (SensorMessage, AutonomyMessage)
├── inference/
│   ├── openvino_engine.py       # YOLOPv2 inference
│   ├── object_engine.py         # YOLO detection + ObjectPerception
│   └── tensorrt_engine.py       # TensorRT backend (optional)
├── adas/
│   ├── perception/
│   │   └── road/
│   │       ├── road_v2.py       # RoadPerception (centerline extraction)
│   │       └── segmentation.py  # Morphological mask cleaning
│   └── control/
│       └── mpcv2.py             # CenterlineMPC controller
├── utils/
│   ├── image.py                 # letterbox(), unletterbox(), scale_boxes()
│   └── codec.py                 # msgpack encode/decode
├── weights/
│   ├── yolop/                   # YOLOPv2 model files
│   └── yolo/                    # YOLO26n model files
└── test/
    └── test_infer.py            # Benchmark demo (5-7 FPS)
```

---

## Performance

| Mode | FPS | Description |
|------|-----|-------------|
| Inference only | ~7 | No visualization, no video write |
| Demo mode | ~5 | With overlay drawing + AVI output |

**Tested on:** Intel Core i5 + Iris Xe GPU, 1080p input video

---

## Quick Start

### Prerequisites

- Python 3.12+
- Intel CPU/GPU with OpenVINO support
- `uv` package manager

### Install

```bash
# Clone and enter directory
cd autonomy-server

# Install dependencies
uv sync
```

### Run Demo

```bash
# Benchmark mode (no visualization)
uv run python -m src.test.test_infer

# Demo mode (with debug video output)
uv run python -m src.test.test_infer --demo --morph
```

### Start Server

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## WebSocket Protocol

### Client → Server (SensorMessage)

```json
{
  "type": "sensor",
  "payload": {
    "timestamp": 1706745600.123,
    "image": "<JPEG bytes>",
    "gps": {"lat": 37.7749, "lon": -122.4194}
  }
}
```

### Server → Client (AutonomyMessage)

```json
{
  "type": "autonomy",
  "payload": {
    "trajectory": [[320, 400], [318, 350], ...],
    "control": {"steeringAngle": -0.05, "confidence": 0.85},
    "status": "NORMAL"
  }
}
```

---

## License

MIT License

## Acknowledgments

- [YOLOPv2](https://github.com/CAIC-AD/YOLOPv2) — Panoptic driving perception
- [OpenVINO](https://docs.openvino.ai/) — Intel inference toolkit
