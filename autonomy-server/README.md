# YOLOPv2 ADAS Backend (PoC)

**A modular, hardware-optimized Autonomous Driving Assistance System (ADAS) backend**  
Built with **Python 3.12**, **FastAPI**, and **OpenVINO 2025**.

---

## 📖 Project Overview

This proof-of-concept implements a **monocular ADAS stack** designed to run efficiently on resource-constrained hardware (e.g., Intel Iris Xe or NVIDIA Jetson Nano) using a single front-facing camera.

The system **decouples AI inference** (OpenVINO on GPU) from **control logic** (CPU) via an asynchronous WebSocket server. This ensures real-time responsiveness even when model inference (~10 FPS) is slower than camera input (30 FPS).

### Core Capabilities

1. **Perception**  
   Lane line detection & drivable area segmentation using YOLOPv2.

2. **Sensor Fusion**  
   Combines lane masks with road boundaries for robust lane keeping.

3. **State Estimation**  
   Temporal smoothing of polynomial coefficients with **Kalman filtering**.

4. **Planning & Control**  
   Cost-function-based trajectory scoring (Dynamic Window Approach) fused with GPS-biased Pure Pursuit.

---

## 🏗️ System Architecture

The project uses **`uv`** for deterministic dependency management and follows a strict **"Producer-Consumer"** pattern to handle latency.

### Directory Structure

```text
adas-backend/
├── data/                   # Model files (yolopv2fp16.xml, .bin)
├── src/
│   ├── main.py             # Entry point: async "fresh frame" loop
│   ├── engine.py           # Hardware abstraction (OpenVINO singleton)
│   ├── models.py           # Pydantic schemas (frontend ↔ backend)
│   ├── adas/
│   │   ├── perception.py   # Vision logic: extraction → fusion → Kalman
│   │   ├── segmentation.py # Image post-processing: morphological cleaning
│   │   ├── control.py      # Pilot logic: candidate scoring & bicycle model
│   │   ├── tracking.py     # State memory: Kalman filter implementation
│   │   └── checkpoint.py   # Navigation: GPS route manager
│   └── utils/
│       └── image.py        # Coordinate adapters (letterbox, crop)
└── pyproject.toml          # Project metadata & dependencies
```

### Data Flow Pipeline

1. **WebSocket Receiver**  
   Buffers incoming JPEG frames using a **zero-latency overwrite queue** (keeps only the latest frame).

2. **Pre-processing (GPU)**  
   Normalization & color conversion via OpenVINO PrePostProcessor.

3. **Inference (GPU)**  
   YOLOPv2 → lane logits + drivable area logits.

4. **Post-processing (CPU)**  
   Morphological cleaning to repair holes in road mask (`segmentation.py`).

5. **Perception (CPU)**  
   Fuse lane & road points → fit polynomial (`perception.py`).  
   Smooth coefficients with Kalman filter (`tracking.py`).

6. **Control (CPU)**  
   Compute GPS bias (`checkpoint.py`).  
   Generate 15 candidate trajectories (`control.py`).  
   Score candidates using weighted cost function (safety, lane adherence, GPS goal).

---

## Design Decisions

| Decision                          | Problem                                                                 | Solution                                                                                   |
|-----------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| **"Fresh Frame" Pattern**         | Server processes ~10 FPS, camera sends 30 FPS → queue lag               | Single-element buffer: receiver overwrites latest frame; processor always uses newest     |
| **Kalman on Polynomial Coeffs**   | Smoothing full 640×480 mask is too expensive on CPU                     | Fit curve first (`ax² + bx + c`), then smooth only 3 coefficients with Kalman              |
| **Cost-based Control**            | Hard rules fail in edge cases (GPS says turn right, but road ends)      | Weighted cost function: W_road = 10.0 (safety), W_goal = 2.0 (GPS)                        |
| **Reverse Letterboxing**          | Model expects 640×640, real input is 640×480 (4:3)                      | Crop masks to native 4:3 immediately after inference; all logic uses native resolution    |

---

## Hardware & Optimization Roadmap

| Phase              | Hardware                          | Constraints                              | Optimizations / Next Steps                                                                 |
|--------------------|-----------------------------------|------------------------------------------|--------------------------------------------------------------------------------------------|
| **Phase 1** (Current) | Laptop (Intel Core i5 / Iris Xe)  | Shared RAM bandwidth → ~10-15 FPS        | FP16 model, OpenVINO throughput hint, NumPy-heavy processing                               |
| **Phase 2** (Deployment) | Jetson Nano 4GB                   | Weak ARM CPU, slow Python loops          | TensorRT engine, PyCUDA shared memory, Numba JIT, downsample masks to 160×120              |
| **Phase 3** (Future) | Desktop (RTX 3060+) / Drive Orin  | —                                        | BEV projection (IPM), end-to-end models (UniAD, Comma Supercombo), virtual lane projection |

---

## 🚀 Setup & Running

### Prerequisites

- Python 3.12+
- `uv` (recommended package manager)

```bash
# Install uv (if not already installed)
pip install uv
```

### Install Dependencies

```bash
# Sync virtual environment & dependencies
uv sync
```

### Prepare Model Files

1. Download the **YOLOPv2 FP16 OpenVINO model** (`yolopv2fp16.xml` and `.bin`).
2. Place both files in the `data/` directory.

### Start the Server

```bash
# Run with uvicorn (recommended)
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or for development with auto-reload
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### License

[MIT License](LICENSE)

### Acknowledgments

- [YOLOPv2](https://github.com/chenyuntc/YOLOPv2)
- OpenVINO Toolkit
- FastAPI & WebSockets

---
