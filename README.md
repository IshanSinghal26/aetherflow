# AetherFlow — Local Neuroadaptive Workspace Engine

> **A 100% local, closed-loop workspace engine that passively monitors user cognitive states and dynamically adapts the desktop environment (screen warmth, UI density, and notification suppression) based on active creative context.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero Cloud APIs](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)]()
[![Tests: 30 Passed](https://img.shields.io/badge/Tests-30%20Passed-brightgreen.svg)]()

---

## 1. Architectural Overview

AetherFlow runs entirely offline without cloud telemetry, paid APIs, or biometric sensors. It fuses commodity RGB webcam video with standard keyboard and mouse kinematic telemetry to estimate cognitive states and actuate the desktop environment in a real-time closed loop.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LAYER 1: PASSIVE INGESTION                      │
│                                                                        │
│   Webcam Feed (30 FPS)                     Peripheral Hooks (pynput)   │
│   ├── MediaPipe FaceLandmarker             ├── Anonymous Dwell Time    │
│   ├── Eye Aspect Ratio (EAR)               ├── Keystroke Flight Time   │
│   ├── Blink Rate (BPM) & PERCLOS           ├── Cadence (KPM / WPM)     │
│   └── Gaze Stability & Iris Offset         └── Mouse Jitter & Velocity │
└────────────────────┬───────────────────────────────────┬───────────────┘
                     │                                   │
                     ▼                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               LAYER 2: 60s SLIDING WINDOW FEATURE BUFFER               │
│                                                                        │
│   • Ring buffer accumulating synchronized temporal telemetry           │
│   • Statistical moments: Mean, Std, Min, Max, Percentiles (p25, p75)   │
│   • First-order linear trends (Eyelid drooping slope, Jitter slope)    │
│   • Normalization via pre-fitted StandardScaler (29 Feature Vector)    │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                LAYER 3: ML COGNITIVE STATE CLASSIFIER                  │
│                                                                        │
│   • Supervised Ensemble (RandomForestClassifier, calibrated proba)     │
│   • States: FLOW | NORMAL_FOCUS | COGNITIVE_OVERLOAD | FATIGUE | IDLE  │
│   • Explainability: Dominant biometric/behavioral markers extracted    │
└──────────────────┬─────────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│   LAYER 3 INFERENCE: State (FLOW)    │  │  LAYER 4: CONTEXT DETECTOR   │
│   • Confidence: 92%                  │  │  • Win32 / Linux OS Query    │
│   • Markers: Gaze=0.92, Jitter=0.06  │  │  • Active: Code.exe (IDE)    │
└──────────────────┬───────────────────┘  └──────────────┬───────────────┘
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     LAYER 5: WORKSPACE EXECUTION                       │
│                                                                        │
│   Profile Matrix: (CognitiveState x ContextDomain) -> Intent           │
│   ├── Screen Warmth: Smooth Gamma Ramp stepping (2700K - 6500K)        │
│   ├── UI Density Mode: Compact / Standard / Comfort / Relaxed          │
│   ├── Notification Policy: Total Silence / Priority Only / Allow All   │
│   ├── Ergonomic Guidance: Gentle Pacing / Mandatory Break alerts       │
│   └── Safeguards: Hysteresis debouncing & automatic display rollback   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Theoretical Principles & Mathematical Formulations

### 2.1 Eye Aspect Ratio (EAR) & PERCLOS
Ocular fatigue and micro-sleep events are quantified via landmark distances across 6 cardinal eye coordinates:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- **Open Eye**: $\text{EAR} \approx 0.25 - 0.35$.
- **Blink**: $\text{EAR} < 0.21$ with duration $60\text{ ms} \le \Delta t \le 500\text{ ms}$.
- **PERCLOS**: Percentage of eyelid closure over the 60-second window:
  $$\text{PERCLOS} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{EAR}_i < \tau_{\text{perclos}})$$
  Elevated PERCLOS ($> 0.20$) strongly correlates with ocular fatigue and cognitive exhaustion.

### 2.2 Gaze Stability & Head Pose
Iris coordinates ($p_{468}, p_{473}$) are normalized relative to the inner and outer eye cantus span. Gaze stability is defined as the inverse dispersion over sliding frames:

$$\text{Stability} = \text{clip}\left(1.0 - 8.0 \cdot \left(\sigma_x + \sigma_y\right), 0.0, 1.0\right)$$

### 2.3 Keyboard Cadence & Interaction Kinematics
- **Dwell Time**: Duration between key press and key release:
  $$\Delta t_{\text{dwell}} = t_{\text{release}}^{(i)} - t_{\text{press}}^{(i)}$$
- **Flight Time**: Latency between consecutive keystroke transitions:
  $$\Delta t_{\text{flight}} = t_{\text{press}}^{(i)} - t_{\text{release}}^{(i-1)}$$
- **Typing Burstiness**: Normalized variability (coefficient of variation) of inter-keystroke intervals (IKIs). Fragmented, irregular typing bursts indicate cognitive hesitation or task switching.

### 2.4 Mouse Jitter & Path Tortuosity
Rapid directional fluctuations and hesitation are captured through micro-heading angular changes and path tortuosity:

$$\Delta \theta = |\theta_t - \theta_{t-1}|, \quad \text{Tortuosity} = \frac{\sum_{i=1}^n \|p_i - p_{i-1}\|}{\|p_n - p_0\| + \epsilon}$$

$$\text{Jitter Index} = \text{clip}\left(\frac{\bar{\Delta \theta}}{\pi} \cdot 0.6 + \frac{\min(3.0, \text{Tortuosity} - 1.0)}{3.0} \cdot 0.4, 0.0, 1.0\right)$$

High jitter ($> 0.35$) reflects cognitive overload, agitation, or visual disorientation.

### 2.5 Screen Warmth & Planckian Blackbody Radiance
Screen color temperature (Kelvin) is mapped to RGB multipliers using the Tanner-Helland formulation of Planckian locus radiation:

- For $T \le 6600\text{ K}$:
  $$R = 255, \quad G = 99.47 \cdot \ln(T/100) - 161.12, \quad B = 138.52 \cdot \ln((T/100) - 10) - 305.04$$
- The RGB scaling factor is normalized against $6500\text{ K}$ neutral daylight and interpolated smoothly across 12 discrete gamma ramp updates to prevent visual flickering.

---

## 3. Adaptation Matrix: State × Context

| Creative Domain | Cognitive State | Screen Warmth | UI Density | Notification Policy | Ergonomic Advisory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CODE_IDE** | `FLOW` | **4200 K** | `COMPACT` | `TOTAL_SILENCE` | `NONE` |
| **CODE_IDE** | `NORMAL_FOCUS` | **5000 K** | `STANDARD` | `PRIORITY_ONLY` | `NONE` |
| **CODE_IDE** | `COGNITIVE_OVERLOAD` | **3500 K** | `COMFORT` | `PRIORITY_ONLY` | `GENTLE_PROMPT` |
| **CODE_IDE** | `FATIGUE` | **3000 K** | `RELAXED` | `ALLOW_ALL` | `MANDATORY_BREAK` |
| **CREATIVE_DESIGN** | `FLOW` | **6500 K** *(Protected)* | `STANDARD` | `TOTAL_SILENCE` | `NONE` |
| **CREATIVE_DESIGN** | `COGNITIVE_OVERLOAD` | **6000 K** *(Subtle)* | `COMFORT` | `PRIORITY_ONLY` | `GENTLE_PROMPT` |
| **CREATIVE_DESIGN** | `FATIGUE` | **5500 K** | `RELAXED` | `ALLOW_ALL` | `MANDATORY_BREAK` |
| **WRITING_TEXT** | `FLOW` | **4000 K** (Parchment) | `COMFORT` | `TOTAL_SILENCE` | `NONE` |
| **WRITING_TEXT** | `COGNITIVE_OVERLOAD` | **3400 K** | `COMFORT` | `PRIORITY_ONLY` | `GENTLE_PROMPT` |
| **WRITING_TEXT** | `FATIGUE` | **2900 K** (Candle) | `RELAXED` | `ALLOW_ALL` | `MANDATORY_BREAK` |
| **MEDIA_VIDEO** | `FLOW` | **5800 K** (Cinema) | `STANDARD` | `TOTAL_SILENCE` | `NONE` |
| **MEDIA_VIDEO** | `FATIGUE` | **3200 K** | `RELAXED` | `ALLOW_ALL` | `GENTLE_PROMPT` |
| **ANY** | `IDLE` | **6500 K** | `STANDARD` | `ALLOW_ALL` | `NONE` |

> **Critical Design Rule**: In **Creative Design** applications (Photoshop, Illustrator, Figma, Blender), color accuracy is preserved at **6500 K true daylight** to ensure visual art and UI palettes are never distorted.

---

## 4. Privacy & Hardware Safeguards

1. **Zero Keylogging Guarantee**:
   The telemetry system hooks only key press and release timestamps (`time.time()`). No key names, characters, strings, or text are ever read, stored, or processed.
2. **Local Computer Vision Only**:
   Webcam frames are analyzed frame-by-frame entirely in memory on the CPU via MediaPipe FaceLandmarker. No video frames or face images are ever saved to disk or transmitted across any network interface.
3. **Display Restoration Safeguard**:
   Original hardware gamma ramps are backed up to memory on engine initialization. `atexit` handlers and OS signal traps (`SIGINT`, `SIGTERM`) ensure that the display is guaranteed to revert to its pristine hardware gamma ramp upon program termination.
4. **Hysteresis Filtering**:
   A state must be continuously sustained across multiple evaluation ticks before hardware gamma shifts are actuated, preventing jarring visual flapping.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Standard RGB webcam
- Windows 10/11 (for native gamma ramp actuation; cross-platform mock/dry-run for Linux/macOS)

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Usage Guide

### 1. Pre-Train the Cognitive Classifier
Generate synthetic physiological and behavioral telemetry, evaluate classification metrics, and save serialized `.joblib` model artifacts to `data/`:

```bash
python -m aetherflow.simulation.train_synthetic --samples 800
```

### 2. Launch AetherFlow Live Engine
Start the full neuroadaptive closed loop with active webcam and peripheral tracking:

```bash
python -m aetherflow.main
```

### 3. Command-Line Options
| Flag | Description |
| :--- | :--- |
| `--dry-run` | Run in simulation mode without writing to hardware display gamma ramp |
| `--synthetic-sensing` | Generate simulated biometric telemetry (for headless CI or machines without webcam) |
| `--camera-id <ID>` | Specify webcam index (default: `0`) |
| `--window-size <SEC>` | Sliding feature aggregation window in seconds (default: `60.0`) |
| `--interval <SEC>` | Main loop tick interval in seconds (default: `1.0`) |
| `--iterations <N>` | Run for $N$ iterations and cleanly exit (useful for automated jobs) |
| `--train` | Automatically pre-train model before starting the engine |
| `--debug` | Output detailed internal engine diagnostics |

**Example (Headless / Safe Simulation):**
```bash
python -m aetherflow.main --dry-run --synthetic-sensing --iterations 10
```

---

## 7. Automated Testing Suite

The repository includes a comprehensive unit and integration test suite covering all layers:

```bash
pytest -v
```

### Test Coverage Highlights:
- **`tests/test_sensing.py`**: EAR geometry, open vs. closed eye ratios, blink state machine, PERCLOS, keystroke dwell/flight time injection, mouse jitter, privacy guarantees.
- **`tests/test_feature_buffer.py`**: Sliding window temporal pruning, 29-feature moment extraction, trend linear regression, StandardScaler normalization.
- **`tests/test_classifier.py`**: Heuristic rules, machine learning inference, probability normalization, dominant feature explainability.
- **`tests/test_context.py`**: Windows process mapping, browser tab keyword heuristics, cross-platform fallbacks.
- **`tests/test_adaptation.py`**: Tanner-Helland Kelvin-to-RGB conversion, creative design color protection, hysteresis debouncing, display rollback.
- **`tests/test_end_to_end.py`**: Full closed loop orchestration cycle with analytics summary reporting.

---

## 8. Directory Structure

```text
aetherflow/
│
├── aetherflow/
│   ├── __init__.py                   # Package exports and version metadata
│   ├── config.py                     # Centralized configurations, enums, and hyperparameters
│   │
│   ├── sensing/                      # Layer 1: Passive Ingestion
│   │   ├── __init__.py
│   │   ├── vision_tracker.py         # MediaPipe FaceLandmarker: EAR, Blink Rate, Gaze Stability
│   │   └── peripheral_tracker.py     # Keyboard cadence (flight/dwell time) & Mouse jitter
│   │
│   ├── engine/                       # Layer 2 & 3: Processing & ML Classification
│   │   ├── __init__.py
│   │   ├── feature_buffer.py         # 60s sliding window aggregator & vector normalizer
│   │   └── classifier.py             # Random Forest cognitive state classifier & explainability
│   │
│   ├── context/                      # Layer 4: Context Awareness
│   │   ├── __init__.py
│   │   └── app_detector.py           # Active foreground window detection & process classification
│   │
│   ├── adaptation/                   # Layer 5: Workspace Execution
│   │   ├── __init__.py
│   │   ├── executor.py               # Closed-loop actuator (Win32 gamma ramp, hysteresis, rollback)
│   │   └── profiles.py               # Action mapping matrix for IDE, Design, Writing, Media
│   │
│   ├── simulation/                   # Synthetic Data & Pre-training
│   │   ├── __init__.py
│   │   └── train_synthetic.py        # Telemetry generator, ML trainer, and evaluation reporter
│   │
│   └── main.py                       # Non-blocking orchestrator, event loop & console dashboard
│
├── data/                             # Serialized models and assets
│   ├── cognitive_classifier.joblib   # Trained RandomForest classifier
│   ├── scaler.joblib                 # Trained StandardScaler
│   └── face_landmarker.task          # MediaPipe FaceLandmarker model asset
│
├── tests/                            # Comprehensive automated pytest suite (30/30 passed)
│   ├── __init__.py
│   ├── test_sensing.py
│   ├── test_feature_buffer.py
│   ├── test_classifier.py
│   ├── test_context.py
│   ├── test_adaptation.py
│   └── test_end_to_end.py
│
├── requirements.txt                  # Pinned production dependencies
└── README.md                         # Architecture, mathematics, and operational documentation
```
