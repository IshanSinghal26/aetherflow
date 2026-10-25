# AetherFlow — Comprehensive Telemetry & Metrics Guide

This document provides an exhaustive reference for every physiological, kinematic, and environmental metric monitored and actuated by the **AetherFlow** neuroadaptive engine.

---

## Table of Contents
1. [Ocular & Facial Biometrics](#1-ocular--facial-biometrics)
   - [Eye Aspect Ratio (EAR)](#11-eye-aspect-ratio-ear)
   - [Blink Frequency (BPM)](#12-blink-frequency-bpm)
   - [PERCLOS (Percentage of Eye Closure)](#13-perclos-percentage-of-eye-closure)
   - [Mean Blink Duration](#14-mean-blink-duration)
   - [Gaze Stability Index](#15-gaze-stability-index)
   - [Head Pose Orientation (Pitch, Yaw, Roll)](#16-head-pose-orientation-pitch-yaw-roll)
   - [Face Presence Ratio](#17-face-presence-ratio)
2. [Peripheral Interaction Kinematics](#2-peripheral-interaction-kinematics)
   - [Keystroke Dwell Time](#21-keystroke-dwell-time)
   - [Keystroke Flight Time](#22-keystroke-flight-time)
   - [Keystrokes Per Minute (KPM) & WPM](#23-keystrokes-per-minute-kpm--wpm)
   - [Typing Burstiness](#24-typing-burstiness)
   - [Typing Pause Ratio](#25-typing-pause-ratio)
   - [Cursor Velocity & Peak Speed](#26-cursor-velocity--peak-speed)
   - [Mouse Jitter Index & Path Tortuosity](#27-mouse-jitter-index--path-tortuosity)
   - [Click & Scroll Frequency](#28-click--scroll-frequency)
   - [Peripheral Inactivity (Idle Seconds)](#29-peripheral-inactivity-idle-seconds)
3. [Sliding Window Dynamics & Temporal Trends](#3-sliding-window-dynamics--temporal-trends)
   - [EAR Trend (Eyelid Drooping Slope)](#31-ear-trend-eyelid-drooping-slope)
   - [Jitter Trend (Frustration Slope)](#32-jitter-trend-frustration-slope)
   - [Activity Level](#33-activity-level)
4. [Creative Context Domains](#4-creative-context-domains)
5. [Workspace Adaptation Parameters](#5-workspace-adaptation-parameters)
   - [Screen Warmth (Kelvin)](#51-screen-warmth-kelvin)
   - [UI Density Modes](#52-ui-density-modes)
   - [Notification Policies](#53-notification-policies)
   - [Ergonomic Advisories](#54-ergonomic-advisories)

---

## 1. Ocular & Facial Biometrics

Captured non-invasively via standard RGB webcam at 30 FPS using MediaPipe FaceLandmarker (478 landmarks).

### 1.1 Eye Aspect Ratio (EAR)
- **Unit**: Dimensionless scalar (typically `0.10` to `0.40`).
- **Formula**:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
  Where $p_1, p_4$ are outer/inner eye corners and $(p_2, p_6)$ and $(p_3, p_5)$ are vertical eyelid landmark pairs.
- **Physiological Meaning**: Quantifies the vertical aperture of the eye opening relative to horizontal span. Invariant to camera distance.
- **Reference Ranges**:
  - `0.26 - 0.35`: Wide open, engaged wakefulness.
  - `< 0.21`: Eye closure threshold (classified as a blink).
  - `< 0.18`: Heavy eyelid drooping / partial ptosis indicating drowsiness.
- **How AetherFlow Uses It**: Detects blinks, micro-sleeps, and gradual eyelid closure trends over time.

---

### 1.2 Blink Frequency (BPM)
- **Unit**: Blinks per minute (`BPM`).
- **Calculation**: Count of completed blinks ($60\text{ ms} \le \text{duration} \le 500\text{ ms}$) within a 60-second sliding window.
- **Physiological Meaning**: Spontaneous eyeblink rate reflects dopaminergic activity in the striatum and cognitive demand.
- **Reference Ranges**:
  - `10 - 15 BPM` (Low): **Flow State / Deep Immersion** (blink rate drops during intense visual focus).
  - `16 - 22 BPM` (Baseline): Normal resting task engagement.
  - `24 - 38 BPM` (High): **Cognitive Overload** (anxiety, task switching) or compensatory response to ocular fatigue.
- **How AetherFlow Uses It**: Differentiates between calm, deep immersion (low blinks) and stressful overload (flurries of rapid blinks).

---

### 1.3 PERCLOS (Percentage of Eye Closure)
- **Unit**: Percentage / Ratio (`0.0%` to `100.0%`).
- **Formula**:
  $$\text{PERCLOS} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{EAR}_i < 0.20)$$
- **Physiological Meaning**: Gold-standard physiological fatigue biomarker recognized by NHTSA and chronobiology research. Measures the proportion of time over the last 60 seconds that eyes were $\ge 80\%$ closed.
- **Reference Ranges**:
  - `0.0% - 5.0%`: Alert and refreshed.
  - `5.0% - 12.0%`: Normal prolonged reading.
  - `> 15.0%`: **Onset of Fatigue / Drowsiness**.
  - `> 25.0%`: **Severe Exhaustion / Micro-sleep episodes**.
- **How AetherFlow Uses It**: Primary trigger for **FATIGUE** state, shifting display warmth to warm amber/candlelight (3000K) and recommending breaks.

---

### 1.4 Mean Blink Duration
- **Unit**: Milliseconds (`ms`).
- **Measurement**: Duration elapsed between EAR crossing below $0.21$ and recovering above $0.21$.
- **Physiological Meaning**: Reflexive blinks are crisp and rapid. As the central nervous system tires, levator palpebrae superioris muscle tone drops, slowing down the blink.
- **Reference Ranges**:
  - `100 - 180 ms`: Alert, snappy blinks.
  - `180 - 250 ms`: Moderate fatigue.
  - `> 300 ms`: **Sluggish blinks / micro-sleep warning**.
- **How AetherFlow Uses It**: Corroborates PERCLOS to verify fatigue without triggering on fast, intentional double-blinks.

---

### 1.5 Gaze Stability Index
- **Unit**: Normalized score from `0.0` (erratic dispersion) to `1.0` (rock steady).
- **Calculation**: Inverse standard deviation of iris center coordinates relative to eye cantus landmarks over a 5-second window:
  $$\text{Stability} = \text{clip}\left(1.0 - 8.0 \cdot (\sigma_{\text{iris\_x}} + \sigma_{\text{iris\_y}}), 0.0, 1.0\right)$$
- **Physiological Meaning**: Fixation stability. During deep reading or code editing, saccades are orderly and gaze remains focused. Under mental strain, disorientation, or fatigue, eyes wander erratically.
- **Reference Ranges**:
  - `> 0.85`: High fixation focus (typical of **FLOW**).
  - `0.65 - 0.85`: Standard reading / scanning.
  - `< 0.50`: Saccadic jitter, visual confusion, or distracted scanning (**COGNITIVE_OVERLOAD**).
- **How AetherFlow Uses It**: Distinguishes between rhythmic coding flow and frantic searching.

---

### 1.6 Head Pose Orientation (Pitch, Yaw, Roll)
- **Unit**: Degrees ($^\circ$).
- **Calculation**: 3D perspective-n-point projection (`cv2.solvePnP`) using 6 canonical facial points (nose tip, chin, eye outer corners, mouth corners).
  - **Pitch**: Nodding up ($+$) / down ($-$). Negative pitch indicates head slumping toward the chest (slouching/drowsiness).
  - **Yaw**: Turning left / right.
  - **Roll**: Tilting ear toward shoulder.
- **Physiological Meaning**: Postural ergonomics and physical attention direction.
- **Reference Ranges**:
  - `|Pitch| < 15^\circ, |Yaw| < 20^\circ`: Centered at monitor.
  - `Pitch < -25^\circ`: Severe head slump (ergonomic strain / fatigue).

---

### 1.7 Face Presence Ratio
- **Unit**: Ratio (`0.0` to `1.0`).
- **Calculation**: Percentage of frames in the 60s window where a human face was successfully resolved by the detector.
- **How AetherFlow Uses It**: When $< 0.10$ and peripherals are inactive, immediately transitions state to **IDLE** without unnecessary inference latency.

---

## 2. Peripheral Interaction Kinematics

Captured via low-level OS event hooks (`pynput`). **STRICT ZERO-KEYLOGGING POLICY**: Keys are never decoded to characters, strings, or text. Only timestamps ($t$) and physical durations ($\Delta t$) are stored.

### 2.1 Keystroke Dwell Time
- **Unit**: Milliseconds (`ms`).
- **Formula**:
  $$\Delta t_{\text{dwell}} = t_{\text{key\_up}} - t_{\text{key\_down}}$$
- **Biomechanical Meaning**: How long a finger holds down a physical key. Highly sensitive to neuromuscular tension and fatigue.
- **Reference Ranges**:
  - `85 - 110 ms`: Crisp, fluid, confident typing (**FLOW**).
  - `110 - 135 ms`: Normal baseline typing.
  - `> 150 ms`: Sluggish motor execution caused by physical or mental fatigue.

---

### 2.2 Keystroke Flight Time
- **Unit**: Milliseconds (`ms`).
- **Formula**:
  $$\Delta t_{\text{flight}} = t_{\text{key\_down}}^{(i)} - t_{\text{key\_up}}^{(i-1)}$$
- **Cognitive Meaning**: Latency between releasing one key and pressing the next. Represents the cognitive planning interval for sequential motor chunking.
- **Reference Ranges**:
  - `130 - 200 ms` (Low variance): Rhythmic, subconscious prose/code generation.
  - `200 - 350 ms`: Standard composed typing.
  - `> 450 ms` (High variance): Hesitation, debugging difficulty, search latency, or drowsiness.

---

### 2.3 Keystrokes Per Minute (KPM) & WPM
- **Unit**: Keystrokes per minute (`KPM`) and Words per minute ($\text{WPM} \approx \text{KPM} / 5$).
- **Calculation**: Rolling count of keypresses normalized across the active window:
  $$\text{KPM} = \frac{\text{Keystrokes}}{\Delta t_{\text{minutes}}}$$
- **Cognitive Meaning**: Gross input cadence and productivity throughput.
- **Reference Ranges**:
  - `> 200 KPM (> 40 WPM)`: Sustained high-throughput flow.
  - `80 - 180 KPM (16 - 36 WPM)`: Intermittent thinking and writing.
  - `< 40 KPM (< 8 WPM)`: Reading, passive browsing, or idle stalling.

---

### 2.4 Typing Burstiness
- **Unit**: Dimensionless coefficient of variation ($CV$).
- **Formula**:
  $$\text{Burstiness} = \frac{\sigma_{\text{IKI}}}{\mu_{\text{IKI}}}$$
  Where $\text{IKI}$ are Inter-Keystroke Intervals ($t_{\text{down}}^{(i)} - t_{\text{down}}^{(i-1)}$).
- **Cognitive Meaning**:
  - Low $CV$ ($< 0.70$): Metronomic, uniform typing cadence (**FLOW**).
  - High $CV$ ($> 1.50$): Choppy, fragmented bursts with abrupt stops (**COGNITIVE_OVERLOAD** or frustration).

---

### 2.5 Typing Pause Ratio
- **Unit**: Ratio (`0.0` to `1.0`).
- **Calculation**: Fraction of time in the sliding window where no typing occurred.
- **Cognitive Meaning**: Distinguishes continuous prose/code generation from reading or contemplation.

---

### 2.6 Cursor Velocity & Peak Speed
- **Unit**: Pixels per second (`px/s`).
- **Formula**:
  $$v = \frac{\sqrt{\Delta x^2 + \Delta y^2}}{\Delta t}$$
- **Kinematic Meaning**: Speed of hand movement across the mousepad.
  - Moderate, smooth velocity ($120 - 220\text{ px/s}$): Purposeful navigation.
  - Violent spikes ($> 800\text{ px/s}$): Frantic flailing or sudden erratic gestures.
  - Low velocity ($< 50\text{ px/s}$): Lethargic hand movements or reading.

---

### 2.7 Mouse Jitter Index & Path Tortuosity
- **Unit**: Normalized index from `0.000` (perfect straight line) to `1.000` (chaotic zigzag).
- **Formula**:
  $$\Delta \theta = |\theta_t - \theta_{t-1}|, \quad \text{Tortuosity} = \frac{\text{Actual Path Length}}{\text{Euclidean Chord Distance}}$$
  $$\text{Jitter Index} = \text{clip}\left(\frac{\bar{\Delta \theta}}{\pi} \cdot 0.6 + \frac{\min(3.0, \text{Tortuosity} - 1.0)}{3.0} \cdot 0.4, 0.0, 1.0\right)$$
- **Cognitive Meaning**: High-frequency micro-corrections and directional reversals. When a user is confused, searching for a misplaced button, experiencing mental overload, or nervous anxiety, mouse trajectories lose smoothness and exhibit elevated jitter.
- **Reference Ranges**:
  - `0.02 - 0.12`: Smooth, intentional strokes (**FLOW**).
  - `0.12 - 0.25`: Standard navigational usage.
  - `> 0.35`: **High Jitter — Cognitive Overload, agitation, or confusion**.

---

### 2.8 Click & Scroll Frequency
- **Unit**: Events per second (`Hz`).
- **Cognitive Meaning**:
  - High click rate ($> 0.8\text{ Hz}$): Rapid button clicking, debugging, or UI thrashing.
  - High scroll rate ($> 0.6\text{ Hz}$): Rapid skimming, documentation searching, or social feed scrolling.

---

### 2.9 Peripheral Inactivity (Idle Seconds)
- **Unit**: Seconds (`s`).
- **Calculation**: $\Delta t = t_{\text{current}} - \max(t_{\text{last\_key}}, t_{\text{last\_mouse}})$.
- **How AetherFlow Uses It**: If $> 10.0\text{s}$, peripheral status is marked `IS_IDLE`. If sustained alongside absent face, triggers the **IDLE** state.

---

## 3. Sliding Window Dynamics & Temporal Trends

Features calculated as first-order linear regression slopes ($\beta = \frac{d y}{d t}$) over the 60-second window.

### 3.1 EAR Trend (Eyelid Drooping Slope)
- **Formula**: Slope of $\text{EAR}$ vs. time over $60\text{s}$.
- **Physiological Meaning**:
  - Negative slope ($\beta < -0.002$): **Progressive eyelid drooping**. Indicates the user is actively fighting to stay awake.
  - Zero slope ($\beta \approx 0$): Stable eyelid aperture.

### 3.2 Jitter Trend (Frustration Slope)
- **Formula**: Slope of Mouse Jitter vs. time over $60\text{s}$.
- **Cognitive Meaning**:
  - Positive slope ($\beta > +0.002$): Escalating trajectory chaos, signaling mounting frustration or cognitive overload.

### 3.3 Activity Level
- **Formula**: $1.0 - \text{idle\_ratio}$.
- **Meaning**: Overall temporal density of user interactions ($0.0 = \text{completely away}$, $1.0 = \text{continuous interaction}$).

---

## 4. Creative Context Domains

AetherFlow detects the foreground application using native OS APIs and maps it to a creative domain:

| Domain | Example Processes | Creative Context Characteristics |
| :--- | :--- | :--- |
| **`CODE_IDE`** | `code.exe`, `pycharm.exe`, `cursor.exe`, `powershell.exe` | Deep syntax focus, high typing bursts, dense information packing. |
| **`CREATIVE_DESIGN`** | `photoshop.exe`, `figma.exe`, `blender.exe`, `illustrator.exe` | High visual precision, cursor dragging. **Color accuracy is critical**. |
| **`WRITING_TEXT`** | `winword.exe`, `obsidian.exe`, `notion.exe`, `slack.exe` | Prose rhythm, typography readability, distraction minimization. |
| **`MEDIA_VIDEO`** | `vlc.exe`, `spotify.exe`, YouTube/Netflix browser tabs | Passive viewing/listening, cinematic contrast. |
| **`RESEARCH_BROWSING`** | `chrome.exe`, `msedge.exe`, `firefox.exe`, `brave.exe` | Documentation reading, multi-tab scanning. |
| **`SYSTEM_UTILITY`** | `explorer.exe`, `taskmgr.exe` | Quick OS file/process operations. |
| **`GENERAL`** | Unmapped applications | Balanced baseline workspace. |

---

## 5. Workspace Adaptation Parameters

### 5.1 Screen Warmth (Kelvin)
Display color temperature modulated via Win32 `SetDeviceGammaRamp` using the Planckian blackbody locus algorithm:

```text
 2700K ──────── 3200K ──────── 4200K ──────── 5000K ──────── 6500K
Candlelight      Amber       Paper Warm     Neutral       Daylight
(Deep Rest)    (Fatigue)    (Focus/Flow)   (Standard)    (True Color)
```

- **6500 K (Daylight Neutral)**:
  - Preserved strictly in `CREATIVE_DESIGN` to ensure graphic designers, video colorists, and UI artists never experience color distortion.
  - Baseline for `IDLE` standby.
- **4200 K – 5000 K (Paper Warmth)**:
  - Actuated during `FLOW` and `NORMAL_FOCUS` in `CODE_IDE` and `WRITING_TEXT`. Attenuates high-energy blue light to prevent digital eye strain while maintaining contrast.
- **3000 K – 3500 K (Soothing Amber)**:
  - Actuated during `COGNITIVE_OVERLOAD` and `FATIGUE`. Significantly softens display luminescence to relieve eye strain and mental agitation.

---

### 5.2 UI Density Modes
- **`COMPACT`**: Maximizes visible lines of code and canvas real estate. Ideal for `FLOW` where the user holds large mental models in working memory.
- **`STANDARD`**: Balanced default UI spacing and typography.
- **`COMFORT`**: Slightly enlarged text, relaxed line-height, and padding. Reduces visual clutter during `COGNITIVE_OVERLOAD`.
- **`RELAXED`**: High padding and enlarged elements. Minimizes visual hunting during `FATIGUE`.

---

### 5.3 Notification Policies
- **`TOTAL_SILENCE`**: Windows Focus Assist Alarms-Only. Zero popups, sounds, or banners permitted. Actuated during `FLOW` to shield deep focus.
- **`PRIORITY_ONLY`**: Suppresses marketing, social, and routine notifications while allowing urgent calendar meetings and priority messages.
- **`ALLOW_ALL`**: Unfiltered notification delivery. Active during `FATIGUE` (so the user does not miss break alarms) and `IDLE`.

---

### 5.4 Ergonomic Advisories
- **`NONE`**: User is in healthy cognitive equilibrium.
- **`GENTLE_PROMPT`**: Mild cognitive overload or extended concentration detected. Recommends pacing and slow breathing.
- **`MANDATORY_BREAK`**: Significant ocular fatigue ($PERCLOS > 0.15$ or sluggish blinks) sustained over multiple minutes. Recommends a 5-minute 20-20-20 rule rest cycle (look 20 feet away for 20 seconds).

---

## Summary Reference Table

| Metric | Normal Alert Range | Flow Indicator | Overload Indicator | Fatigue Indicator |
| :--- | :--- | :--- | :--- | :--- |
| **EAR** | `0.26 - 0.35` | `0.28 - 0.34` (steady) | `0.25 - 0.32` (variable) | `< 0.22` (drooping) |
| **Blink Rate** | `15 - 22 BPM` | `10 - 15 BPM` | `24 - 38 BPM` | `18 - 30 BPM` (slow) |
| **PERCLOS** | `< 6%` | `< 4%` | `5 - 10%` | `> 15%` |
| **Blink Duration**| `120 - 180 ms` | `120 - 160 ms` | `130 - 200 ms` | `> 280 ms` |
| **Gaze Stability**| `70 - 85%` | `> 88%` | `< 50%` | `40 - 65%` |
| **Dwell Time** | `100 - 130 ms` | `90 - 110 ms` (crisp) | `110 - 140 ms` | `> 150 ms` (sluggish) |
| **Flight Time** | `200 - 320 ms` | `140 - 200 ms` (rhythmic) | `250 - 500 ms` (erratic) | `> 450 ms` |
| **Mouse Jitter** | `0.08 - 0.18` | `< 0.10` (calm) | `> 0.35` (chaotic) | `0.15 - 0.25` (slow) |
| **Typing Cadence**| `100 - 180 KPM`| `> 200 KPM` | `Choppy bursts` | `< 60 KPM` |
