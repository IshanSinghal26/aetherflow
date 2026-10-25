"""Vision sensing layer using MediaPipe FaceLandmarker and OpenCV.

Extracts Eye Aspect Ratio (EAR), Blink Rate, PERCLOS, Gaze Stability,
and head pose proxies for real-time passive cognitive state ingestion.
Strictly local execution without cloud transmission.
"""

from __future__ import annotations

import collections
import logging
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from aetherflow.config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class VisionSample:
    """Snapshot of biometric indicators derived from facial telemetry."""
    timestamp: float
    face_detected: bool = False
    ear_left: float = 0.0
    ear_right: float = 0.0
    ear_mean: float = 0.0
    is_blinking: bool = False
    blink_count_total: int = 0
    blink_rate_bpm: float = 0.0
    perclos: float = 0.0                     # Proportion of time eyes closed in window
    mean_blink_duration_ms: float = 0.0
    gaze_stability: float = 1.0              # 1.0 = perfectly steady, 0.0 = high saccades/drift
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0


def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_ear(landmarks, indices: list[int], img_w: int, img_h: int) -> float:
    """Compute Eye Aspect Ratio (EAR) from standard 6-point eye landmarks.
    
    indices: [p1, p2, p3, p4, p5, p6]
    p1, p4 are horizontal eye corners.
    p2, p6 and p3, p5 are vertical landmark pairs.
    """
    pts = [(landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in indices]
    p1, p2, p3, p4, p5, p6 = pts
    
    # Vertical distances
    d_v1 = euclidean_dist(p2, p6)
    d_v2 = euclidean_dist(p3, p5)
    
    # Horizontal distance
    d_h = euclidean_dist(p1, p4)
    
    if d_h < 1e-6:
        return 0.0
    return (d_v1 + d_v2) / (2.0 * d_h)


class VisionTracker:
    """Passive vision tracker ingesting RGB webcam frames to quantify cognitive fatigue and focus."""

    # Landmark indices for MediaPipe Face Mesh
    # Left eye: corners (33, 133), top/bottom pairs (160, 144) and (158, 153)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    # Right eye: corners (362, 263), top/bottom pairs (385, 380) and (387, 373)
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # Irises
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # 3D model points for head pose estimation
    MODEL_POINTS_3D = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye outer corner
        (225.0, 170.0, -135.0),   # Right eye outer corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ], dtype=np.float64)

    POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    def __init__(self, config: Optional[VisionConfig] = None) -> None:
        self.config = config or VisionConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # State buffers
        self._latest_sample = VisionSample(timestamp=time.time())
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._blink_timestamps: collections.deque[float] = collections.deque(maxlen=300)
        self._blink_durations_ms: collections.deque[float] = collections.deque(maxlen=100)
        self._ear_history: collections.deque[Tuple[float, float]] = collections.deque(maxlen=1800) # (timestamp, ear)
        self._gaze_positions: collections.deque[Tuple[float, float]] = collections.deque(maxlen=150) # (norm_x, norm_y)
        
        # Blink tracking state machine
        self._in_blink = False
        self._blink_start_time = 0.0
        self._total_blinks = 0
        
        # MediaPipe landmarker instance
        self._landmarker = None
        if not self.config.synthetic_mode:
            self._init_landmarker()

    def _init_landmarker(self) -> None:
        """Initialize MediaPipe FaceLandmarker task."""
        try:
            model_path = Path(self.config.model_asset_path)
            if not model_path.exists():
                logger.info(f"Downloading face landmarker model to {model_path}...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                urllib.request.urlretrieve(url, str(model_path))

            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker successfully initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe FaceLandmarker ({e}). Falling back to synthetic vision mode.")
            self.config.synthetic_mode = True

    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> VisionSample:
        """Process a single BGR frame, compute ocular metrics, and return a VisionSample."""
        t_now = timestamp or time.time()
        h, w = frame_bgr.shape[:2]

        if self._landmarker is None or self.config.synthetic_mode:
            return self._generate_synthetic_sample(t_now)

        import mediapipe as mp
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.debug(f"Landmark detection failed: {e}")
            return self._record_no_face_sample(t_now)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return self._record_no_face_sample(t_now)

        landmarks = result.face_landmarks[0]
        num_landmarks = len(landmarks)

        # 1. Calculate EAR for left and right eyes
        ear_left = calculate_ear(landmarks, self.LEFT_EYE_INDICES, w, h)
        ear_right = calculate_ear(landmarks, self.RIGHT_EYE_INDICES, w, h)
        ear_mean = (ear_left + ear_right) / 2.0

        # Store EAR history for PERCLOS
        self._ear_history.append((t_now, ear_mean))

        # 2. Blink detection state machine
        is_blinking = ear_mean < self.config.ear_blink_threshold
        if is_blinking and not self._in_blink:
            # Transition to blink
            self._in_blink = True
            self._blink_start_time = t_now
        elif not is_blinking and self._in_blink:
            # Transition out of blink
            self._in_blink = False
            duration_ms = (t_now - self._blink_start_time) * 1000.0
            if self.config.min_blink_duration_ms <= duration_ms <= self.config.max_blink_duration_ms:
                self._total_blinks += 1
                self._blink_timestamps.append(t_now)
                self._blink_durations_ms.append(duration_ms)

        # 3. Blink rate (BPM) over 60s sliding window
        cutoff_60s = t_now - 60.0
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff_60s:
            self._blink_timestamps.popleft()
        
        # Calculate blink rate per minute
        blink_rate_bpm = float(len(self._blink_timestamps))

        # 4. PERCLOS (% time eye closed below threshold in past 60s)
        while self._ear_history and self._ear_history[0][0] < cutoff_60s:
            self._ear_history.popleft()
        
        if self._ear_history:
            closed_samples = sum(1 for _, ear in self._ear_history if ear < self.config.ear_perclos_threshold)
            perclos = closed_samples / len(self._ear_history)
        else:
            perclos = 0.0

        # 5. Mean blink duration
        mean_blink_dur = float(np.mean(self._blink_durations_ms)) if self._blink_durations_ms else 180.0

        # 6. Gaze stability using iris displacement relative to eye corners
        gaze_stability = 1.0
        if num_landmarks > self.RIGHT_IRIS_CENTER:
            l_iris = landmarks[self.LEFT_IRIS_CENTER]
            l_corner1 = landmarks[self.LEFT_EYE_INDICES[0]]
            l_corner2 = landmarks[self.LEFT_EYE_INDICES[3]]
            
            # Normalized iris horizontal position within eye span [0.0, 1.0]
            denom = abs(l_corner2.x - l_corner1.x)
            if denom > 1e-4:
                rel_x = (l_iris.x - min(l_corner1.x, l_corner2.x)) / denom
                rel_y = l_iris.y
                self._gaze_positions.append((rel_x, rel_y))
            
            if len(self._gaze_positions) >= 10:
                xs = [p[0] for p in self._gaze_positions]
                ys = [p[1] for p in self._gaze_positions]
                dispersion = float(np.std(xs) + np.std(ys))
                # Map dispersion to stability index [0.0, 1.0]
                gaze_stability = float(np.clip(1.0 - (dispersion * 8.0), 0.0, 1.0))

        # 7. Head pose estimation
        pitch, yaw, roll = self._estimate_head_pose(landmarks, w, h)

        # 8. Render visual annotations on frame for dedicated window/GUI
        annotated = frame_bgr.copy()
        self._annotate_face(
            annotated, landmarks, w, h,
            is_blinking, ear_mean, blink_rate_bpm,
            gaze_stability, pitch, yaw, roll
        )

        sample = VisionSample(
            timestamp=t_now,
            face_detected=True,
            ear_left=round(ear_left, 4),
            ear_right=round(ear_right, 4),
            ear_mean=round(ear_mean, 4),
            is_blinking=is_blinking,
            blink_count_total=self._total_blinks,
            blink_rate_bpm=round(blink_rate_bpm, 2),
            perclos=round(perclos, 4),
            mean_blink_duration_ms=round(mean_blink_dur, 2),
            gaze_stability=round(gaze_stability, 4),
            head_pitch=round(pitch, 2),
            head_yaw=round(yaw, 2),
            head_roll=round(roll, 2),
        )

        with self._lock:
            self._latest_sample = sample
            self._latest_raw_frame = frame_bgr
            self._latest_annotated_frame = annotated

        return sample

    def _estimate_head_pose(self, landmarks, w: int, h: int) -> Tuple[float, float, float]:
        """Estimate head pitch, yaw, and roll using solvePnP."""
        try:
            image_points = np.array([
                (landmarks[idx].x * w, landmarks[idx].y * h)
                for idx in self.POSE_LANDMARK_INDICES
            ], dtype=np.float64)

            focal_length = w
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))

            success, rot_vec, trans_vec = cv2.solvePnP(
                self.MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0

            rot_mat, _ = cv2.Rodrigues(rot_vec)
            # Extract Euler angles
            sy = math.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])
            singular = sy < 1e-6
            if not singular:
                pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
                yaw = math.atan2(-rot_mat[2, 0], sy)
                roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
                yaw = math.atan2(-rot_mat[2, 0], sy)
                roll = 0.0

            return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
        except Exception:
            return 0.0, 0.0, 0.0

    def _record_no_face_sample(self, timestamp: float) -> VisionSample:
        """Record sample when no face is present in the frame."""
        sample = VisionSample(
            timestamp=timestamp,
            face_detected=False,
            ear_left=0.0,
            ear_right=0.0,
            ear_mean=0.0,
            is_blinking=False,
            blink_count_total=self._total_blinks,
            blink_rate_bpm=float(len(self._blink_timestamps)),
            perclos=0.0,
            mean_blink_duration_ms=0.0,
            gaze_stability=0.0,
            head_pitch=0.0,
            head_yaw=0.0,
            head_roll=0.0,
        )
        with self._lock:
            self._latest_sample = sample
        return sample

    def _generate_synthetic_sample(self, timestamp: float) -> VisionSample:
        """Generate realistic synthetic vision sample for headless CI or fallback mode."""
        # Simulated breathing / micro-saccades
        t = timestamp
        ear = 0.28 + 0.03 * math.sin(t * 0.5)
        # Random occasional blink
        is_blink = (int(t * 10) % 40) == 0
        if is_blink:
            ear = 0.12
            self._total_blinks += 1
            self._blink_timestamps.append(t)
            self._blink_durations_ms.append(150.0)

        cutoff_60s = t - 60.0
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff_60s:
            self._blink_timestamps.popleft()

        synth_frame = self._draw_synthetic_avatar(
            t, ear, is_blink, float(max(8, len(self._blink_timestamps))), 0.88
        )

        sample = VisionSample(
            timestamp=t,
            face_detected=True,
            ear_left=round(ear, 4),
            ear_right=round(ear, 4),
            ear_mean=round(ear, 4),
            is_blinking=is_blink,
            blink_count_total=self._total_blinks,
            blink_rate_bpm=round(float(max(8, len(self._blink_timestamps))), 2),
            perclos=0.04,
            mean_blink_duration_ms=160.0,
            gaze_stability=0.88,
            head_pitch=0.0,
            head_yaw=0.0,
            head_roll=0.0,
        )
        with self._lock:
            self._latest_sample = sample
            self._latest_raw_frame = synth_frame
            self._latest_annotated_frame = synth_frame
        return sample

    def _annotate_face(
        self,
        frame: np.ndarray,
        landmarks,
        w: int,
        h: int,
        is_blinking: bool,
        ear: float,
        blink_rate: float,
        gaze_stab: float,
        pitch: float,
        yaw: float,
        roll: float
    ) -> None:
        """Draw eye contours, irises, head pose axis, and biometric telemetry on frame."""
        eye_color = (0, 0, 255) if is_blinking else (0, 255, 128)
        thickness = 2 if is_blinking else 1

        for eye_indices in [self.LEFT_EYE_INDICES, self.RIGHT_EYE_INDICES]:
            pts = np.array([
                [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
                for idx in eye_indices
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=eye_color, thickness=thickness)

        if len(landmarks) > self.RIGHT_IRIS_CENTER:
            l_iris = landmarks[self.LEFT_IRIS_CENTER]
            r_iris = landmarks[self.RIGHT_IRIS_CENTER]
            cv2.circle(frame, (int(l_iris.x * w), int(l_iris.y * h)), 3, (255, 0, 255), -1)
            cv2.circle(frame, (int(r_iris.x * w), int(r_iris.y * h)), 3, (255, 0, 255), -1)

        nose = landmarks[1]
        nx, ny = int(nose.x * w), int(nose.y * h)
        cv2.circle(frame, (nx, ny), 4, (0, 255, 255), -1)
        ray_len = 35
        rad_pitch = math.radians(pitch)
        rad_yaw = math.radians(yaw)
        dx = int(-ray_len * math.sin(rad_yaw))
        dy = int(ray_len * math.sin(rad_pitch))
        cv2.line(frame, (nx, ny), (nx + dx, ny + dy), (0, 255, 255), 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        blink_text = "BLINK" if is_blinking else "OPEN"
        blink_color = (0, 0, 255) if is_blinking else (0, 255, 128)
        cv2.putText(frame, f"EAR: {ear:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"[{blink_text}]", (135, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, blink_color, 2)
        cv2.putText(frame, f"BPM: {blink_rate:.0f}", (245, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(frame, f"Gaze: {int(gaze_stab*100)}%", (365, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 230, 255), 1)

    def _draw_synthetic_avatar(self, t: float, ear: float, is_blink: bool, bpm: float, gaze: float) -> np.ndarray:
        """Render animated stylized avatar frame for synthetic / camera-off mode."""
        w, h = self.config.target_width, self.config.target_height
        img = np.full((h, w, 3), (25, 28, 38), dtype=np.uint8)

        cx, cy = w // 2, h // 2
        cv2.ellipse(img, (cx, cy), (110, 150), 0, 0, 360, (45, 52, 70), -1)
        cv2.ellipse(img, (cx, cy), (110, 150), 0, 0, 360, (70, 80, 105), 2)

        eye_y = cy - 25
        eye_w = 32
        eye_h = int(22 * (ear / 0.30)) if not is_blink else 2

        for eye_x in [cx - 45, cx + 45]:
            if is_blink:
                cv2.line(img, (eye_x - eye_w // 2, eye_y), (eye_x + eye_w // 2, eye_y), (0, 0, 255), 3)
            else:
                cv2.ellipse(img, (eye_x, eye_y), (eye_w // 2, max(2, eye_h // 2)), 0, 0, 360, (240, 240, 245), -1)
                cv2.ellipse(img, (eye_x, eye_y), (eye_w // 2, max(2, eye_h // 2)), 0, 0, 360, (0, 255, 128), 2)
                pupil_dx = int(5.0 * math.sin(t * 1.2))
                cv2.circle(img, (eye_x + pupil_dx, eye_y), 5, (20, 20, 25), -1)
                cv2.circle(img, (eye_x + pupil_dx, eye_y), 2, (255, 0, 255), -1)

        cv2.line(img, (cx, cy - 5), (cx - 6, cy + 25), (100, 115, 140), 2)
        cv2.line(img, (cx - 6, cy + 25), (cx + 8, cy + 25), (100, 115, 140), 2)
        cv2.ellipse(img, (cx, cy + 65), (28, 10), 0, 0, 180, (100, 115, 140), 2)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        blink_text = "BLINK" if is_blink else "OPEN"
        blink_color = (0, 0, 255) if is_blink else (0, 255, 128)
        cv2.putText(img, f"EAR: {ear:.2f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, f"[{blink_text}]", (135, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, blink_color, 2)
        cv2.putText(img, f"BPM: {bpm:.0f}", (245, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
        cv2.putText(img, f"Gaze: {int(gaze*100)}%", (365, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 230, 255), 1)
        cv2.putText(img, "[SIMULATED AVATAR]", (w - 180, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 150), 1)
        return img

    def get_latest_sample(self) -> VisionSample:
        """Return the most recent processed vision sample in a thread-safe manner."""
        with self._lock:
            return self._latest_sample

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], VisionSample]:
        """Return the most recent annotated frame and sample in a thread-safe manner."""
        with self._lock:
            frame = self._latest_annotated_frame.copy() if self._latest_annotated_frame is not None else None
            return frame, self._latest_sample

    def start(self) -> None:
        """Start non-blocking camera capture worker thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="VisionTrackerThread")
        self._thread.start()
        logger.info("VisionTracker thread started.")

    def stop(self) -> None:
        """Stop background worker thread and release video resource."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        logger.info("VisionTracker thread stopped.")

    def _worker_loop(self) -> None:
        """Continuous background capture loop."""
        cap = None
        if not self.config.synthetic_mode:
            try:
                cap = cv2.VideoCapture(self.config.camera_id, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)
                cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
                if not cap.isOpened():
                    logger.warning(f"Could not open camera {self.config.camera_id}. Falling back to synthetic vision mode.")
                    self.config.synthetic_mode = True
                    cap = None
            except Exception as e:
                logger.warning(f"Failed to initialize video capture: {e}. Falling back to synthetic vision.")
                self.config.synthetic_mode = True
                cap = None

        while self._running:
            t_start = time.time()
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.process_frame(frame, t_start)
                else:
                    self.process_frame(np.zeros((self.config.target_height, self.config.target_width, 3), dtype=np.uint8), t_start)
            else:
                self.process_frame(np.zeros((self.config.target_height, self.config.target_width, 3), dtype=np.uint8), t_start)

            # Throttle to target FPS
            elapsed = time.time() - t_start
            sleep_time = max(0.005, (1.0 / self.config.target_fps) - elapsed)
            time.sleep(sleep_time)

        if cap is not None:
            cap.release()
