"""
Video Fall Detection Module
Detects falls in uploaded videos using frame-differencing motion analysis.

Detection algorithm (two-phase):
  Phase 1 – Candidate detection:
    Track motion scores (fraction of changed pixels between consecutive frames).
    A fall candidate is flagged when:
      a) There were enough "active" frames in the recent lookback window
         (motion score > active_threshold for at least min_active_frames).
      b) The current frame is "still" (motion score < still_threshold)
         for at least min_still_frames consecutive frames.

  Phase 2 – Confirmation:
    After a candidate is flagged, examine the NEXT confirm_window frames.
    If at least confirm_required of those frames are also still, the fall
    is confirmed (person stayed on the ground).
    If the person starts moving again, the candidate is rejected.

This handles:
  ✗ Standing still  → no recent active frames
  ✗ Walking         → never truly still (scores stay above still_threshold)
  ✗ Camera startup  → brief activity then stillness, but person resumes → rejected
  ✓ Falls           → sudden activity then sustained stillness on the ground
"""

import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from collections import deque


class VideoFallDetector:
    """
    Frame-differencing fall detector for uploaded video files.

    Parameters
    ----------
    still_threshold    : float  – score below this counts as "still" (default 0.015)
    active_threshold   : float  – score above this counts as "active" (default 0.03)
    min_still_frames   : int    – consecutive still frames to trigger candidate (default 2)
    min_active_frames  : int    – active frames required in lookback window (default 3)
    lookback_window    : int    – size of the sliding window for activity history (default 25)
    cooldown_frames    : int    – processed frames to skip after a confirmed detection (default 15)
    confirm_window     : int    – how many future frames to check for confirmation (default 15)
    confirm_required   : int    – how many of those must be still to confirm (default 10)
    diff_threshold     : int    – pixel intensity difference to count as "changed" (default 25)
    """

    def __init__(self,
                 output_dir='app/static/detections',
                 still_threshold=0.015,
                 active_threshold=0.03,
                 min_still_frames=2,
                 min_active_frames=3,
                 lookback_window=25,
                 cooldown_frames=15,
                 confirm_window=15,
                 confirm_required=10,
                 diff_threshold=25):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.still_threshold   = still_threshold
        self.active_threshold  = active_threshold
        self.min_still_frames  = min_still_frames
        self.min_active_frames = min_active_frames
        self.lookback_window   = lookback_window
        self.cooldown_frames   = cooldown_frames
        self.confirm_window    = confirm_window
        self.confirm_required  = confirm_required
        self.diff_threshold    = diff_threshold

        print("VideoFallDetector: Initialised (activity→stillness + confirmation)")

    # ------------------------------------------------------------------ public

    def analyze_video(self, video_path, analysis_fps=10):
        """
        Analyse a video file for fall events.

        Returns dict with: fall_detected, screenshots, detections, video_info
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {'fall_detected': False,
                    'error': 'Could not open video file',
                    'screenshots': [], 'detections': []}

        fps         = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration    = frame_count / fps if fps > 0 else 0

        print(f"Video Info: {width}x{height}, {fps:.1f} FPS, "
              f"{frame_count} frames, {duration:.1f}s")

        frame_skip = max(1, int(fps / analysis_fps)) if fps > 0 else 1
        print(f"Analysing every {frame_skip} frame(s) (~{analysis_fps} analysis FPS)")

        # --- Pass 1: compute all motion scores and collect key frames ---
        all_scores, key_frames = self._compute_scores(cap, frame_skip, width, height, fps)
        cap.release()

        # --- Pass 2: detect falls with confirmation ---
        fall_indices = self._detect_falls(all_scores)

        # --- Pass 3: generate outputs ---
        fall_detections = []
        screenshots     = []

        for det_num, score_idx in enumerate(fall_indices, 1):
            fidx, timestamp, _ = all_scores[score_idx]
            frame = key_frames.get(fidx)
            if frame is None:
                continue

            # Compute a confidence score based on peak activity in lookback
            start = max(0, score_idx - self.lookback_window)
            peak = max(s for _, _, s in all_scores[start:score_idx + 1])
            confidence = min(1.0, peak / 0.10)  # 0.10+ → 100%

            fname = (f"fall_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                     f"_{det_num:03d}.jpg")
            screenshot_path = self.output_dir / fname
            annotated = self._annotate(frame.copy(), det_num, confidence)
            cv2.imwrite(str(screenshot_path), annotated)

            screenshots.append(f"detections/{fname}")
            fall_detections.append({
                'timestamp':  round(timestamp, 2),
                'frame':      fidx,
                'confidence': round(confidence, 2),
            })
            print(f"  Fall confirmed at {timestamp:.2f}s "
                  f"(frame {fidx}, conf {confidence:.0%})")

        return {
            'fall_detected': len(fall_detections) > 0,
            'screenshots':   screenshots,
            'detections':    fall_detections,
            'video_info': {
                'width':           width,
                'height':          height,
                'fps':             round(fps, 2),
                'duration':        round(duration, 2),
                'total_frames':    frame_count,
                'analyzed_frames': len(all_scores),
            },
        }

    # ------------------------------------------------------------------ private

    def _compute_scores(self, cap, frame_skip, width, height, fps):
        """
        First pass: read video, compute motion scores, keep BGR frames.

        Returns:
            all_scores  – list of (frame_idx, timestamp, motion_score)
            key_frames  – dict {frame_idx: BGR frame} (every analysed frame kept)
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_gray  = None
        all_scores = []
        key_frames = {}
        frame_idx  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                gray = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0)

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, mask = cv2.threshold(
                        diff, self.diff_threshold, 1, cv2.THRESH_BINARY)
                    score = float(np.sum(mask)) / (width * height)
                else:
                    score = 0.0

                prev_gray = gray
                timestamp = frame_idx / fps if fps > 0 else 0
                all_scores.append((frame_idx, timestamp, score))
                key_frames[frame_idx] = frame

            frame_idx += 1

        return all_scores, key_frames

    def _detect_falls(self, all_scores):
        """
        Detect falls using activity → sustained-stillness pattern with confirmation.

        Returns list of indices (into all_scores) where falls are confirmed.
        """
        buf         = deque(maxlen=self.lookback_window)
        still_count = 0
        cooldown    = 0
        confirmed   = []

        for i, (fidx, t, score) in enumerate(all_scores):
            buf.append(score)

            if cooldown > 0:
                cooldown -= 1
                still_count = 0
                continue

            # Count active frames in the lookback window (excluding current)
            buf_list = list(buf)
            active_frames = sum(
                1 for s in buf_list[:-1] if s > self.active_threshold)
            was_active = active_frames >= self.min_active_frames

            # Track consecutive still frames
            if score < self.still_threshold:
                still_count += 1
            else:
                still_count = 0

            # Phase 1: candidate detection
            if was_active and still_count >= self.min_still_frames:
                # Phase 2: confirmation – check future frames
                future = all_scores[i + 1: i + 1 + self.confirm_window]
                if len(future) < self.confirm_window:
                    # Near end of video → person stayed down → auto-confirm
                    future_still = self.confirm_required
                else:
                    future_still = sum(
                        1 for _, _, s in future if s < self.still_threshold)

                if future_still >= self.confirm_required:
                    confirmed.append(i)
                    cooldown    = self.cooldown_frames
                    still_count = 0

        return confirmed

    def _annotate(self, frame, detection_number, confidence):
        """Draw fall overlay on a frame."""
        h, w = frame.shape[:2]

        # semi-transparent red banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "FALL DETECTED",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (255, 255, 255), 3, cv2.LINE_AA)

        info = f"Detection #{detection_number}  |  Confidence: {confidence:.0%}"
        cv2.putText(frame, info,
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 220, 220), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        return frame
