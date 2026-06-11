"""
Desktop Webcam Fall Detection Application
Real-time fall detection using computer webcam

Run this standalone application on desktop for:
- Real-time webcam monitoring
- Instant fall detection alerts
- No web lag (direct camera access)
- Visual + Audio alerts
- Screenshot capture on detection

Usage:
    python webcam_fall_detector.py
    
Controls:
    Q - Quit
    S - Save current frame
    R - Reset detection
    P - Pause/Resume
"""

import cv2
import numpy as np
from collections import deque
import os
from datetime import datetime
from pathlib import Path
import sys
import threading
import time
import json

# Audio alerts (Windows only)
try:
    import winsound
    _WINSOUND = True
except ImportError:
    _WINSOUND = False

# Add src to path
sys.path.insert(0, 'src')

try:
    from email_alert import send_fall_alert, send_test_email
    _EMAIL_AVAILABLE = True
except ImportError:
    _EMAIL_AVAILABLE = False
    print("⚠️  email_alert module not found — email alerts disabled")


class RealtimeWebcamFallDetector:
    """
    Real-time fall detection from webcam feed
    Optimized for desktop performance
    """
    
    def __init__(self,
                 camera_id=0,
                 still_threshold=0.015,
                 active_threshold=0.03,
                 min_still_frames=3,
                 min_active_frames=4,
                 lookback_window=25,
                 cooldown_frames=30,
                 confirm_window=15,
                 confirm_required=10,
                 output_dir='webcam_detections',
                 # ─ email alert settings ─────────────────────────────────
                 email_enabled=False,
                 email_recipient='',
                 email_sender='',
                 email_password='',
                 smtp_host='smtp.gmail.com',
                 smtp_port=587):
        """
        Initialize webcam fall detector

        Args:
            camera_id        : Webcam ID (0 for default camera)
            still_threshold  : Motion score below this = still
            active_threshold : Motion score above this = active
            min_still_frames : Consecutive still frames to trigger
            min_active_frames: Active frames required in history
            lookback_window  : Sliding window size
            cooldown_frames  : Frames to skip after detection
            confirm_window   : Future frames to check for confirmation
            confirm_required : How many future frames must be still
            output_dir       : Where to save detection screenshots
            email_enabled    : Enable email alerts on fall detection
            email_recipient  : Comma-separated recipient email address(es)
            email_sender     : SMTP sender email address
            email_password   : SMTP password / Gmail App Password
            smtp_host        : SMTP server host
            smtp_port        : SMTP server port
        """
        
        self.camera_id = camera_id
        self.still_threshold = still_threshold
        self.active_threshold = active_threshold
        self.min_still_frames = min_still_frames
        self.min_active_frames = min_active_frames
        self.lookback_window = lookback_window
        self.cooldown_frames = cooldown_frames
        self.confirm_window = confirm_window
        self.confirm_required = confirm_required
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Email settings
        self.email_enabled   = email_enabled and _EMAIL_AVAILABLE
        self.email_recipient = email_recipient
        self.email_sender    = email_sender
        self.email_password  = email_password
        self.smtp_host       = smtp_host
        self.smtp_port       = smtp_port
        
        # State variables
        self.motion_buffer = deque(maxlen=lookback_window)
        self.still_count = 0
        self.cooldown = 0
        self.detection_count = 0
        self.prev_gray = None
        
        # Confirmation phase
        self.confirming = False
        self.confirmation_start = 0
        self.confirmation_frames = []
        
        # Statistics
        self.total_frames = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_frame_count = 0
        
        # Status
        self.paused = False
        self.running = True
        
        print("="*60)
        print("🎥 WEBCAM FALL DETECTOR - Desktop Application")
        print("="*60)
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Camera ID: {camera_id}")
        print("\nControls:")
        print("  Q - Quit")
        print("  S - Save current frame")
        print("  R - Reset detection state")
        print("  P - Pause/Resume")
        print("="*60)
    
    def calculate_motion_score(self, gray_frame):
        """
        Calculate motion score between current and previous frame
        
        Returns:
            float: Motion score (0-1), fraction of changed pixels
        """
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return 0.0
        
        # Calculate difference
        diff = cv2.absdiff(self.prev_gray, gray_frame)
        self.prev_gray = gray_frame
        
        # Threshold
        _, thresh = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)
        
        # Calculate score
        h, w = thresh.shape
        score = float(np.sum(thresh)) / (w * h)
        
        return score
    
    def detect_fall(self, motion_score):
        """
        Check if current motion pattern indicates a fall
        
        Args:
            motion_score: Current frame's motion score
            
        Returns:
            tuple: (is_fall_candidate, confidence)
        """
        # Add to buffer
        self.motion_buffer.append(motion_score)
        
        # Handle cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            self.still_count = 0
            return False, 0.0
        
        # If in confirmation phase, collect frames
        if self.confirming:
            self.confirmation_frames.append(motion_score)
            
            # Check if we have enough confirmation frames
            if len(self.confirmation_frames) >= self.confirm_window:
                # Count still frames
                still_count = sum(1 for s in self.confirmation_frames 
                                 if s < self.still_threshold)
                
                # Decide
                if still_count >= self.confirm_required:
                    # FALL CONFIRMED!
                    self.confirming = False
                    self.confirmation_frames = []
                    self.cooldown = self.cooldown_frames
                    
                    # Calculate confidence
                    peak = max(self.motion_buffer) if self.motion_buffer else 0
                    confidence = min(1.0, peak / 0.10)
                    
                    return True, confidence
                else:
                    # Not a fall, reset
                    self.confirming = False
                    self.confirmation_frames = []
            
            return False, 0.0
        
        # Count active and still frames
        buffer_list = list(self.motion_buffer)
        active_frames = sum(1 for s in buffer_list[:-1] 
                           if s > self.active_threshold)
        was_active = active_frames >= self.min_active_frames
        
        # Track consecutive still frames
        if motion_score < self.still_threshold:
            self.still_count += 1
        else:
            self.still_count = 0
        
        # CANDIDATE DETECTION
        if was_active and self.still_count >= self.min_still_frames:
            # Person was moving, now still
            # Start confirmation phase
            self.confirming = True
            self.confirmation_start = self.total_frames
            self.confirmation_frames = []
            self.still_count = 0
        
        return False, 0.0
    
    def play_alert_sound(self):
        """Play alert sound in separate thread (non-blocking)"""
        def play_sound():
            try:
                if _WINSOUND:
                    winsound.Beep(1000, 500)  # 1000 Hz, 500ms
                    time.sleep(0.1)
                    winsound.Beep(1000, 500)
            except:
                pass

        thread = threading.Thread(target=play_sound, daemon=True)
        thread.start()
    
    def save_detection(self, frame, confidence):
        """
        Save detection screenshot with annotation
        
        Args:
            frame: BGR frame to save
            confidence: Detection confidence
        """
        self.detection_count += 1
        
        # Annotate frame
        annotated = self.annotate_fall(frame.copy(), 
                                       self.detection_count, 
                                       confidence)
        
        # Save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"fall_{timestamp}_{self.detection_count:03d}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), annotated)
        
        print(f"🚨 FALL DETECTED #{self.detection_count}")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Confidence: {confidence*100:.1f}%")
        print(f"   Saved: {filename}")

        # Play alert sound
        self.play_alert_sound()

        # ── Email alert ───────────────────────────────────────
        if self.email_enabled:
            def _send():
                ok, msg = send_fall_alert(
                    recipient_email = self.email_recipient,
                    sender_email    = self.email_sender,
                    sender_password = self.email_password,
                    image_path      = str(filepath),
                    detection_info  = {
                        'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'confidence':    confidence,
                        'source_file':   'Webcam (Live)',
                        'detection_num': self.detection_count,
                    },
                    source    = 'webcam',
                    smtp_host = self.smtp_host,
                    smtp_port = self.smtp_port,
                )
                status = '✅ Sent' if ok else f'\u26a0\ufe0f Failed: {msg}'
                print(f"   Email alert: {status}")
            threading.Thread(target=_send, daemon=True).start()
    
    def annotate_fall(self, frame, detection_num, confidence):
        """
        Draw fall detection overlay on frame
        
        Args:
            frame: BGR image
            detection_num: Detection number
            confidence: Confidence score
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Red semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text
        cv2.putText(frame, "FALL DETECTED!", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        cv2.putText(frame, f"Detection #{detection_num} | Confidence: {confidence*100:.0f}%",
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Red border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 5)
        
        return frame
    
    def draw_overlay(self, frame, motion_score, status_text="Monitoring"):
        """
        Draw status overlay on frame
        
        Args:
            frame: BGR image to draw on
            motion_score: Current motion score
            status_text: Status message
        """
        h, w = frame.shape[:2]
        
        # Semi-transparent black bar at bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status
        color = (0, 255, 0) if "Monitoring" in status_text else (0, 165, 255)
        cv2.putText(frame, status_text, 
                   (20, h-85), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, color, 2, cv2.LINE_AA)
        
        # Motion score bar
        bar_width = int((w - 40) * min(motion_score / 0.1, 1.0))
        cv2.rectangle(frame, (20, h-60), (20 + bar_width, h-40), 
                     (0, 255, 255) if motion_score > self.active_threshold else (0, 255, 0), 
                     -1)
        cv2.rectangle(frame, (20, h-60), (w-20, h-40), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Motion: {motion_score*100:.2f}%", 
                   (20, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Buffer status
        buffer_status = f"Buffer: {len(self.motion_buffer)}/{self.lookback_window}"
        active_count = sum(1 for s in self.motion_buffer 
                          if s > self.active_threshold)
        buffer_status += f" | Active: {active_count}/{self.min_active_frames}"
        
        cv2.putText(frame, buffer_status,
                   (w-400, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # FPS counter (top right)
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                   (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Detection count (top right)
        cv2.putText(frame, f"Falls: {self.detection_count}",
                   (w-120, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # Update every second
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.last_fps_time = current_time
    
    def run(self):
        """
        Main loop - capture and process webcam feed
        """
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"❌ ERROR: Cannot open camera {self.camera_id}")
            print("   Check if camera is connected and not used by another app")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n✅ Camera opened successfully")
        print(f"   Resolution: {actual_width}x{actual_height}")
        print(f"   Camera FPS: {actual_fps:.1f}")
        print("\nStarting detection...\n")
        
        cv2.namedWindow('Webcam Fall Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam Fall Detector', 800, 600)
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                self.total_frames += 1
                self.update_fps()
                
                # Process if not paused
                if not self.paused:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (9, 9), 0)
                    
                    # Calculate motion
                    motion_score = self.calculate_motion_score(gray)
                    
                    # Detect fall
                    is_fall, confidence = self.detect_fall(motion_score)
                    
                    # Handle detection
                    if is_fall:
                        self.save_detection(frame, confidence)
                    
                    # Status text
                    if self.confirming:
                        confirm_progress = len(self.confirmation_frames)
                        status = f"Confirming... ({confirm_progress}/{self.confirm_window})"
                    elif self.cooldown > 0:
                        status = f"Cooldown ({self.cooldown} frames)"
                    else:
                        status = "Monitoring"
                    
                    # Draw overlay
                    self.draw_overlay(frame, motion_score, status)
                else:
                    # Paused overlay
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                    cv2.putText(frame, "PAUSED", (w//2-100, h//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # Show frame
                cv2.imshow('Webcam Fall Detector', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Quitting...")
                    self.running = False
                    
                elif key == ord('s') or key == ord('S'):
                    # Save current frame
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"snapshot_{timestamp}.jpg"
                    filepath = self.output_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"📸 Snapshot saved: {filename}")
                    
                elif key == ord('r') or key == ord('R'):
                    # Reset state
                    self.motion_buffer.clear()
                    self.still_count = 0
                    self.cooldown = 0
                    self.confirming = False
                    self.confirmation_frames = []
                    self.prev_gray = None
                    print("🔄 Detection state reset")
                    
                elif key == ord('p') or key == ord('P'):
                    # Toggle pause
                    self.paused = not self.paused
                    status = "⏸️  PAUSED" if self.paused else "▶️  RESUMED"
                    print(status)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*60)
            print("📊 SESSION SUMMARY")
            print("="*60)
            print(f"Total frames processed: {self.total_frames}")
            print(f"Falls detected: {self.detection_count}")
            print(f"Output directory: {self.output_dir.absolute()}")
            print("="*60)


def main():
    """Main entry point – loads email settings from app/email_settings.json if present."""
    print("\n" + "="*60)
    print("🎥 WEBCAM FALL DETECTION - Desktop Application")
    print("="*60)
    print("\nInitializing...\n")

    # ── Load email settings saved via the web UI ───────────────────────────
    email_settings_path = Path('app') / 'email_settings.json'
    email_cfg = {}
    if email_settings_path.exists():
        try:
            with open(email_settings_path, 'r') as f:
                email_cfg = json.load(f)
            if email_cfg.get('enabled'):
                print(f"📧 Email alerts ENABLED → recipient: {email_cfg.get('recipient_email','')}")
            else:
                print("📧 Email alerts disabled (enable via web UI or edit app/email_settings.json)")
        except Exception as e:
            print(f"⚠️  Could not load email settings: {e}")
    else:
        print("📧 No email settings found. Configure via web UI or create app/email_settings.json")

    # ── Create detector ────────────────────────────────────────────────────
    detector = RealtimeWebcamFallDetector(
        camera_id         = 0,
        still_threshold   = 0.015,
        active_threshold  = 0.03,
        min_still_frames  = 3,
        min_active_frames = 4,
        confirm_window    = 15,
        confirm_required  = 10,
        output_dir        = 'webcam_detections',
        # email
        email_enabled    = bool(email_cfg.get('enabled', False)),
        email_recipient  = email_cfg.get('recipient_email', ''),
        email_sender     = email_cfg.get('sender_email', ''),
        email_password   = email_cfg.get('sender_password', ''),
        smtp_host        = email_cfg.get('smtp_host', 'smtp.gmail.com'),
        smtp_port        = int(email_cfg.get('smtp_port', 587)),
    )

    # Run
    detector.run()


if __name__ == '__main__':
    main()
