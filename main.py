#!/usr/bin/env python3
"""
Eyeway Main Application - Raspberry Pi 5
=========================================
Real-time object detection with depth estimation for navigation assistance.
"""

import cv2
import time
import argparse
import numpy as np
from typing import Optional

from config import (
    CAMERA_INDEX, SKIP_FRAMES, SHOW_PREVIEW,
    DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_PITCH,
    ENABLE_AUDIO, DISTANCE_WARNING_THRESHOLD
)
from camera import PiCamera
from detector import ObjectDetector
from depth import DepthEstimator


class EyewayApp:
    """Main Eyeway application combining detection and depth estimation."""
    
    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        enable_depth: bool = True,
        enable_audio: bool = ENABLE_AUDIO,
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        camera_pitch: float = DEFAULT_CAMERA_PITCH
    ):
        self.camera = PiCamera(camera_index=camera_index)
        self.detector = ObjectDetector()
        self.depth_estimator = DepthEstimator() if enable_depth else None
        
        self.enable_depth = enable_depth
        self.enable_audio = enable_audio
        self.camera_height = camera_height
        self.camera_pitch = camera_pitch
        
        self.running = False
        self.frame_count = 0
        self.fps = 0.0
        self.fps_timer = time.time()
        
    def initialize(self) -> bool:
        """Initialize all components."""
        print("=" * 50)
        print("  EYEWAY - Raspberry Pi 5")
        print("=" * 50)
        
        # Load detector
        print("\n[1/3] Loading object detector...")
        if not self.detector.load():
            print("Failed to load detector!")
            return False
        
        # Load depth estimator
        if self.enable_depth:
            print("\n[2/3] Loading depth estimator...")
            if not self.depth_estimator.load():
                print("Warning: Depth estimation disabled")
                self.enable_depth = False
        else:
            print("\n[2/3] Depth estimation disabled")
        
        # Open camera
        print("\n[3/3] Opening camera...")
        if not self.camera.open():
            print("Failed to open camera!")
            return False
        
        print("\n" + "=" * 50)
        print("  Initialization complete!")
        print("=" * 50)
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with detection and depth."""
        output = frame.copy()
        
        # Run object detection
        detections = self.detector.detect(frame)
        
        # Get depth map if enabled
        depth_map = None
        if self.enable_depth:
            depth_map = self.depth_estimator.estimate_metric(
                frame,
                camera_height=self.camera_height,
                camera_pitch=self.camera_pitch
            )
        
        # Process each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = det['center']
            cls_name = det['class_name']
            conf = det['confidence']
            
            # Get distance if depth available
            distance = None
            if depth_map is not None:
                # Sample depth at detection center
                if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                    distance = depth_map[cy, cx]
            
            # Choose color based on distance
            if distance is not None and distance < DISTANCE_WARNING_THRESHOLD:
                color = (0, 0, 255)  # Red for close objects
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if distance is not None:
                label = f"{cls_name}: {distance:.1f}m"
            else:
                label = f"{cls_name}: {conf:.2f}"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(output, (x1, y1 - label_h - 10), 
                         (x1 + label_w + 5, y1), color, -1)
            cv2.putText(output, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw HUD
        self._draw_hud(output, len(detections))
        
        return output
    
    def _draw_hud(self, frame: np.ndarray, detection_count: int):
        """Draw heads-up display with stats."""
        h, w = frame.shape[:2]
        
        # FPS counter
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Detection count
        cv2.putText(
            frame, f"Objects: {detection_count}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Depth mode
        depth_text = "Depth: ON" if self.enable_depth else "Depth: OFF"
        cv2.putText(
            frame, depth_text, (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Controls hint
        cv2.putText(
            frame, "Q=Quit | D=Toggle Depth", (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
    
    def run(self):
        """Main application loop."""
        if not self.initialize():
            return
        
        self.running = True
        print("\nRunning... Press 'Q' to quit\n")
        
        try:
            for frame in self.camera.stream():
                if not self.running:
                    break
                
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % SKIP_FRAMES != 0:
                    continue
                
                # Update FPS
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.fps_timer
                    self.fps = 30 / elapsed if elapsed > 0 else 0
                    self.fps_timer = time.time()
                
                # Process frame
                output = self.process_frame(frame)
                
                # Show preview
                if SHOW_PREVIEW:
                    cv2.imshow("Eyeway", output)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord('d'):
                        self.enable_depth = not self.enable_depth
                        print(f"Depth: {'ON' if self.enable_depth else 'OFF'}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        self.camera.close()
        cv2.destroyAllWindows()
        print("\nEyeway stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Eyeway - Navigation Assistance for Raspberry Pi 5"
    )
    parser.add_argument(
        "--camera", "-c", type=int, default=CAMERA_INDEX,
        help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Disable depth estimation for faster inference"
    )
    parser.add_argument(
        "--height", type=float, default=DEFAULT_CAMERA_HEIGHT,
        help="Camera height from ground in meters (default: 1.5)"
    )
    parser.add_argument(
        "--pitch", type=float, default=DEFAULT_CAMERA_PITCH,
        help="Camera downward pitch angle in degrees (default: 45)"
    )
    parser.add_argument(
        "--audio", action="store_true",
        help="Enable audio feedback"
    )
    
    args = parser.parse_args()
    
    app = EyewayApp(
        camera_index=args.camera,
        enable_depth=not args.no_depth,
        enable_audio=args.audio,
        camera_height=args.height,
        camera_pitch=args.pitch
    )
    
    app.run()


if __name__ == "__main__":
    main()
