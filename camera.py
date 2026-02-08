#!/usr/bin/env python3
"""
Camera Module - Raspberry Pi 5
==============================
Camera capture using V4L2 backend for Pi Camera or USB webcam.
"""

import cv2
import time
from typing import Optional, Tuple, Generator

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)


class PiCamera:
    """Camera handler optimized for Raspberry Pi 5."""
    
    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
    def open(self) -> bool:
        """Open camera with V4L2 backend (Linux/Pi native)."""
        # Try V4L2 first (native Pi camera support)
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            # Fallback to auto backend
            print("V4L2 failed, trying auto backend...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)
            
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Reduce buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Read actual settings
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        backend = self.cap.getBackendName()
        
        print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps ({backend})")
        return True
    
    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """Read a single frame."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def stream(self) -> Generator[cv2.typing.MatLike, None, None]:
        """Generator that yields frames continuously."""
        if not self.cap or not self.cap.isOpened():
            if not self.open():
                return
        
        while True:
            ret, frame = self.read()
            if not ret:
                print("Frame read failed, retrying...")
                time.sleep(0.1)
                continue
            yield frame
    
    def close(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera closed.")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def list_cameras(max_cameras: int = 5) -> list:
    """List all available cameras on the system."""
    print("\n" + "=" * 50)
    print("  CAMERA DISCOVERY (Raspberry Pi)")
    print("=" * 50)
    
    available = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            
            ret, _ = cap.read()
            status = "✓ Working" if ret else "⚠ Opens but can't read"
            
            available.append({
                'index': i,
                'resolution': f"{w}x{h}",
                'fps': fps,
                'backend': backend,
                'status': status
            })
            print(f"  Camera {i}: {w}x{h} @ {fps:.0f}fps ({backend}) - {status}")
            cap.release()
    
    if not available:
        print("  No cameras found!")
        print("  Tips:")
        print("    - Check if Pi Camera is enabled: sudo raspi-config")
        print("    - For USB camera, ensure it's connected")
        print("    - Check permissions: ls -la /dev/video*")
    
    print("=" * 50 + "\n")
    return available


if __name__ == "__main__":
    # Test camera when run directly
    cameras = list_cameras()
    
    if cameras:
        print("Testing first available camera...")
        with PiCamera(camera_index=cameras[0]['index']) as cam:
            print("Press 'q' to quit, 's' to save screenshot")
            
            for frame in cam.stream():
                cv2.imshow("Pi Camera Test", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    fname = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(fname, frame)
                    print(f"Saved: {fname}")
            
            cv2.destroyAllWindows()
