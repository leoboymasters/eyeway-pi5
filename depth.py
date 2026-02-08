#!/usr/bin/env python3
"""
Depth Estimator - Raspberry Pi 5
================================
Depth estimation using Depth Anything V3.
"""

import os
import sys
import cv2
import numpy as np
import torch
from typing import Optional
from pathlib import Path

# Add path to depth_anything_3 module
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
DA3_PATH = PROJECT_ROOT / 'dependencies' / 'depth-anything-3' / 'src'
if str(DA3_PATH) not in sys.path:
    sys.path.insert(0, str(DA3_PATH))

from config import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_PITCH


class DepthEstimator:
    """Depth Anything V3 estimator for Raspberry Pi 5."""
    
    # Available DA3 models (smaller = faster on Pi)
    MODELS = {
        'small': 'depth-anything/DA3Metric-Small',
        'base': 'depth-anything/DA3Metric-Base', 
        'large': 'depth-anything/DA3Metric-Large',
    }
    
    def __init__(self, model_size: str = 'small', process_res: int = 384):
        """
        Initialize Depth Anything V3 estimator.
        
        Args:
            model_size: 'small', 'base', or 'large'
                - small: Fastest, recommended for Pi 5
                - base: Balanced
                - large: Best quality, very slow on Pi
            process_res: Processing resolution (lower = faster)
                - 256: Fastest
                - 384: Good balance for Pi
                - 504: Better quality, slower
        """
        self.model_size = model_size
        self.model_name = self.MODELS.get(model_size, self.MODELS['small'])
        self.process_res = process_res
        self.model = None
        self.device = 'cpu'  # Pi 5 = CPU only
        
    def load(self) -> bool:
        """Load the Depth Anything V3 model."""
        try:
            from depth_anything_3.api import DepthAnything3
            
            print(f"Loading Depth Anything V3 ({self.model_size})...")
            print(f"Model: {self.model_name}")
            print(f"Device: {self.device}")
            print(f"Process resolution: {self.process_res}")
            
            self.model = DepthAnything3.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            print("Depth Anything V3 loaded successfully!")
            return True
            
        except ImportError as e:
            print(f"Error: Could not import Depth Anything V3: {e}")
            print("Make sure depth-anything-3 is installed:")
            print("  pip install ../dependencies/depth-anything-3")
            return False
        except Exception as e:
            print(f"Error loading Depth Anything V3: {e}")
            return False
    
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate metric depth from an RGB frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Depth map in meters (metric depth)
        """
        if self.model is None:
            if not self.load():
                return None
        
        try:
            h, w = frame.shape[:2]
            
            # DA3 expects RGB, convert from BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run DA3 inference
            prediction = self.model.inference(
                image=[rgb], 
                process_res=self.process_res
            )
            
            # Get depth map
            depth = prediction.depth[0]
            
            # Resize to original frame size if needed
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h))
            
            return depth
            
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None
    
    def estimate_with_calibration(
        self, 
        frame: np.ndarray, 
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        camera_pitch: float = DEFAULT_CAMERA_PITCH
    ) -> Optional[np.ndarray]:
        """
        Estimate depth with camera calibration correction.
        
        DA3 outputs metric depth, but calibration can improve accuracy
        for specific camera setups.
        
        Args:
            frame: BGR image
            camera_height: Height of camera from ground (meters)
            camera_pitch: Downward tilt angle (degrees)
            
        Returns:
            Calibrated metric depth map
        """
        depth = self.estimate(frame)
        if depth is None:
            return None
        
        # DA3 already outputs metric depth, return as-is
        # Additional calibration could be applied here if needed
        return depth
    
    def colorize(self, depth_map: np.ndarray, min_depth: float = 0.0, max_depth: float = 10.0) -> np.ndarray:
        """
        Convert depth map to colorized visualization.
        
        Args:
            depth_map: Metric depth map in meters
            min_depth: Minimum depth for colormap (meters)
            max_depth: Maximum depth for colormap (meters)
            
        Returns:
            Colored depth visualization (BGR)
        """
        # Clip and normalize
        depth_clipped = np.clip(depth_map, min_depth, max_depth)
        normalized = (depth_clipped - min_depth) / (max_depth - min_depth + 1e-8)
        
        depth_uint8 = (normalized * 255).astype(np.uint8)
        
        # Apply colormap (INFERNO matches original project)
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        
        return colored
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int, radius: int = 5) -> float:
        """
        Get depth at a specific point with neighborhood averaging.
        
        Args:
            depth_map: Depth map
            x, y: Pixel coordinates
            radius: Radius for averaging (reduces noise)
            
        Returns:
            Depth in meters at the point
        """
        h, w = depth_map.shape[:2]
        
        # Clamp coordinates
        x = max(radius, min(x, w - radius - 1))
        y = max(radius, min(y, h - radius - 1))
        
        # Get neighborhood
        neighborhood = depth_map[y-radius:y+radius+1, x-radius:x+radius+1]
        
        # Return median (robust to outliers)
        return float(np.median(neighborhood))


if __name__ == "__main__":
    # Test depth estimator
    from camera import PiCamera
    import time
    
    # Use small model for Pi 5 performance
    estimator = DepthEstimator(model_size='small', process_res=384)
    
    if not estimator.load():
        print("Failed to load depth model!")
        exit(1)
    
    with PiCamera() as cam:
        print("\nRunning Depth Anything V3 estimation...")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        fps_start = time.time()
        fps = 0
        
        for frame in cam.stream():
            frame_count += 1
            
            # Skip frames for performance on Pi
            if frame_count % 3 != 0:
                continue
            
            # Estimate depth
            depth = estimator.estimate(frame)
            
            if depth is not None:
                # Colorize depth
                colored_depth = estimator.colorize(depth)
                
                # Calculate FPS
                if frame_count % 15 == 0:
                    fps = 15 / (time.time() - fps_start)
                    fps_start = time.time()
                
                # Get center depth
                h, w = depth.shape[:2]
                center_depth = estimator.get_depth_at_point(depth, w//2, h//2)
                
                # Draw info on depth visualization
                cv2.putText(colored_depth, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(colored_depth, f"Center: {center_depth:.2f}m", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw center crosshair
                cv2.circle(colored_depth, (w//2, h//2), 5, (0, 255, 0), -1)
                
                # Side-by-side view
                combined = np.hstack([frame, colored_depth])
                cv2.imshow("RGB | Depth Anything V3", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
