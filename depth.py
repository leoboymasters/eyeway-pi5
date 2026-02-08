#!/usr/bin/env python3
"""
Depth Estimator - Raspberry Pi 5
================================
Lightweight depth estimation using MiDaS (CPU-optimized).
"""

import cv2
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from config import DEPTH_MODEL, MAX_INFERENCE_SIZE


class DepthEstimator:
    """Lightweight depth estimator for Raspberry Pi 5 using MiDaS."""
    
    def __init__(self, model_type: str = DEPTH_MODEL):
        """
        Initialize depth estimator.
        
        Args:
            model_type: MiDaS model variant
                - "midas_v21_small" - Fastest, good for Pi (recommended)
                - "midas_v21" - Better quality, slower
                - "dpt_swin2_tiny_256" - Modern, balanced
        """
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = torch.device('cpu')  # Pi 5 = CPU only
        
    def load(self) -> bool:
        """Load the MiDaS depth model."""
        try:
            print(f"Loading MiDaS model: {self.model_type}")
            
            # Load model from torch hub
            self.model = torch.hub.load(
                "intel-isl/MiDaS", 
                self.model_type,
                trust_repo=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", 
                "transforms",
                trust_repo=True
            )
            
            if "small" in self.model_type:
                self.transform = midas_transforms.small_transform
            else:
                self.transform = midas_transforms.dpt_transform
            
            print("Depth model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading depth model: {e}")
            return False
    
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from an RGB frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Depth map as numpy array (higher values = farther)
        """
        if self.model is None:
            if not self.load():
                return None
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(rgb).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        return depth_map
    
    def estimate_metric(
        self, 
        frame: np.ndarray, 
        camera_height: float = 1.5,
        camera_pitch: float = 45.0
    ) -> Optional[np.ndarray]:
        """
        Estimate metric depth using camera calibration.
        
        Note: MiDaS outputs relative depth. This function applies
        a simple geometric correction based on camera pose.
        
        Args:
            frame: BGR image
            camera_height: Height of camera from ground (meters)
            camera_pitch: Downward tilt angle (degrees)
            
        Returns:
            Approximate metric depth map
        """
        relative_depth = self.estimate(frame)
        if relative_depth is None:
            return None
        
        # Normalize relative depth
        depth_min = relative_depth.min()
        depth_max = relative_depth.max()
        if depth_max - depth_min > 0:
            normalized = (relative_depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(relative_depth)
        
        # Simple geometric approximation
        # Assumes flat ground plane
        pitch_rad = np.radians(camera_pitch)
        
        # Scale factor based on camera geometry
        # This is a rough approximation - proper calibration would be more accurate
        scale = camera_height / np.sin(pitch_rad)
        
        # Convert to approximate metric depth
        metric_depth = normalized * scale * 5.0  # Empirical scaling
        
        return metric_depth
    
    def colorize(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to colorized visualization."""
        # Normalize to 0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max - depth_min > 0:
            normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_map)
        
        depth_uint8 = (normalized * 255).astype(np.uint8)
        
        # Apply colormap
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        
        return colored


if __name__ == "__main__":
    # Test depth estimator
    from camera import PiCamera
    
    estimator = DepthEstimator()
    if not estimator.load():
        print("Failed to load depth model!")
        exit(1)
    
    with PiCamera() as cam:
        print("Running depth estimation... Press 'q' to quit")
        
        frame_count = 0
        for frame in cam.stream():
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % 5 != 0:
                continue
            
            depth = estimator.estimate(frame)
            if depth is not None:
                colored_depth = estimator.colorize(depth)
                
                # Side-by-side view
                combined = np.hstack([frame, colored_depth])
                cv2.imshow("RGB | Depth", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
