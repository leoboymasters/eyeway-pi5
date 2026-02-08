#!/usr/bin/env python3
"""
Object Detector - Raspberry Pi 5
================================
YOLO-based object detection optimized for Pi 5 CPU inference.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import (
    YOLO_MODEL, CONFIDENCE_THRESHOLD, MAX_INFERENCE_SIZE
)


class ObjectDetector:
    """YOLO object detector optimized for Raspberry Pi 5."""
    
    def __init__(self, model_path: Optional[Path] = None, confidence: float = CONFIDENCE_THRESHOLD):
        self.model_path = model_path or YOLO_MODEL
        self.confidence = confidence
        self.model = None
        self.class_names = []
        
    def load(self) -> bool:
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            
            # Force CPU inference on Pi
            self.model.to('cpu')
            
            self.class_names = self.model.names
            print(f"Model loaded with {len(self.class_names)} classes")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run object detection on a frame.
        
        Returns:
            List of detections with keys: bbox, confidence, class_id, class_name
        """
        if self.model is None:
            if not self.load():
                return []
        
        # Run inference with smaller image size for Pi performance
        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=MAX_INFERENCE_SIZE,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                })
        
        return detections
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, Any]],
        show_labels: bool = True
    ) -> np.ndarray:
        """Draw detection boxes on frame."""
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            if show_labels:
                # Draw label background
                label = f"{cls_name}: {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    output, 
                    (x1, y1 - label_h - 10), 
                    (x1 + label_w + 5, y1), 
                    color, 
                    -1
                )
                cv2.putText(
                    output, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )
        
        return output


if __name__ == "__main__":
    # Test detector when run directly
    from camera import PiCamera
    
    detector = ObjectDetector()
    if not detector.load():
        print("Failed to load model!")
        exit(1)
    
    with PiCamera() as cam:
        print("Running detection... Press 'q' to quit")
        
        frame_count = 0
        for frame in cam.stream():
            frame_count += 1
            
            # Skip frames for better performance on Pi
            if frame_count % 3 != 0:
                continue
            
            detections = detector.detect(frame)
            output = detector.draw_detections(frame, detections)
            
            # Show FPS info
            cv2.putText(
                output, f"Detections: {len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            cv2.imshow("Object Detection", output)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
