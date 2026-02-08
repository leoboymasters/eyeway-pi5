"""
Eyeway App Configuration - Raspberry Pi 5
==========================================
Configuration settings optimized for Pi 5 hardware.
"""

from pathlib import Path

# Paths
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = APP_DIR / "data"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

# Camera Settings
CAMERA_INDEX = 0  # Default camera (0 = Pi Camera via V4L2, or USB webcam)
CAMERA_WIDTH = 640  # Lower resolution for Pi performance
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Model Settings
YOLO_MODEL = MODELS_DIR / "yolo11n.pt"  # Use nano model for speed
DEPTH_MODEL = "midas_v21_small"  # Lightweight depth model for Pi
CONFIDENCE_THRESHOLD = 0.5

# Performance Settings
USE_HALF_PRECISION = False  # Disabled for CPU
MAX_INFERENCE_SIZE = 384  # Smaller input size for faster inference
SKIP_FRAMES = 2  # Process every Nth frame to maintain real-time

# Display Settings
SHOW_PREVIEW = True
PREVIEW_SCALE = 1.0

# Audio Feedback
ENABLE_AUDIO = False  # Set True for audio alerts
DISTANCE_WARNING_THRESHOLD = 1.0  # meters

# Calibration
DEFAULT_CAMERA_HEIGHT = 1.5  # meters (typical mounting height)
DEFAULT_CAMERA_PITCH = 45.0  # degrees (downward angle)
