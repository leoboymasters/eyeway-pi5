# Eyeway App - Raspberry Pi 5

Real-time navigation assistance using object detection and depth estimation, optimized for Raspberry Pi 5.

## 🚀 Quick Start

### 1. First-Time Setup

```bash
# SSH into your Pi 5
ssh pi@raspberrypi.local

# Clone/copy this folder to Pi
cd ~/eyeway-depth-measurement/eyeway\ app

# Make startup script executable
chmod +x start.sh

# Run (will auto-install dependencies on first run)
./start.sh
```

### 2. Enable Pi Camera (if using)

```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

## 📁 Files

| File | Description |
|------|-------------|
| `main.py` | Main application entry point |
| `camera.py` | Pi Camera/USB webcam handler (V4L2) |
| `detector.py` | YOLO object detection |
| `depth.py` | MiDaS depth estimation |
| `config.py` | Configuration settings |
| `start.sh` | Startup script |

## ⚙️ Configuration

Edit `config.py` to adjust:

```python
CAMERA_WIDTH = 640       # Lower = faster
CAMERA_HEIGHT = 480
SKIP_FRAMES = 2          # Process every Nth frame
MAX_INFERENCE_SIZE = 384 # Model input size
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle depth estimation |

## 🏃 Running Without Display (Headless)

If running headless via SSH:

```bash
# Set display to dummy
export DISPLAY=:0

# Or disable preview in config.py:
# SHOW_PREVIEW = False
```

## 📊 Performance Tips

1. **Disable depth** for faster detection:
   ```bash
   ./start.sh --no-depth
   ```

2. **Lower resolution** in `config.py`:
   ```python
   CAMERA_WIDTH = 320
   CAMERA_HEIGHT = 240
   ```

3. **Increase frame skip**:
   ```python
   SKIP_FRAMES = 4  # Process every 4th frame
   ```

## 🔧 Troubleshooting

### Camera not found
```bash
# Check devices
ls -la /dev/video*

# Enable camera
sudo raspi-config
```

### Slow performance
- Use `--no-depth` flag
- Lower resolution in config
- Increase SKIP_FRAMES

### Import errors
```bash
# Reinstall dependencies
rm -rf .venv
./start.sh  # Will recreate venv
```

## 📋 Requirements

- Raspberry Pi 5 (8GB recommended)
- Raspberry Pi OS 64-bit
- Pi Camera Module 3 or USB webcam
- Python 3.11+
