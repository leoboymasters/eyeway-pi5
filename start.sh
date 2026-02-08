#!/bin/bash
# Eyeway Startup Script - Raspberry Pi 5
# =======================================
# This script sets up the environment and launches Eyeway.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "================================================"
echo "  EYEWAY - Raspberry Pi 5"
echo "  Navigation Assistance System"
echo "================================================"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found!${NC}"
    echo "Install with: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Check/create virtual environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    
    echo -e "${YELLOW}Installing dependencies...${NC}"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

# Check for camera
echo -e "${YELLOW}Checking camera...${NC}"
if [ -e /dev/video0 ]; then
    echo -e "${GREEN}Camera found at /dev/video0${NC}"
else
    echo -e "${RED}Warning: No camera found at /dev/video0${NC}"
    echo "Make sure Pi Camera is enabled (sudo raspi-config) or USB camera is connected"
fi

# Parse arguments
NO_DEPTH=""
CAMERA="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-depth)
            NO_DEPTH="--no-depth"
            shift
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./start.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-depth    Disable depth estimation (faster)"
            echo "  --camera N    Use camera index N (default: 0)"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Launch application
echo ""
echo -e "${GREEN}Starting Eyeway...${NC}"
echo ""

python3 main.py --camera "$CAMERA" $NO_DEPTH

# Cleanup
deactivate 2>/dev/null || true
