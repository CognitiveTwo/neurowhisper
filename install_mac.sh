#!/bin/bash
# NeuroWhisper - Mac Installation Script
# This script sets up the virtual environment and installs dependencies

cd "$(dirname "$0")"

echo "========================================="
echo "  NeuroWhisper Mac/Linux Installer"
echo "========================================="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found!"
    echo "Please install Python 3.10 or later from https://www.python.org/"
    exit 1
fi

# Check Python version
PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PY_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Install pynput for Mac keyboard support
echo ""
echo "Installing Mac-specific dependencies..."
pip install pynput

# Additional Mac dependencies for sound
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing pyobjc for Mac sound support..."
    pip install pyobjc-framework-Cocoa 2>/dev/null || true
fi

echo ""
echo "========================================="
echo "  Installation Complete!"
echo "========================================="
echo ""
echo "To run NeuroWhisper:"
echo "  ./run.sh"
echo ""
echo "Note: On Mac, you may need to grant accessibility"
echo "permissions for keyboard shortcuts to work."
echo ""
