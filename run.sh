#!/bin/bash
# NeuroWhisper - Mac/Linux Run Script
# This script activates the virtual environment and runs the application

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found!"
    echo "Please run ./install_mac.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the application
python whisper_gui.pyw
