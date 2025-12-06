#!/bin/bash

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install torch tqdm psutil lion-pytorch

# Verify installation
if python -c "import torch; import tqdm; import psutil" 2>/dev/null; then
    echo "Installation successful!"
else
    echo "Error: Some packages failed to install."
    exit 1
fi
pip install 
