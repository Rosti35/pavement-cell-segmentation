#!/bin/bash

# Pavement Cell Segmentation Setup Script
# This script installs dependencies and creates necessary directories

set -e  # Exit on any error

echo "🚀 Pavement Cell Segmentation - Setup Script"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if Python 3 is installed
echo "=== Checking Python Installation ==="
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_status "Python 3 found: $PYTHON_VERSION"
    PYTHON_CMD="python3"
elif command -v python &> /dev/null && python --version 2>&1 | grep -q "Python 3"; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    print_status "Python 3 found: $PYTHON_VERSION"
    PYTHON_CMD="python"
else
    print_error "Python 3 is required but not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version compatibility
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
    exit 1
fi

# Check if pip is installed
echo -e "\n=== Checking pip Installation ==="
if command -v pip3 &> /dev/null; then
    print_status "pip3 found"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    print_status "pip found"
    PIP_CMD="pip"
else
    print_error "pip is required but not found. Please install pip."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo -e "\n=== Virtual Environment Setup ==="
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
    else
        print_status "Virtual environment already exists"
    fi
    
    print_status "Activating virtual environment..."
    source venv/bin/activate
    PIP_CMD="pip"
    print_status "Virtual environment activated"
fi

# Upgrade pip
echo -e "\n=== Upgrading pip ==="
$PIP_CMD install --upgrade pip
print_status "pip upgraded"

# Install dependencies
echo -e "\n=== Installing Dependencies ==="
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

print_status "Installing dependencies from requirements.txt..."
$PIP_CMD install -r requirements.txt
print_status "Dependencies installed successfully"

# Create necessary directories
echo -e "\n=== Creating Directories ==="
directories=("images" "images/input" "images/output" "images/output/debug")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    else
        print_status "Directory already exists: $dir"
    fi
done

# Check if model file exists
echo -e "\n=== Checking Model File ==="
if [ -f "models/unet_model.onnx" ]; then
    print_status "Model file found: models/unet_model.onnx"
else
    print_warning "Model file not found: models/unet_model.onnx"
    echo "          Please ensure the model file is in the models/ directory"
fi

# Check GPU support
echo -e "\n=== Checking GPU Support ==="
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
else
    print_warning "No NVIDIA GPU detected. Will use CPU for inference."
fi

# Final verification
echo -e "\n=== Verifying Installation ==="
$PYTHON_CMD -c "
try:
    import torch, torchvision, numpy, PIL, matplotlib, onnxruntime
    print('✓ All required packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo -e "\n============================================================"
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo "============================================================"
echo ""
echo "Usage:"
echo "1. Place your input images in the 'input_images/' directory"
echo "2. Run the segmentation:"
echo "   $PYTHON_CMD src/process_images.py input_images output_masks"
echo ""
echo "3. Check results in the 'output_masks/' directory"
echo ""
echo "For Docker usage:"
echo "   docker-compose up pavement-segmentation"
echo ""
echo "Directory structure created:"
for dir in "${directories[@]}"; do
    echo "   $dir/"
done
echo "============================================================"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Note: Virtual environment is activated."
    echo "To deactivate it later, run: deactivate"
fi
