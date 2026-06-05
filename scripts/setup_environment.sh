#!/bin/bash
#
# DSRQS: Environment Setup Script
# For Ubuntu 22.04 LTS server (as in paper)
#

set -e

echo "=========================================="
echo "DSRQS Environment Setup"
echo "=========================================="
echo ""

# Check Ubuntu version
if ! grep -q "Ubuntu 22.04" /etc/os-release; then
    echo "WARNING: This script is designed for Ubuntu 22.04 LTS"
    echo "Current OS:"
    cat /etc/os-release
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo "Updating package lists..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y python3.10 python3.10-dev python3-pip git wget curl unzip -y

# Set Python 3.10 as default
echo "Setting Python 3.10 as default..."
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --config python3

# Install PyTorch with CUDA 12.1
echo "Installing PyTorch 2.1.2 with CUDA 12.1..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data runs checkpoints results

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

python3 -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate your environment"
echo "2. Run: python reproducibility_check.py"
echo "3. Download datasets: python scripts/download_datasets.py"
echo ""
