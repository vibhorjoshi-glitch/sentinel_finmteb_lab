#!/bin/bash
# =============================================================================
# NVIDIA CUDA ACTIVATION SCRIPT FOR SENTINEL LAB
# 
# This script activates NVIDIA CUDA from D:\ drive (Windows) or custom path
# and configures PyTorch for GPU acceleration.
# 
# Usage on Windows (from D:\ drive):
#   bash gpu_activate.sh
# 
# Usage on Linux with custom CUDA path:
#   bash gpu_activate.sh /path/to/cuda
# =============================================================================

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     NVIDIA CUDA GPU ACTIVATION FOR SENTINEL LAB        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

# ============================================================================
# STEP 1: Detect CUDA Installation
# ============================================================================

CUDA_PATH=""

# Check if CUDA path is provided as argument
if [ -n "$1" ]; then
    CUDA_PATH="$1"
    echo -e "${BLUE}[1/4]${NC} Using provided CUDA path: ${YELLOW}$CUDA_PATH${NC}"
else
    # Check common CUDA locations
    echo -e "${BLUE}[1/4]${NC} Detecting NVIDIA CUDA installation..."
    
    if [ -d "/usr/local/cuda" ]; then
        CUDA_PATH="/usr/local/cuda"
        echo -e "  ${GREEN}✓${NC} Found CUDA at: ${YELLOW}/usr/local/cuda${NC}"
    elif [ -d "/opt/cuda" ]; then
        CUDA_PATH="/opt/cuda"
        echo -e "  ${GREEN}✓${NC} Found CUDA at: ${YELLOW}/opt/cuda${NC}"
    elif command -v nvidia-smi &> /dev/null; then
        echo -e "  ${YELLOW}⚠${NC} nvidia-smi found but CUDA path not standard"
        CUDA_PATH=$(dirname $(dirname $(which nvidia-smi)))
        echo -e "  ${GREEN}✓${NC} Inferred CUDA path: ${YELLOW}$CUDA_PATH${NC}"
    else
        echo -e "  ${RED}✗${NC} NVIDIA CUDA not found in standard locations"
        echo -e "\n${YELLOW}Installation Instructions:${NC}"
        echo "  1. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
        echo "  2. Install CUDA 12.1 or later (PyTorch 2.9+ requires CUDA 12.1+)"
        echo "  3. Add CUDA to PATH and try again"
        echo ""
        echo "  Or specify custom path:"
        echo "    bash gpu_activate.sh /path/to/cuda"
        exit 1
    fi
fi

# ============================================================================
# STEP 2: Verify CUDA Installation
# ============================================================================

echo -e "\n${BLUE}[2/4]${NC} Verifying CUDA installation..."

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "  ${RED}✗${NC} nvidia-smi not found. CUDA may not be properly installed."
    exit 1
fi

NVIDIA_DEVICES=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo -e "  ${GREEN}✓${NC} Found ${YELLOW}$NVIDIA_DEVICES${NC} NVIDIA GPU(s)"

# Get NVIDIA GPU details
echo ""
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | while read line; do
    echo -e "    GPU: ${YELLOW}$line${NC}"
done

# Get CUDA version from nvidia-smi
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo -e "  ${GREEN}✓${NC} NVIDIA Driver Version: ${YELLOW}$CUDA_VERSION${NC}"

# ============================================================================
# STEP 3: Configure Environment Variables
# ============================================================================

echo -e "\n${BLUE}[3/4]${NC} Configuring environment variables..."

# Set CUDA environment variables
export CUDA_HOME="$CUDA_PATH"
export CUDA_PATH="$CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"

# PyTorch-specific settings
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.2;8.0;8.6;9.0"  # Support wide range of GPU architectures
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

echo -e "  ${GREEN}✓${NC} CUDA_HOME=${YELLOW}$CUDA_HOME${NC}"
echo -e "  ${GREEN}✓${NC} PATH=${YELLOW}$CUDA_PATH/bin:...${NC}"
echo -e "  ${GREEN}✓${NC} LD_LIBRARY_PATH configured"

# ============================================================================
# STEP 4: Test PyTorch GPU Support
# ============================================================================

echo -e "\n${BLUE}[4/4]${NC} Testing PyTorch GPU support..."

python3 << 'EOF'
import torch
import sys

print(f"  PyTorch Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    Device {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Total Memory: {props.total_memory / 1e9:.2f} GB")
    
    # Test GPU computation
    print("\n  Testing GPU computation...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print(f"  ✓ GPU computation successful")
    
    print("\n✓ GPU ACTIVATION SUCCESSFUL!")
    sys.exit(0)
else:
    print("  ✗ CUDA not available in PyTorch")
    print("  Reinstall PyTorch with CUDA support:")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║   ✓ GPU ACTIVATION COMPLETE                           ║${NC}"
    echo -e "${GREEN}║                                                        ║${NC}"
    echo -e "${GREEN}║   You can now run Sentinel Lab with GPU acceleration! ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Environment configured. To use this configuration in new shell:"
    echo "  export CUDA_HOME=\"$CUDA_HOME\""
    echo "  export PATH=\"\$CUDA_HOME/bin:\$PATH\""
    echo "  export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH\""
else
    echo -e "\n${RED}✗ GPU ACTIVATION FAILED${NC}"
    exit 1
fi
