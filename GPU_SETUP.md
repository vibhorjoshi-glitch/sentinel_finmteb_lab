# GPU Setup Guide for SENTINEL Project

## System Configuration
- **GPU**: NVIDIA GeForce RTX 3060 (6GB VRAM)
- **CUDA Version**: 13.1
- **Driver**: 591.74
- **OS**: Windows

## Quick Start (5 Minutes)

### Step 1: Install PyTorch with CUDA Support

Open **Command Prompt** or **PowerShell** and run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or with specific versions:

```bash
pip install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
```

### Step 2: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Step 3: Run GPU Test Script

```bash
python test_gpu_setup.py
```

---

## Detailed Setup Instructions

### Option A: Using pip with PyTorch Index (Recommended)

```bash
# Create virtual environment (optional but recommended)
python -m venv sentinel_env
sentinel_env\Scripts\activate

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Option B: Using pip with --extra-index-url

Modify requirements.txt or install directly:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
```

### Option C: Using conda (If you have Anaconda)

```bash
# Create environment
conda create -n sentinel python=3.10
conda activate sentinel

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
pip install -r requirements.txt
```

---

## Troubleshooting

### "CUDA out of memory" Error
- Reduce batch size in your code
- RTX 3060 has 6GB VRAM - keep batches under 64 for Qwen-1.5-2B

### "CUDA driver not found"
- Update NVIDIA drivers from https://www.nvidia.com/drivers
- Your current driver (591.74) supports CUDA 13.1

### "ModuleNotFoundError: No module named 'torch'"
- Reinstall PyTorch with pip command above
- Ensure you're using correct Python environment

### "OMP: Error #15: Initializing libiomp5md.dll"
- This is a known Windows issue with Intel OpenMP
- Fix: `pip install intel-openmp` or restart Python

---

## GPU Performance Tips for SENTINEL

1. **Batch Size**: Start with 32-64 for embedding generation
2. **Mixed Precision**: Add `torch.cuda.amp` for 2x speedup
3. **Memory Management**: Call `torch.cuda.empty_cache()` periodically
4. **Device Setting**: Code auto-detects GPU with `torch.cuda.is_available()`

---

## Verify Your Setup

```python
# Quick verification
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Expected Output:
```
PyTorch: 2.1.1+cu121
CUDA Available: True
GPU: NVIDIA GeForce RTX 3060
VRAM: 6.00 GB
```

---

## Your Project is Ready!

Your `src/config.py` already has GPU detection:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

Your `src/embedder.py` will automatically use GPU when available!

