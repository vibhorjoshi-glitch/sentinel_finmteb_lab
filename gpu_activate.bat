@echo off
REM =============================================================================
REM NVIDIA CUDA ACTIVATION SCRIPT FOR SENTINEL LAB (Windows)
REM 
REM This script activates NVIDIA CUDA from D:\ drive (or custom path)
REM and configures PyTorch for GPU acceleration on Windows.
REM 
REM Usage:
REM   gpu_activate.bat                    (uses D:\cuda by default)
REM   gpu_activate.bat C:\path\to\cuda    (uses custom CUDA path)
REM =============================================================================

setlocal enabledelayedexpansion

color 0B
title NVIDIA CUDA GPU Activation for Sentinel Lab

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║     NVIDIA CUDA GPU ACTIVATION FOR SENTINEL LAB        ║
echo ╚════════════════════════════════════════════════════════╝
echo.

REM =========================================================================
REM STEP 1: Detect CUDA Installation
REM =========================================================================

set CUDA_PATH=
if not "%1"=="" (
    set CUDA_PATH=%1
    echo [1/4] Using provided CUDA path: %CUDA_PATH%
) else (
    echo [1/4] Detecting NVIDIA CUDA installation...
    
    REM Check D:\ drive first (as requested)
    if exist "D:\cuda\" (
        set CUDA_PATH=D:\cuda
        echo   ✓ Found CUDA at: D:\cuda
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1" (
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
        echo   ✓ Found CUDA at: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
        set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
        echo   ✓ Found CUDA at: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
    ) else (
        echo   ✗ NVIDIA CUDA not found in standard locations
        echo.
        echo Installation Instructions:
        echo   1. Download CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
        echo   2. Install CUDA 12.1 or later (PyTorch 2.9+ requires CUDA 12.1+)
        echo   3. Add CUDA to PATH or run: gpu_activate.bat "C:\path\to\cuda"
        echo.
        pause
        exit /b 1
    )
)

REM =========================================================================
REM STEP 2: Verify CUDA Installation
REM =========================================================================

echo.
echo [2/4] Verifying CUDA installation...

where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    REM Try from CUDA path
    if exist "%CUDA_PATH%\bin\nvidia-smi.exe" (
        set PATH=%CUDA_PATH%\bin;!PATH!
    ) else (
        echo   ✗ nvidia-smi not found. CUDA may not be properly installed.
        pause
        exit /b 1
    )
)

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > temp_gpu_info.txt
for /f "tokens=1,2 delims=," %%A in (temp_gpu_info.txt) do (
    echo   ✓ GPU: %%A
    echo   ✓ Driver Version: %%B
)
del temp_gpu_info.txt

REM =========================================================================
REM STEP 3: Configure Environment Variables
REM =========================================================================

echo.
echo [3/4] Configuring environment variables...

set CUDA_HOME=%CUDA_PATH%
set PATH=%CUDA_PATH%\bin;%PATH%
set PATH=%CUDA_PATH%\libnvvp;%PATH%

if exist "%CUDA_PATH%\lib\x64" (
    set PATH=%CUDA_PATH%\lib\x64;%PATH%
)

set TORCH_CUDA_ARCH_LIST=6.0;6.1;7.0;7.2;8.0;8.6;9.0
set CUDA_DEVICE_ORDER=PCI_BUS_ID

echo   ✓ CUDA_HOME=%CUDA_HOME%
echo   ✓ PATH configured with CUDA binaries
echo   ✓ PyTorch CUDA architecture list configured

REM =========================================================================
REM STEP 4: Test PyTorch GPU Support
REM =========================================================================

echo.
echo [4/4] Testing PyTorch GPU support...

python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); torch.cuda.is_available() and print(f'CUDA Device Count: {torch.cuda.device_count()}') and [print(f'  Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>nul

if %errorlevel% equ 0 (
    echo.
    echo   ✓ GPU computation test...
    python << PYTHON_EOF
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.matmul(x, y)
    print('   ✓ GPU computation successful')
else:
    print('   ✗ CUDA not available in PyTorch')
    print('   Reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu121')
PYTHON_EOF
) else (
    echo   ✗ PyTorch GPU test failed
)

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║   ✓ GPU ACTIVATION COMPLETE                           ║
echo ║                                                        ║
echo ║   You can now run Sentinel Lab with GPU acceleration! ║
echo ╚════════════════════════════════════════════════════════╝
echo.
echo Environment Variables Set:
echo   CUDA_HOME=%CUDA_HOME%
echo   PATH includes CUDA bin directory
echo.
echo To use this configuration in a new command prompt:
echo   set CUDA_HOME=%CUDA_HOME%
echo   set PATH=%CUDA_PATH%\bin;%%PATH%%
echo.
pause
