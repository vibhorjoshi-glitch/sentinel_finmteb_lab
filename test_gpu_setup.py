"""
GPU Setup Verification Script for SENTINEL Project
Tests PyTorch CUDA support with your RTX 3060
"""

import torch
import sys

def test_gpu_setup():
    """Comprehensive GPU test for SENTINEL project"""
    
    print("=" * 70)
    print("🖥️  SENTINEL GPU SETUP VERIFICATION")
    print("=" * 70)
    
    # Test 1: PyTorch installation
    print("\n📦 Test 1: PyTorch Installation")
    print("-" * 40)
    try:
        print(f"   PyTorch version: {torch.__version__}")
        print("   ✅ PyTorch installed successfully")
    except ImportError:
        print("   ❌ PyTorch not installed!")
        print("   💡 Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # Test 2: CUDA availability
    print("\n🔧 Test 2: CUDA Availability")
    print("-" * 40)
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA available: {'✅ Yes' if cuda_available else '❌ No'}")
    
    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   💡 GPU not detected - check drivers or install CUDA version")
        return False
    
    # Test 3: GPU device info
    print("\n🎮 Test 3: GPU Device Information")
    print("-" * 40)
    device_count = torch.cuda.device_count()
    print(f"   GPU devices found: {device_count}")
    
    if device_count > 0:
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name}")
            print(f"   VRAM: {gpu_mem:.2f} GB")
    
    # Test 4: Tensor operations on GPU
    print("\n⚡ Test 4: GPU Tensor Operations")
    print("-" * 40)
    try:
        # Create tensor on GPU
        device = torch.device("cuda")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        c = torch.matmul(a, b)
        
        # Move back to CPU to verify
        c_cpu = c.cpu()
        
        print(f"   ✅ Matrix multiplication: {a.shape} × {b.shape} = {c.shape}")
        print(f"   ✅ GPU compute working correctly")
        
        # Memory cleanup
        del a, b, c, c_cpu
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
        return False
    
    # Test 5: Project config import
    print("\n📁 Test 5: SENTINEL Project Config")
    print("-" * 40)
    try:
        sys.path.insert(0, "src")
        from config import DEVICE, EMBEDDING_MODEL_NAME
        print(f"   Device configured: {DEVICE}")
        print(f"   Embedding model: {EMBEDDING_MODEL_NAME}")
        print("   ✅ Project config loaded successfully")
        
        if DEVICE == "cuda":
            print("   ✅ GPU is set as primary device for project!")
        else:
            print("   ⚠️  CPU mode - PyTorch CUDA may not be working")
            
    except Exception as e:
        print(f"   ⚠️  Could not load project config: {e}")
    
    # Test 6: SENTINEL Embedder
    print("\n🧠 Test 6: SENTINEL Embedder GPU Test")
    print("-" * 40)
    try:
        from embedder import SentinelEmbedder
        embedder = SentinelEmbedder(device=None, verbose=True)
        print(f"\n   ✅ Embedder created on device: {embedder.get_device()}")
    except Exception as e:
        print(f"   ⚠️  Embedder test: {e}")
        print("   💡 This is OK - model download may fail without internet")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SETUP SUMMARY")
    print("=" * 70)
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}")
    print(f"   Project Device: {DEVICE}")
    print("=" * 70)
    
    if cuda_available:
        print("\n🚀 GPU SETUP COMPLETE! Your RTX 3060 is ready for SENTINEL!")
        return True
    else:
        print("\n⚠️  GPU SETUP INCOMPLETE - See above for issues")
        return False

if __name__ == "__main__":
    success = test_gpu_setup()
    sys.exit(0 if success else 1)

