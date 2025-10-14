#!/usr/bin/env python3
"""
Simple CUDA test without DGL to verify GPU 3 is accessible
简单的 CUDA 测试，不使用 DGL，验证 GPU 3 是否可用
"""

import torch
import sys

def test_cuda():
    """Test basic CUDA functionality on device 3"""

    print("=" * 60)
    print("Testing CUDA Device 3")
    print("=" * 60)

    # 1. Check PyTorch CUDA availability
    print("\n1. PyTorch CUDA Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("   ❌ CUDA is not available!")
        return False

    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    # 2. Check GPU 3
    gpu_id = 3
    if torch.cuda.device_count() <= gpu_id:
        print(f"   ❌ GPU {gpu_id} not available! Only {torch.cuda.device_count()} GPUs found.")
        # Try to use GPU 0 instead
        gpu_id = 0
        print(f"   → Will try to use GPU {gpu_id} instead")

    print(f"\n2. GPU {gpu_id} Information:")
    print(f"   Name: {torch.cuda.get_device_name(gpu_id)}")

    device = torch.device(f'cuda:{gpu_id}')
    print(f"   Device: {device}")

    # 3. Test tensor operations
    print(f"\n3. Testing tensor operations on GPU {gpu_id}:")

    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        print(f"   ✓ Created tensors: x.shape={x.shape}, y.shape={y.shape}")
        print(f"   ✓ Tensor device: {x.device}")

        # Matrix multiplication
        z = torch.matmul(x, y)
        print(f"   ✓ Matrix multiplication: z.shape={z.shape}")

        # Test some operations
        mean = z.mean()
        std = z.std()
        print(f"   ✓ Statistics: mean={mean.item():.4f}, std={std.item():.4f}")

        # Test neural network operations
        print("\n4. Testing neural network operations:")
        linear = torch.nn.Linear(1000, 512).to(device)
        out = linear(x)
        print(f"   ✓ Linear layer: out.shape={out.shape}")

        relu = torch.nn.ReLU()
        out = relu(out)
        print(f"   ✓ ReLU activation applied")

        # Test with gradients
        out.mean().backward()
        print(f"   ✓ Backward pass successful")
        print(f"   ✓ Gradient computed: linear.weight.grad.shape={linear.weight.grad.shape}")

        # Memory info
        print(f"\n5. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB")
        print(f"   Reserved: {torch.cuda.memory_reserved(gpu_id) / 1024**2:.2f} MB")
        print(f"   Max allocated: {torch.cuda.max_memory_allocated(gpu_id) / 1024**2:.2f} MB")

        print("\n" + "=" * 60)
        print(f"✓ All tests passed! CUDA device {gpu_id} is working")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cuda()
    sys.exit(0 if success else 1)
