#!/usr/bin/env python3
"""
Test the improved PepLand model components on CUDA without requiring DGL
测试改进的 PepLand 模型组件在 CUDA 上的运行（不需要 DGL）
"""

import torch
import torch.nn as nn
import sys
import os

# Add claude_test to path
sys.path.insert(0, '/home/qlyu/AA_peptide/pepland/claude_test')

def test_model_components():
    """Test individual model components without DGL dependency"""

    print("=" * 60)
    print("Testing Improved PepLand Model Components on CUDA")
    print("=" * 60)

    # Get device
    device = torch.device('cuda:0')  # With CUDA_VISIBLE_DEVICES=3, this is physical GPU 3
    print(f"\nDevice: {device} (Physical GPU 3)")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    try:
        # Test 1: Performer Attention
        print("\n1. Testing Performer Attention:")
        from models.performer import PerformerAttention

        batch_size, seq_len, hidden_dim = 4, 100, 512
        performer = PerformerAttention(hidden_dim=hidden_dim, num_heads=8, nb_features=256).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        out = performer(x)
        print(f"   ✓ Input shape: {x.shape}")
        print(f"   ✓ Output shape: {out.shape}")
        print(f"   ✓ Performer attention working on GPU")

        # Test 2: Hierarchical Pooling
        print("\n2. Testing Hierarchical Pooling:")
        from models.hierarchical_pool import GlobalAttentionPooling, SetTransformer

        # Global attention pooling
        num_nodes, hidden_dim = 50, 512
        pool = GlobalAttentionPooling(hidden_dim).to(device)
        node_features = torch.randn(num_nodes, hidden_dim, device=device)
        pooled = pool(node_features)
        print(f"   ✓ Global Attention Pooling: {node_features.shape} -> {pooled.shape}")

        # Set transformer
        set_pool = SetTransformer(hidden_dim, num_seeds=4).to(device)
        pooled = set_pool(node_features)
        print(f"   ✓ Set Transformer: {node_features.shape} -> {pooled.shape}")

        # Test 3: 3D Conformer Features (if RDKit available)
        print("\n3. Testing 3D Conformer Encoder:")
        try:
            from features.conformer_3d import Conformer3DEncoder

            num_atoms = 20
            coords = torch.randn(num_atoms, 3, device=device)
            encoder = Conformer3DEncoder(hidden_dim=512).to(device)
            features = encoder(coords)
            print(f"   ✓ 3D Conformer Encoder: {coords.shape} -> {features.shape}")
        except ImportError:
            print("   ⚠ RDKit not available, skipping 3D conformer test")

        # Test 4: Physicochemical Encoder
        print("\n4. Testing Physicochemical Encoder:")
        from features.physicochemical import PhysicoChemicalEncoder

        num_molecules = 16
        descriptors = torch.randn(num_molecules, 200, device=device)  # 200 RDKit descriptors
        encoder = PhysicoChemicalEncoder(descriptor_dim=200, hidden_dim=512).to(device)
        features = encoder(descriptors)
        print(f"   ✓ Physicochemical Encoder: {descriptors.shape} -> {features.shape}")

        # Test 5: Sequence Encoder (simple version, no ESM)
        print("\n5. Testing Sequence Encoder:")
        from features.sequence import SimpleSequenceEncoder

        batch_size, seq_len = 8, 50
        sequences = torch.randint(0, 20, (batch_size, seq_len), device=device)  # 20 amino acids
        encoder = SimpleSequenceEncoder(vocab_size=20, hidden_dim=512).to(device)
        features = encoder(sequences)
        print(f"   ✓ Sequence Encoder: {sequences.shape} -> {features.shape}")

        # Test 6: Adapter Fine-tuning
        print("\n6. Testing Adapter Fine-tuning:")
        from finetuning.adapter import Adapter

        adapter = Adapter(hidden_dim=512, adapter_dim=64).to(device)
        x = torch.randn(batch_size, 512, device=device)
        out = adapter(x)
        print(f"   ✓ Adapter: {x.shape} -> {out.shape}")

        # Count trainable parameters
        trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"   ✓ Trainable parameters: {trainable:,}")

        # Test 7: LoRA Fine-tuning
        print("\n7. Testing LoRA Fine-tuning:")
        from finetuning.lora import LoRALayer

        lora = LoRALayer(in_features=512, out_features=512, rank=8).to(device)
        original_weight = torch.randn(512, 512, device=device)
        x = torch.randn(batch_size, 512, device=device)
        out = lora(x, original_weight)
        print(f"   ✓ LoRA: {x.shape} -> {out.shape}")

        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        print(f"   ✓ Trainable parameters: {trainable:,}")

        # Test 8: Memory efficiency
        print("\n8. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        print(f"   Max allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

        # Test 9: Mixed precision
        print("\n9. Testing Mixed Precision (FP16):")
        from torch.cuda.amp import autocast

        with autocast():
            x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            out = performer(x)
            print(f"   ✓ Mixed precision forward pass successful")
            print(f"   ✓ Output dtype: {out.dtype}")

        # Test 10: Gradient computation
        print("\n10. Testing Gradient Computation:")
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)
        out = performer(x)
        loss = out.mean()
        loss.backward()
        print(f"   ✓ Backward pass successful")
        print(f"   ✓ Input gradient shape: {x.grad.shape}")

        print("\n" + "=" * 60)
        print("✓ All component tests passed!")
        print("✓ Ready to train on CUDA device 3")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_components()
    sys.exit(0 if success else 1)
