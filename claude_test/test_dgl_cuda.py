#!/usr/bin/env python3
"""
Test script to verify DGL can use CUDA device 3
测试 DGL 是否能使用 CUDA 设备 3
"""

import torch
import dgl
import sys

def test_dgl_cuda():
    """Test DGL with CUDA support"""

    print("=" * 60)
    print("Testing DGL with CUDA")
    print("=" * 60)

    # 1. Check PyTorch CUDA availability
    print("\n1. PyTorch CUDA Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("   ❌ CUDA is not available in PyTorch!")
        return False

    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")

    # Check if GPU 3 exists
    if torch.cuda.device_count() <= 3:
        print(f"   ❌ GPU 3 not available! Only {torch.cuda.device_count()} GPUs found.")
        return False

    print(f"   GPU 3 name: {torch.cuda.get_device_name(3)}")

    # 2. Check DGL
    print("\n2. DGL Information:")
    print(f"   DGL version: {dgl.__version__}")

    # 3. Test creating a graph on CUDA device 3
    print("\n3. Testing DGL graph on CUDA device 3:")

    try:
        # Set device to CUDA:3
        device = torch.device('cuda:3')
        print(f"   Using device: {device}")

        # Create a simple graph
        # Graph: 0 -> 1 -> 2
        #        0 -> 2
        u = torch.tensor([0, 1, 0])
        v = torch.tensor([1, 2, 2])
        g = dgl.graph((u, v))

        print(f"   Created graph: {g}")
        print(f"   Number of nodes: {g.num_nodes()}")
        print(f"   Number of edges: {g.num_edges()}")

        # Move graph to CUDA:3
        g = g.to(device)
        print(f"   ✓ Graph moved to {device}")
        print(f"   Graph device: {g.device}")

        # Add node features
        g.ndata['h'] = torch.randn(g.num_nodes(), 64, device=device)
        print(f"   ✓ Added node features (shape: {g.ndata['h'].shape})")
        print(f"   Node features device: {g.ndata['h'].device}")

        # Add edge features
        g.edata['e'] = torch.randn(g.num_edges(), 32, device=device)
        print(f"   ✓ Added edge features (shape: {g.edata['e'].shape})")
        print(f"   Edge features device: {g.edata['e'].device}")

        # Test message passing
        import dgl.function as fn
        g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_new'))
        print(f"   ✓ Message passing successful")
        print(f"   New node features shape: {g.ndata['h_new'].shape}")

        # Test batch of graphs
        print("\n4. Testing batched graphs on CUDA device 3:")
        graphs = [dgl.rand_graph(10, 20).to(device) for _ in range(8)]
        batched_g = dgl.batch(graphs)
        print(f"   ✓ Created batched graph: {batched_g}")
        print(f"   Total nodes: {batched_g.num_nodes()}")
        print(f"   Total edges: {batched_g.num_edges()}")
        print(f"   Batch size: {batched_g.batch_size}")

        # Test heterogeneous graph
        print("\n5. Testing heterogeneous graph on CUDA device 3:")
        hetero_g = dgl.heterograph({
            ('atom', 'bond', 'atom'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 0])),
            ('atom', 'in', 'fragment'): (torch.tensor([0, 1]), torch.tensor([0, 0]))
        })
        hetero_g = hetero_g.to(device)
        print(f"   ✓ Created heterogeneous graph: {hetero_g}")
        print(f"   Node types: {hetero_g.ntypes}")
        print(f"   Edge types: {hetero_g.etypes}")
        print(f"   Device: {hetero_g.device}")

        # Memory info
        print("\n6. GPU Memory Usage:")
        print(f"   Allocated: {torch.cuda.memory_allocated(3) / 1024**2:.2f} MB")
        print(f"   Reserved: {torch.cuda.memory_reserved(3) / 1024**2:.2f} MB")

        print("\n" + "=" * 60)
        print("✓ All tests passed! DGL works with CUDA device 3")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dgl_cuda()
    sys.exit(0 if success else 1)
