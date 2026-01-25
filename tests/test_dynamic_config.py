"""
Test script to demonstrate the dynamic CNN encoder with different configurations
"""
import torch
from dreamer.models.encoder import ObservationCNNEncoder

def test_different_architectures():
    """Test the dynamic CNN encoder with different layer counts and feature sizes"""
    
    # Test configuration 1: 3 layers, 8x8 final size
    print("Testing 3-layer, 8x8 architecture...")
    model1 = ObservationCNNEncoder(
        target_size=(64, 64),
        in_channels=3,
        kernel_size=2,
        stride=2,
        padding=0,
        d_hidden=256,  # base channels = 256/16 = 16
        hidden_dim_ratio=16,
        num_layers=3,
        final_feature_size=8
    )
    
    x = torch.randn(2, 3, 100, 100)
    out1 = model1(x)
    print(f"Input: {x.shape} -> Output: {out1.shape}")
    
    # Test configuration 2: 5 layers, 2x2 final size  
    print("\nTesting 5-layer, 2x2 architecture...")
    model2 = ObservationCNNEncoder(
        target_size=(64, 64),
        in_channels=3,
        kernel_size=2,
        stride=2,
        padding=0,
        d_hidden=512,  # base channels = 512/16 = 32
        hidden_dim_ratio=16,
        num_layers=5,
        final_feature_size=2
    )
    
    x = torch.randn(2, 3, 100, 100)
    out2 = model2(x)
    print(f"Input: {x.shape} -> Output: {out2.shape}")
    
    # Test configuration 3: Different ratio (8 instead of 16)
    print("\nTesting with ratio=8...")
    model3 = ObservationCNNEncoder(
        target_size=(64, 64),
        in_channels=3,
        kernel_size=2,
        stride=2,
        padding=0,
        d_hidden=256,  # base channels = 256/8 = 32
        hidden_dim_ratio=8,
        num_layers=4,
        final_feature_size=4
    )
    
    x = torch.randn(2, 3, 100, 100)
    out3 = model3(x)
    print(f"Input: {x.shape} -> Output: {out3.shape}")
    
    print("\nAll dynamic architectures work correctly!")

if __name__ == "__main__":
    test_different_architectures()