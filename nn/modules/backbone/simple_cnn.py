import torch
import torch.nn as nn
from typing import Optional


class SimpleCNN(nn.Module):
    """
    A simple CNN backbone with 4 convolutional blocks.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB, 1 for grayscale)
        base_channels (int): Base number of channels for the first conv layer
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 32 channels
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224 -> 112x112
            
            # Block 2: 64 channels
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112 -> 56x56
            
            # Block 3: 128 channels
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56 -> 28x28
            
            # Block 4: 256 channels
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            
            # Global average pooling and flatten
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1)
        )
        
        self.out_features = base_channels * 8  # 256 features for default base_channels=32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SimpleCNN backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            torch.Tensor: Feature tensor of shape (batch_size, out_features)
        """
        return self.features(x)


def simple_cnn(in_channels: int = 3, base_channels: int = 32, **kwargs) -> nn.Module:
    """
    Create a SimpleCNN backbone.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB, 1 for grayscale)
        base_channels (int): Base number of channels for the first conv layer
        **kwargs: Additional arguments (for compatibility with other backbones)
    
    Returns:
        nn.Module: SimpleCNN backbone model
    """
    model = SimpleCNN(in_channels=in_channels, base_channels=base_channels)
    return model


if __name__ == '__main__':
    # Test with RGB input
    model_rgb = simple_cnn(in_channels=3, base_channels=32)
    print("SimpleCNN RGB model:")
    print(f"Output features: {model_rgb.out_features}")
    
    dummy_input_rgb = torch.randn(2, 3, 224, 224)
    output_rgb = model_rgb(dummy_input_rgb)
    print(f"Input shape: {dummy_input_rgb.shape}")
    print(f"Output shape: {output_rgb.shape}")
    print()
    
    # Test with grayscale input
    model_gray = simple_cnn(in_channels=1, base_channels=32)
    print("SimpleCNN Grayscale model:")
    print(f"Output features: {model_gray.out_features}")
    
    dummy_input_gray = torch.randn(2, 1, 224, 224)
    output_gray = model_gray(dummy_input_gray)
    print(f"Input shape: {dummy_input_gray.shape}")
    print(f"Output shape: {output_gray.shape}")
    print()
    
    # Test with different base channels
    model_large = simple_cnn(in_channels=3, base_channels=64)
    print("SimpleCNN Large model:")
    print(f"Output features: {model_large.out_features}")
    
    output_large = model_large(dummy_input_rgb)
    print(f"Output shape: {output_large.shape}") 