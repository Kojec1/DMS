import torch
import sys
import os

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.modules.backbone.convnext import convnext_tiny

def example_usage():
    # Create a ConvNeXt Tiny feature extractor
    model = convnext_tiny(pretrained=True)
    
    # Set to evaluation mode
    model.eval()
    
    # Create a sample input tensor
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass to extract features
    with torch.no_grad():
        features = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Feature shape: {features.shape}")  # Should be [batch_size, 768]
    
    custom_head = torch.nn.Linear(768, 10)  # Example: 10 classes
    outputs = custom_head(features.squeeze(-1).squeeze(-1))
    print(f"Custom output shape: {outputs.shape}")
    
    return features

if __name__ == "__main__":
    example_usage() 