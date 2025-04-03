import torch
import sys
import os

# Add the project root directory to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.modules.bottleneck.mobilenet import mobilenet_v2

def example_usage():
    # Create a MobileNetV2 feature extractor
    model = mobilenet_v2(pretrained=True)
    
    # Set to evaluation mode
    model.eval()
    
    # Create a sample input tensor
    batch_size = 1
    x = torch.randn(batch_size, 3, 112, 112)
    
    # Forward pass to extract features
    with torch.no_grad():
        features = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Feature shape: {features.shape}")  # Should be [batch_size, 1280]
    
    custom_head = torch.nn.Linear(1280, 10)  # 10 classes
    outputs = custom_head(features.squeeze(-1).squeeze(-1))
    print(f"Custom output shape: {outputs.shape}")
    
    return features

if __name__ == "__main__":
    example_usage() 