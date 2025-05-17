import torch
import torch.nn as nn
from nn.modules.bottleneck.convnext import convnext_tiny

class FacialLandmarkEstimator(nn.Module):
    def __init__(self, num_landmarks: int = 6, pretrained_backbone: bool = True) -> None:
        """
        Facial Landmark Estimation model using a ConvNeXt Tiny backbone.

        Args:
            num_landmarks (int): Number of facial landmarks to predict. 
            pretrained_backbone (bool): If True, uses a backbone pre-trained on ImageNet.
        """
        super(FacialLandmarkEstimator, self).__init__()
        
        self.backbone = convnext_tiny(pretrained=pretrained_backbone)        
        self.num_landmarks = num_landmarks
        num_outputs = num_landmarks * 2  # Each landmark has (x, y) coordinates
        
        self.landmark_head = nn.Linear(self.backbone.out_features, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width).
        Returns:
            torch.Tensor: Predicted landmarks of shape (batch_size, num_landmarks * 2).
        """
        features = self.backbone(x)
        # Flatten the features before passing to the linear layer
        features_flattened = torch.flatten(features, 1) # Output shape (batch_size, out_features)
        
        landmarks = self.landmark_head(features_flattened)
        return landmarks

if __name__ == '__main__':
    dummy_input = torch.randn(2, 3, 224, 224)

    model = FacialLandmarkEstimator(num_landmarks=6, pretrained_backbone=True)
    model.eval()
    
    # Perform a forward pass
    with torch.no_grad(): # No need to track gradients for this test
        output_landmarks = model(dummy_input)
    
    # Print the output shape
    # Expected: (batch_size, num_landmarks * 2)
    print("Input shape:", dummy_input.shape)
    print("Backbone out_features:", model.backbone.out_features)
    print("Output landmarks shape:", output_landmarks.shape)
    assert output_landmarks.shape == (2, 6 * 2)

    # Test with a different number of landmarks
    model_8_landmarks = FacialLandmarkEstimator(num_landmarks=8)
    model_8_landmarks.eval()
    with torch.no_grad():
      output_8_landmarks = model_8_landmarks(dummy_input)
    print("Output 8 landmarks shape:", output_8_landmarks.shape)
    assert output_8_landmarks.shape == (2, 8*2)
    print("FacialLandmarkEstimator model created and tested successfully.") 