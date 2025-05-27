import torch
import torch.nn as nn
from nn.modules.bottleneck.convnext import convnext_tiny

class FacialLandmarkEstimator(nn.Module):
    def __init__(self, num_landmarks: int = 6, pretrained_backbone: bool = True, in_channels: int = 3, dropout_rate: float = 0.2) -> None:
        """
        Facial Landmark Estimation model using a ConvNeXt Tiny backbone.

        Args:
            num_landmarks (int): Number of facial landmarks to predict.
            pretrained_backbone (bool): If True, uses a backbone pre-trained on ImageNet.
            in_channels (int): Number of input channels for the image (1 for grayscale, 3 for RGB).
        """
        super(FacialLandmarkEstimator, self).__init__()
        
        self.backbone = convnext_tiny(pretrained=pretrained_backbone)
        self.num_landmarks = num_landmarks
        self.in_channels = in_channels
        num_outputs = num_landmarks * 2  # Each landmark has (x, y) coordinates

        # Replace the first conv layer for grayscale input
        if self.in_channels == 1:
            original_first_conv_layer = self.backbone[0][0][0]

            new_first_layer = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=original_first_conv_layer.out_channels,
                kernel_size=original_first_conv_layer.kernel_size,
                stride=original_first_conv_layer.stride,
                padding=original_first_conv_layer.padding,
                dilation=original_first_conv_layer.dilation,
                groups=original_first_conv_layer.groups,
                bias=(original_first_conv_layer.bias is not None)
            )

            if pretrained_backbone:
                original_weights = original_first_conv_layer.weight.data
                # Average RGB weights for the single grayscale channel
                adapted_weights = original_weights.mean(dim=1, keepdim=True)
                new_first_layer.weight.data = adapted_weights
                if original_first_conv_layer.bias is not None:
                    new_first_layer.bias.data = original_first_conv_layer.bias.data.clone()
                print(f"Adapted pretrained weights of first conv layer for {self.in_channels} input channel(s).")
            
            # Replace the nn.Conv2d layer within the Conv2dNormActivation block
            self.backbone[0][0][0] = new_first_layer

        self.dropout = nn.Dropout(dropout_rate)
        self.landmark_head = nn.Linear(self.backbone.out_features, num_outputs) # Access out_features from the actual features sequence

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Predicted landmarks of shape (batch_size, num_landmarks * 2).
        """
        features = self.backbone(x)
        # Flatten the features before passing to the linear layer
        features_flattened = torch.flatten(features, 1) # Output shape (batch_size, out_features)
        features_flattened = self.dropout(features_flattened)
        landmarks = self.landmark_head(features_flattened) # Pass features directly

        return landmarks


if __name__ == '__main__':
    dummy_input = torch.randn(2, 1, 224, 224)

    model = FacialLandmarkEstimator(num_landmarks=6, pretrained_backbone=True, in_channels=1)
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
    model_8_landmarks = FacialLandmarkEstimator(num_landmarks=8, pretrained_backbone=False, in_channels=1)
    model_8_landmarks.eval()
    with torch.no_grad():
      output_8_landmarks = model_8_landmarks(dummy_input)
    print("Output 8 landmarks shape:", output_8_landmarks.shape)
    assert output_8_landmarks.shape == (2, 8*2)
    print("FacialLandmarkEstimator model created and tested successfully.") 