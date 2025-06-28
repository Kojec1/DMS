import torch
import torch.nn as nn
from nn.modules.bottleneck.convnext import convnext_tiny

class MHModel(nn.Module):
    def __init__ (self, num_landmarks: int = 6, pretrained_backbone: bool = True, in_channels: int = 1, dropout_rate: float = 0.2) -> None:
        """
        Multi-Head Model for Facial Landmark Esimation and 2D Gaze Estimation using a ConvNeXt Tiny backbone.

        """
        super(MHModel, self).__init__()

        self.backbone = convnext_tiny(pretrained=pretrained_backbone)
        self.num_landmarks = num_landmarks
        self.pretrained_backbone = pretrained_backbone
        self.in_channels = in_channels
        num_outputs = num_landmarks * 2  # Each landmark has (x, y) coordinates

        # Replace the first conv layer for grayscale input
        if self.in_channels == 1:
            self._replace_first_conv_layer()

        self.dropout = nn.Dropout(dropout_rate)
        self.landmark_head = nn.Linear(self.backbone.out_features, num_outputs)
        self.gaze_2d_head = nn.Linear(self.backbone.out_features, 2)
        self.gaze_3d_head = nn.Linear(self.backbone.out_features, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Returns a tuple of (landmarks, gaze).
        """
        features = self.backbone(x)
        features_flattened = torch.flatten(features, 1)
        features_flattened = self.dropout(features_flattened)

        landmarks = self.landmark_head(features_flattened)
        gaze_2d = self.gaze_2d_head(features_flattened)
        gaze_3d = self.gaze_3d_head(features_flattened) 

        return landmarks, gaze_2d, gaze_3d
    
    def _replace_first_conv_layer(self) -> None:
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

        if self.pretrained_backbone:
            original_weights = original_first_conv_layer.weight.data
            # Average RGB weights for the single grayscale channel
            adapted_weights = original_weights.mean(dim=1, keepdim=True)
            new_first_layer.weight.data = adapted_weights
            if original_first_conv_layer.bias is not None:
                new_first_layer.bias.data = original_first_conv_layer.bias.data.clone()

        self.backbone[0][0][0] = new_first_layer
            