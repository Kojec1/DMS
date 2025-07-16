import torch
import torch.nn as nn
from torchvision.models.convnext import LayerNorm2d
from nn.modules.bottleneck.convnext import convnext_tiny
from functools import partial


class MHModel(nn.Module):
    def __init__(self,
                 num_landmarks: int = 6,
                 pretrained_backbone: bool = True,
                 in_channels: int = 1,
                 dropout_rate: float = 0.2,
                 num_bins: int = 14,
                 num_theta_bins: int = 14,
                 num_phi_bins: int = 14) -> None:
        """Multi-Head model for landmarks and gaze estimation."""
        super().__init__()

        self.backbone = convnext_tiny(pretrained=pretrained_backbone)
        self.num_landmarks = num_landmarks
        self.pretrained_backbone = pretrained_backbone
        self.in_channels = in_channels
        self.num_bins = num_bins
        self.num_theta_bins = num_theta_bins
        self.num_phi_bins = num_phi_bins

        num_outputs = num_landmarks * 2  # Each landmark has (x, y)
        norm_layer = partial(LayerNorm2d, eps=1e-6)

        # Replace first conv for grayscale (if necessary)
        if self.in_channels == 1:
            self._replace_first_conv_layer()

        # Normalization layers
        self.landmark_norm = norm_layer(self.backbone.out_features)
        self.gaze_norm = norm_layer(self.backbone.out_features)
        self.head_pose_norm = norm_layer(self.backbone.out_features)

        self.dropout = nn.Dropout(dropout_rate)

        # Heads
        self.landmark_head = nn.Linear(self.backbone.out_features, num_outputs)
        self.yaw_head = nn.Linear(self.backbone.out_features, num_bins)
        self.pitch_head = nn.Linear(self.backbone.out_features, num_bins)
        self.theta_head = nn.Linear(self.backbone.out_features, num_theta_bins)
        self.phi_head = nn.Linear(self.backbone.out_features, num_phi_bins)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.backbone(x)

        # Facial landmarks head
        landmarks = self.landmark_norm(features)
        landmarks = self.dropout(torch.flatten(landmarks, 1))
        landmarks = self.landmark_head(landmarks)

        # Gaze head
        gaze = self.gaze_norm(features)
        gaze = self.dropout(torch.flatten(gaze, 1))
        yaw_logits = self.yaw_head(gaze)
        pitch_logits = self.pitch_head(gaze)

        # Head pose head
        head_pose = self.head_pose_norm(features)
        head_pose = self.dropout(torch.flatten(head_pose, 1))
        theta_logits = self.theta_head(head_pose)
        phi_logits = self.phi_head(head_pose)

        return landmarks, yaw_logits, pitch_logits, theta_logits, phi_logits
    
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
            