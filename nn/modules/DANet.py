import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import HPHead

class DANet(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super(DANet, self).__init__()
        self.backbone = backbone

        self.fc1 = nn.Linear(backbone.out_features, 254)
        self.fc_head = nn.Linear(254, 64)
        self.fc_gaze = nn.Linear(254, 64)

        self.hphead = HPHead()  # Head Pose 

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc1(x)

        # Head Pose 
        x_head = self.fc_head(x)
        x_head = self.hphead(x_head)

        # Eye Gaze
        x_gaze = self.fc_gaze(x)    

        return torch.cat([x_head, x_gaze], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
