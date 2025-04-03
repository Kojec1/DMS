import torch
import torch.nn as nn
import torch.nn.functional as F

class HPHead(nn.Module):
    def __init__(self) -> None:
        super(HPHead, self).__init__()

        # Classification path
        self.fc_yaw = nn.Linear(64, 12)  # Angle neighbourhood, 12 groups of 15 degrees
        self.fc_roll = nn.Linear(64, 12)
        self.fc_pitch = nn.Linear(64, 12)

        # Regression path
        self.fc_1 = nn.Linear(36, 18)
        self.fc_2 = nn.Linear(18, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_yaw = self.fc_yaw(x)
        x_roll = self.fc_roll(x)
        x_pitch = self.fc_pitch(x)
        x_cls = torch.cat([x_yaw, x_roll, x_pitch], dim=1)

        x_1 = self.fc_1(x_cls)
        x_reg = self.fc_2(x_1)

        return torch.cat([x_cls, x_reg], dim=1)