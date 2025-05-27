import torch
import torch.nn as nn


class SmoothWingLoss(nn.Module):
    """
    SmoothWingLoss is a loss function that is used to train the landmark detection model.
    """
    def __init__(self, omega=0.15, epsilon=0.01):
        super(SmoothWingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        mask = diff < self.omega

        # Nonlinear region
        loss_nonlinear = self.omega * torch.log(1 + diff[mask] / self.epsilon)

        # Linear region with non-negative enforcement
        linear_offset = self.omega - self.omega * torch.log(torch.tensor(1 + self.omega / self.epsilon))
        loss_linear = diff[~mask] - linear_offset.clamp(min=0)

        # Combine losses
        return torch.mean(torch.cat([loss_nonlinear, loss_linear]))
