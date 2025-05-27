import torch
import torch.nn as nn


class SmoothWingLoss(nn.Module):
    """
    SmoothWingLoss is a loss function that is used to train the landmark detection model.
    """
    def __init__(self, omega=0.1, epsilon=0.02):
        super(SmoothWingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(diff < self.epsilon, self.omega * (diff - self.epsilon / 2), self.omega * (diff + self.epsilon / 2))
        return loss.mean()
    