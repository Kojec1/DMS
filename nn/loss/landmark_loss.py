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


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss that handles different landmark configurations
    and allows for task-specific loss weighting.
    """
    def __init__(self, task_weights=None, omega=0.15, epsilon=0.01):
        super().__init__()
        self.base_loss = SmoothWingLoss(omega=omega, epsilon=epsilon)
        self.task_weights = task_weights or {'mpii': 1.0, 'wflw': 1.0, '300w': 1.0}
        
    def forward(self, predictions_dict, targets_dict):
        total_loss = 0.0
        task_losses = {}
        
        for task in predictions_dict:
            if task in targets_dict:
                task_loss = self.base_loss(predictions_dict[task], targets_dict[task])
                weighted_loss = task_loss * self.task_weights[task]
                total_loss += weighted_loss
                task_losses[task] = task_loss.item()
        
        return total_loss, task_losses
