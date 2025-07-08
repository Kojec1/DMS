import torch
import torch.nn as nn
import math
from typing import Optional

class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi–task loss (Kendall et al., 2018)."""
    def __init__(self, num_tasks: int, init_sigma: float = 1.0, dtype: Optional[torch.dtype] = None):
        super().__init__()
        # Convert initial sigma to log-space so that sigma = exp(log_sigma) > 0.
        init_log_sigma = math.log(init_sigma)
        log_sigma = torch.full((num_tasks,), init_log_sigma, dtype=dtype or torch.get_default_dtype())
        self.log_sigma = nn.Parameter(log_sigma, requires_grad=True)

    def forward(self, *task_losses: torch.Tensor) -> torch.Tensor:
        loss_sum = 0.0
        for i, L_i in enumerate(task_losses):
            sigma_i = torch.exp(self.log_sigma[i])
            precision = 0.5 / (sigma_i ** 2)
            regulariser = torch.log1p(sigma_i ** 2)
            loss_sum = loss_sum + precision * L_i + regulariser

        return loss_sum