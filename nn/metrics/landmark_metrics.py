import torch
import torch.nn as nn


def NME(predictions: torch.Tensor,
                              ground_truth: torch.Tensor,
                              left_eye_idx: int,
                              right_eye_idx: int,
                              eps: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Normalized Mean Error (NME) for facial landmark estimation,
    normalized by the inter-ocular distance (IOD).
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if ground_truth.dim() == 2:
        ground_truth = ground_truth.unsqueeze(0)

    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"Predictions and ground truth must have the same shape. "
            f"Got predictions: {predictions.shape}, ground_truth: {ground_truth.shape}"
        )
    if predictions.dim() != 3 or predictions.shape[-1] != 2:
        raise ValueError(
            "Input tensors must be of shape (N, K, 2) or (K,2). "
            f"Got shape: {predictions.shape}"
        )

    # Calculate L2 error for each landmark
    # Shape: (N, K)
    landmark_errors = torch.norm(predictions - ground_truth, dim=2, p=2)

    # Calculate mean error per sample
    mean_error_per_sample = torch.mean(landmark_errors, dim=1)

    # Calculate Inter-Ocular Distance (IOD) from ground truth
    left_eye_gt = ground_truth[:, left_eye_idx, :]
    right_eye_gt = ground_truth[:, right_eye_idx, :]

    inter_ocular_distance = torch.norm(left_eye_gt - right_eye_gt, dim=1, p=2)
    inter_ocular_distance = inter_ocular_distance + eps # Add epsilon to prevent division by zero

    # Normalize mean error by IOD
    nme_per_sample = mean_error_per_sample / inter_ocular_distance

    # Return mean NME over the batch
    return torch.mean(nme_per_sample)
