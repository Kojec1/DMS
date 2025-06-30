import torch
import torch.nn.functional as F


def _angles_to_unit_vector(pitch_deg: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
    """Convert pitch/yaw to a 3-D unit gaze vector (camera coords)."""
    pitch_rad = torch.deg2rad(pitch_deg)
    yaw_rad = torch.deg2rad(yaw_deg)

    cos_pitch = torch.cos(pitch_rad)
    x = -torch.sin(yaw_rad) * cos_pitch
    y = -torch.sin(pitch_rad)
    z = -torch.cos(yaw_rad) * cos_pitch

    vec = torch.stack([x, y, z], dim=-1)
    vec = F.normalize(vec, dim=-1)

    return vec

def angular_error(
    pred_pitch_deg: torch.Tensor,
    pred_yaw_deg: torch.Tensor,
    gt_pitch_deg: torch.Tensor,
    gt_yaw_deg: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Mean angular error (degrees) between predicted and ground-truth gaze directions."""
    pred_vec = _angles_to_unit_vector(pred_pitch_deg, pred_yaw_deg)
    gt_vec = _angles_to_unit_vector(gt_pitch_deg, gt_yaw_deg)

    # Dot product along last dim
    dot = torch.sum(pred_vec * gt_vec, dim=-1).clamp(-1 + eps, 1 - eps)
    error_rad = torch.acos(dot)
    error_deg = torch.rad2deg(error_rad)

    return error_deg.mean() 