import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from nn.modules.model import MHModel
from data.dataset import MPIIFaceGazeMatDataset, MPIIFaceGazeDataset


def build_transforms(img_size: int, input_channels: int) -> transforms.Compose:
    if input_channels == 1:
        normalize = transforms.Normalize(mean=[0.449], std=[0.226])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


def load_model(checkpoint_path: str,
               num_landmarks: int,
               input_channels: int,
               num_bins: int,
               backbone: str,
               device: torch.device) -> MHModel:
    model = MHModel(
        num_landmarks=num_landmarks,
        pretrained_backbone=False,
        in_channels=input_channels,
        dropout_rate=0.0,
        num_bins=num_bins,
        num_theta_bins=32,
        num_phi_bins=60,
        backbone=backbone,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys when loading checkpoint ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Warning: Unexpected keys when loading checkpoint ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()
    return model


def _find_last_conv_module(module: nn.Module) -> Optional[nn.Module]:
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _compute_gradcam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    # activations: [B, C, H, W], gradients: [B, C, H, W]
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    cam = torch.relu(cam)
    # Normalize per-sample
    B, _, H, W = cam.shape
    cam_reshaped = cam.view(B, -1)
    cam_min = cam_reshaped.min(dim=1, keepdim=True)[0].unsqueeze(-1)
    cam_max = cam_reshaped.max(dim=1, keepdim=True)[0].unsqueeze(-1)
    cam_norm = (cam_reshaped - cam_min) / (cam_max - cam_min + 1e-8)
    cam = cam_norm.view(B, 1, H, W)
    return cam


def _overlay_heatmap_on_image(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # img_np: HxW or HxWx3 in [0,1]; heatmap: HxW in [0,1]
    from matplotlib import cm
    if img_np.ndim == 2:
        img_color = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        img_color = img_np
    cmap = plt.colormaps['jet']
    heatmap_color = cmap(np.clip(heatmap, 0.0, 1.0))[:, :, :3]
    overlay = (1 - alpha) * img_color + alpha * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def visualize_attribution(
    model: MHModel,
    image_tensor: torch.Tensor,
    orig_image_np: np.ndarray,
    target_head: str,
    device: torch.device,
    backbone: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (heatmap_resized, overlay_rgb) as numpy arrays in [0,1].
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Try Grad-CAM on last conv layer (CNN backbones). Fallback to input-grad saliency otherwise.
    conv_module = _find_last_conv_module(model.backbone)

    activations = []
    gradients = []
    handles = []

    if conv_module is not None and backbone in ('mobilenet', 'convnext'):
        def fwd_hook(_, __, output):
            activations.append(output)
        def bwd_hook(_, grad_in, grad_out):
            gradients.append(grad_out[0])
        handles.append(conv_module.register_forward_hook(fwd_hook))
        handles.append(conv_module.register_full_backward_hook(bwd_hook))

        image_tensor.requires_grad_(True)
        outputs = model(image_tensor)
        # Choose scalar target based on head
        if target_head == 'landmarks':
            landmarks = outputs[0]
            target_scalar = (landmarks ** 2).sum()
        elif target_head == 'yaw':
            yaw_logits = outputs[1]
            target_scalar = yaw_logits.max(dim=1)[0].sum()
        elif target_head == 'pitch':
            pitch_logits = outputs[2]
            target_scalar = pitch_logits.max(dim=1)[0].sum()
        else:  # combined
            # lmk = (outputs[0] ** 2).sum()
            yaw = outputs[1].max(dim=1)[0].sum()
            pit = outputs[2].max(dim=1)[0].sum()
            target_scalar = yaw + pit

        model.zero_grad(set_to_none=True)
        if image_tensor.grad is not None:
            image_tensor.grad.zero_()
        target_scalar.backward()

        # Compute CAM
        act = activations[-1].detach()
        grad = gradients[-1].detach()
        cam = _compute_gradcam(act, grad)[0, 0]  # HxW

        for h in handles:
            h.remove()

        # Resize CAM to original image size
        import cv2
        H, W = orig_image_np.shape[:2]
        cam_resized = cv2.resize(cam.cpu().numpy(), (W, H), interpolation=cv2.INTER_CUBIC)
        overlay = _overlay_heatmap_on_image(orig_image_np, cam_resized)
        return cam_resized, overlay

    # Fallback: input-gradient saliency
    image_tensor.requires_grad_(True)
    outputs = model(image_tensor)
    if target_head == 'landmarks':
        target_scalar = (outputs[0] ** 2).sum()
    elif target_head == 'yaw':
        target_scalar = outputs[1].max(dim=1)[0].sum()
    elif target_head == 'pitch':
        target_scalar = outputs[2].max(dim=1)[0].sum()
    else:
        target_scalar = outputs[1].max(dim=1)[0].sum() + outputs[2].max(dim=1)[0].sum()

    model.zero_grad(set_to_none=True)
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    target_scalar.backward()

    grad = image_tensor.grad.detach()[0]  # [C,H,W]
    # Take abs, channel-wise max
    sal = grad.abs().max(dim=0)[0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

    heatmap = sal.cpu().numpy()
    overlay = _overlay_heatmap_on_image(orig_image_np, heatmap)
    return heatmap, overlay


def main():
    parser = argparse.ArgumentParser(description="Feature attention visualization for MHModel on MPIIFaceGaze")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root path to MPIIFaceGaze dataset (mat or raw)')
    parser.add_argument('--use_mat', action='store_true', help='Use MPIIFaceGazeMatDataset (.mat files)')
    parser.add_argument('--participant_id', type=int, default=0, help='Participant ID (0-14) to sample from')
    parser.add_argument('--index', type=int, default=0, help='Global sample index within the filtered dataset')
    parser.add_argument('--img_size', type=int, default=224, help='Model input size')
    parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3], help='Number of input channels')
    parser.add_argument('--num_landmarks', type=int, default=6, help='Number of landmarks the model predicts')
    parser.add_argument('--num_bins', type=int, default=14, help='Number of bins for yaw/pitch heads')
    parser.add_argument('--backbone', type=str, default='mobilenet', choices=['mobilenet', 'convnext', 'tinyvit'], help='Backbone architecture')
    parser.add_argument('--target_head', type=str, default='landmarks', choices=['landmarks', 'yaw', 'pitch', 'combined'], help='Which head to target for attribution')
    parser.add_argument('--output', type=str, default='misc/feature_attention.png', help='Where to save the visualization image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = build_transforms(args.img_size, args.input_channels)

    # Dataset
    if args.use_mat:
        dataset = MPIIFaceGazeMatDataset(
            dataset_path=args.dataset_root,
            participant_ids=[args.participant_id],
            transform=transform,
            input_channels=args.input_channels,
            use_cache=False,
            use_clahe=False,
            downscale_size=args.img_size,
            affine_aug=False,
            horizontal_flip=False,
        )
    else:
        dataset = MPIIFaceGazeDataset(
            dataset_path=args.dataset_root,
            participant_ids=[args.participant_id],
            transform=transform,
            is_train=False,
            use_cache=False,
            input_channels=args.input_channels,
            use_clahe=False,
        )

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty; check dataset_root and participant_id")

    sample = dataset[args.index % len(dataset)]
    img_t: torch.Tensor = sample['image']  # [C,H,W]

    # Build an unnormalized image for background
    # Reverse normalization for display
    if args.input_channels == 1:
        mean = np.array([0.449], dtype=np.float32)
        std = np.array([0.226], dtype=np.float32)
        img_np = img_t.squeeze(0).cpu().numpy() * std[0] + mean[0]
    else:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = img_t.cpu().numpy().transpose(1, 2, 0) * std + mean
    img_np = np.clip(img_np, 0.0, 1.0)

    # Model
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_landmarks=args.num_landmarks,
        input_channels=args.input_channels,
        num_bins=32,
        backbone=args.backbone,
        device=device,
    )

    # Attribution
    heatmap, overlay = visualize_attribution(
        model=model,
        image_tensor=img_t,
        orig_image_np=img_np if img_np.ndim == 3 else img_np,
        target_head=args.target_head,
        device=device,
        backbone=args.backbone,
    )

    # Save visualization
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.figure(figsize=(4, 4))
    # plt.subplot(1, 3, 1)
    # if img_np.ndim == 2:
    #     plt.imshow(img_np, cmap='gray')
    # else:
    #     plt.imshow(img_np)
    # plt.axis('off')
    # plt.title('Input')

    # plt.subplot(1, 3, 2)
    # plt.imshow(heatmap, cmap='jet')
    # plt.axis('off')
    # plt.title('Attribution')

    # plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Overlay')

    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()
    print(f"Saved visualization to {args.output}")


if __name__ == '__main__':
    main() 