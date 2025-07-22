import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
import math
from PIL import Image

from nn.modules.facial_landmark_estimator import FacialLandmarkEstimator
from data.dataset import MPIIFaceGazeDataset, WFLWDataset, Face300WDataset
from utils.misc import set_seed
from nn.metrics.landmark_metrics import NME
from nn.modules.model import MHModel  # Multi-head model for landmarks + gaze
from nn.metrics.gaze_metrics import angular_error


def get_args():
    parser = argparse.ArgumentParser(description='Visualize predictions from Facial Landmark Estimator')
    parser.add_argument('--dataset', type=str, required=True, choices=['mpii', 'wflw', '300w'],
                        help='Dataset to use: mpii for MPIIFaceGaze, wflw for WFLW, 300w for 300W')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Root directory for dataset')
    parser.add_argument('--annotation_file', type=str, 
                        help='Annotation file path (required for WFLW)')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--grid_size', type=str, default="2x3", 
                        help='Grid size for display, e.g., "2x3" for 2 rows, 3 columns')
    parser.add_argument('--participant_ids', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14", 
                        help='Comma-separated list of participant IDs (for MPII only)')
    parser.add_argument('--num_landmarks', type=int, default=6, 
                        help='Number of facial landmarks')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Input image size used during training')
    parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3],
                        help='Number of input image channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--display_mode', type=str, default="all", choices=["all", "correct", "incorrect"],
                        help='Which samples to display based on NME')
    parser.add_argument('--nme_threshold', type=float, default=0.1, 
                        help='Threshold for NME to be considered correct')
    parser.add_argument('--left_eye_idx', type=int, default=0,
                        help='Index of the left eye landmark for NME calculation')
    parser.add_argument('--right_eye_idx', type=int, default=3,
                        help='Index of the right eye landmark for NME calculation')
    parser.add_argument('--mpii_landmarks', action='store_true',
                        help='Extract MPII-style 6 landmarks from WFLW/300W (for WFLW and 300W only)')
    parser.add_argument('--subset', type=str, choices=['indoor', 'outdoor'], 
                        help='Subset to load for 300W dataset (indoor/outdoor, default: both)')
    parser.add_argument('--padding_ratio', type=float, default=0.3,
                        help='Padding ratio for landmark-based cropping in 300W dataset (default: 0.3)')
    parser.add_argument('--translation_ratio', type=float, default=0.2,
                        help='Random translation ratio for landmark-based cropping in 300W dataset (default: 0.0, no translation)')
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='Train/test split ratio for 300W dataset (default: 0.8)')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for reproducible train/test splits in 300W dataset (default: 42)')
    parser.add_argument('--data_split', type=str, choices=['train', 'test'], default='test',
                        help='Which split to visualize for 300W dataset (default: test)')
    parser.add_argument('--tasks', type=str, default='landmarks',
                        help='Comma-separated list of tasks to visualise: "landmarks", "gaze"')
    parser.add_argument('--annotation_mode', type=str, default='both',
                        choices=['none', 'gt', 'pred', 'both'],
                        help='What annotations to overlay on images: none | gt | pred | both')
    parser.add_argument('--num_angle_bins', type=int, default=14,
                        help='Number of angle bins for yaw/pitch heads (MHModel)')
    parser.add_argument('--angle_bin_width', type=float, default=3.0,
                        help='Bin width (degrees) used during training (MHModel)')
    
    return parser.parse_args()


def load_model_for_visualization(checkpoint_path, num_landmarks, device, in_channels, tasks, num_angle_bins):
    """Instantiate and load model depending on requested tasks.

    • If 'gaze' is in tasks, we assume a multi-head model (landmarks + gaze).
    • Otherwise, fall back to the landmark-only estimator.
    """

    want_gaze = 'gaze' in tasks

    if want_gaze:
        model = MHModel(
            num_landmarks=num_landmarks,
            pretrained_backbone=False,
            in_channels=in_channels,
            num_bins=num_angle_bins,
        )
    else:
        model = FacialLandmarkEstimator(
            num_landmarks=num_landmarks,
            pretrained_backbone=False,
            in_channels=in_channels,
        )

    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # State-dict handling
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Try strict loading first; if it fails, fall back to non-strict (for safety)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: strict load failed ({e}). Trying non-strict load…")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f" → Missing keys: {missing}")
        if unexpected:
            print(f" → Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    model_type = 'MHModel' if want_gaze else 'FacialLandmarkEstimator'
    print(f"Loaded {model_type} with {in_channels} input channel(s).")
    return model

def plot_image_and_landmarks(
    ax,
    image_tensor_cpu,
    true_landmarks,
    pred_landmarks,
    current_mean,
    current_std,
    show_gt: bool = True,
    show_pred: bool = True,
):
    """Plot image and optionally overlay landmarks."""
    # Unnormalize the image tensor
    img_unnormalized = image_tensor_cpu.clone()
    # Ensure current_mean and current_std are tensors for consistent indexing
    if not isinstance(current_mean, torch.Tensor): current_mean = torch.tensor(current_mean)
    if not isinstance(current_std, torch.Tensor): current_std = torch.tensor(current_std)

    for c_idx in range(img_unnormalized.size(0)):
        img_unnormalized[c_idx].mul_(current_std[c_idx]).add_(current_mean[c_idx])
    
    img_display_np = img_unnormalized.cpu().numpy()

    # Prepare for display
    if img_display_np.shape[0] == 1:  # Grayscale (1, H, W)
        img_display_np = img_display_np.squeeze(0)
    elif img_display_np.shape[0] == 3:  # RGB (3, H, W)
        img_display_np = img_display_np.transpose(1, 2, 0)
    else:
        print(f"Warning: Unexpected image tensor shape for display: {img_display_np.shape}")

    img_display_np = np.clip(img_display_np, 0, 1)  # Ensure values are in [0,1] for display

    # Display image
    if img_display_np.ndim == 2 or (img_display_np.ndim == 3 and img_display_np.shape[-1] == 1):
        ax.imshow(img_display_np, cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(img_display_np, vmin=0, vmax=1)

    # If no annotations requested, finish here
    if not show_gt and not show_pred:
        ax.axis('off')
        return

    true_lmks = true_landmarks.cpu().numpy().reshape(-1, 2)
    pred_lmks = pred_landmarks.cpu().numpy().reshape(-1, 2)

    if show_gt:
        ax.plot(true_lmks[:, 0], true_lmks[:, 1], 'o', color='lime', markersize=3, label='Ground Truth')
    if show_pred:
        ax.plot(pred_lmks[:, 0], pred_lmks[:, 1], 'x', color='red', markersize=3, label='Predicted')
    
    ax.axis('off') # Hide axes ticks

def _draw_gaze_arrow(ax, pitch_rad: float, yaw_rad: float, origin_xy: tuple[float, float], length: float = 40.0, color: str = 'blue') -> None:
    """Draw a 2-D arrow approximating gaze direction given pitch/yaw (in *radians*)."""
    ox, oy = origin_xy
    dx = -length * np.sin(yaw_rad)
    dy = -length * np.sin(pitch_rad)
    ax.arrow(ox, oy, dx, dy, color=color, head_width=3, head_length=4, linewidth=1.5)

def main():
    args = get_args()

    set_seed(42)

    if args.dataset == 'wflw' and not args.annotation_file:
        print("Error: --annotation_file is required for WFLW dataset")
        return

    if args.dataset == '300w':
        if not (0.0 < args.train_test_split < 1.0):
            print(f"Error: train_test_split must be between 0.0 and 1.0, got {args.train_test_split}")
            return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    if args.input_channels == 1:
        CURRENT_MEAN = [0.449]
        CURRENT_STD = [0.226]
        print("Using 1-channel (grayscale) normalization.")
    else:
        CURRENT_MEAN = [0.485, 0.456, 0.406]
        CURRENT_STD = [0.229, 0.224, 0.225]
        print("Using 3-channel (RGB) normalization.")

    # Parse grid size
    try:
        original_rows, original_cols = map(int, args.grid_size.split('x'))
        if original_rows <= 0 or original_cols <= 0:
            raise ValueError("Grid dimensions must be positive.")
    except ValueError as e:
        print(f"Error: Invalid grid_size format '{args.grid_size}'. Expected 'rowsxcols', e.g., '2x3'. {e}")
        return
    
    target_num_samples_to_display = original_rows * original_cols

    # Parse tasks and annotation preferences
    tasks_list = [t.strip().lower() for t in args.tasks.split(',') if t.strip()]
    tasks_list = [t for t in tasks_list if t in {'landmarks', 'gaze'}]
    if not tasks_list:
        print("No valid tasks specified. Nothing to visualise (images only).")

    show_gt_flag = args.annotation_mode in {'gt', 'both'}
    show_pred_flag = args.annotation_mode in {'pred', 'both'}

    if args.display_mode in {'correct', 'incorrect'} and 'landmarks' not in tasks_list:
        print("Error: display_mode 'correct'/'incorrect' requires landmark predictions. Either include 'landmarks' in --tasks or change display_mode.")
        return

    # Load Model
    model = load_model_for_visualization(
        args.checkpoint_path,
        args.num_landmarks,
        device,
        args.input_channels,
        tasks_list,
        args.num_angle_bins,
    )
    if model is None:
        return

    # Data Transformations
    data_normalize = transforms.Normalize(mean=CURRENT_MEAN, std=CURRENT_STD)
    vis_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        data_normalize,
    ])

    try:
        if args.dataset == 'mpii':
            p_ids_str = args.participant_ids.split(',')
            if not all(p.strip().isdigit() for p in p_ids_str if p.strip()):
                raise ValueError("Participant IDs must be integers.")
            vis_participant_ids = [int(p_id) for p_id in p_ids_str if p_id.strip()]
            if not vis_participant_ids:
                raise ValueError("No participant IDs provided.")
            
            print(f"Loading MPII data for participant IDs: {vis_participant_ids}")
            full_dataset = MPIIFaceGazeDataset(
                dataset_path=args.data_dir,
                participant_ids=vis_participant_ids,
                transform=vis_transform,
                is_train=False,
                use_cache=True,
                affine_aug=False,
                flip_aug=False
            )
            landmark_key = 'facial_landmarks'

        elif args.dataset == 'wflw':
            print(f"Loading WFLW data from {args.annotation_file}")
            full_dataset = WFLWDataset(
                annotation_file=args.annotation_file,
                images_dir=args.data_dir,
                transform=vis_transform,
                is_train=False,
                mpii_landmarks=args.mpii_landmarks
            )
            landmark_key = 'landmarks'

        elif args.dataset == '300w':
            subset_str = f" ({args.subset} subset)" if args.subset else " (both subsets)"
            split_str = f" ({args.data_split} split)"
            print(f"Loading 300W data from {args.data_dir}{subset_str}{split_str}")
            full_dataset = Face300WDataset(
                root_dir=args.data_dir,
                subset=args.subset,
                transform=vis_transform,
                is_train=False,
                mpii_landmarks=args.mpii_landmarks,
                padding_ratio=args.padding_ratio,
                translation_ratio=args.translation_ratio,
                train_test_split=args.train_test_split,
                split=args.data_split,
                split_seed=args.split_seed
            )
            landmark_key = 'landmarks'
        else:
            print(f"Error: Invalid dataset '{args.dataset}'")
            return
            
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    if not full_dataset.samples:
        print(f"Error: No samples found in dataset.")
        return
    
    print(f"Loaded {len(full_dataset)} samples from {args.dataset} dataset.")

    final_indices_to_plot = []
    
    if args.display_mode == "all":
        if len(full_dataset) == 0:
            print("No samples in dataset to display.")
            return
        
        num_to_sample_all_mode = min(target_num_samples_to_display, len(full_dataset))
        if len(full_dataset) < target_num_samples_to_display:
             print(f"Warning: Requested {target_num_samples_to_display} samples, but dataset only has {len(full_dataset)}. Displaying {num_to_sample_all_mode}.")
        
        if num_to_sample_all_mode > 0:
            final_indices_to_plot = random.sample(range(len(full_dataset)), num_to_sample_all_mode)
        print(f"Selected {len(final_indices_to_plot)} random samples for display (mode: all).")

    else:
        print(f"Finding '{args.display_mode}' samples...")
        
        num_landmarks_for_nme = args.num_landmarks
        if num_landmarks_for_nme <= max(args.left_eye_idx, args.right_eye_idx):
            print(f"Error: num_landmarks ({num_landmarks_for_nme}) must be greater than eye indices ({args.left_eye_idx}, {args.right_eye_idx}).")
            return

        all_possible_indices = list(range(len(full_dataset)))
        random.shuffle(all_possible_indices)
        
        attempted_count = 0
        for current_random_idx in all_possible_indices:
            if len(final_indices_to_plot) >= target_num_samples_to_display:
                break

            attempted_count += 1
            try:
                sample = full_dataset[current_random_idx]
                img_tensor_for_model = sample['image'].to(device).unsqueeze(0)
                
                true_landmarks_nme = sample[landmark_key].clone().float().view(1, num_landmarks_for_nme, 2).to(device)

                with torch.no_grad():
                    model_out = model(img_tensor_for_model)

                # Determine model output structure
                if isinstance(model_out, tuple):
                    pred_landmarks_from_model_normalized_flat = model_out[0].squeeze(0).cpu()

                    # Gaze predictions (if requested)
                    if 'gaze' in tasks_list:
                        yaw_logits = model_out[1].squeeze(0)
                        pitch_logits = model_out[2].squeeze(0)

                        yaw_probs = torch.softmax(yaw_logits, dim=0)
                        pitch_probs = torch.softmax(pitch_logits, dim=0)

                        offsets = torch.arange(args.num_angle_bins, device=yaw_probs.device).float() - (args.num_angle_bins - 1) / 2.0

                        yaw_pred_deg = args.angle_bin_width * torch.sum(yaw_probs * offsets).item()
                        pitch_pred_deg = args.angle_bin_width * torch.sum(pitch_probs * offsets).item()

                        yaw_pred_rad = np.deg2rad(yaw_pred_deg)
                        pitch_pred_rad = np.deg2rad(pitch_pred_deg)
                else:
                    pred_landmarks_from_model_normalized_flat = model_out.squeeze(0).cpu()
                    if 'gaze' in tasks_list:
                        yaw_pred_rad = None
                        pitch_pred_rad = None
                        print("Warning: Model does not output gaze predictions.")

                pred_landmarks_nme = pred_landmarks_from_model_normalized_flat.view(1, num_landmarks_for_nme, 2).to(device)

                nme_value = NME(predictions=pred_landmarks_nme, 
                                ground_truth=true_landmarks_nme,
                                left_eye_idx=args.left_eye_idx,
                                right_eye_idx=args.right_eye_idx)
                nme_value_item = nme_value.item()
                is_correct = nme_value_item < args.nme_threshold
                
                if (args.display_mode == "correct" and is_correct) or (args.display_mode == "incorrect" and not is_correct):
                    final_indices_to_plot.append(current_random_idx)

            except Exception as e:
                print(f"Skipping sample index {current_random_idx}: {e}")
                continue
        
        if not final_indices_to_plot:
            print(f"No samples found matching display_mode '{args.display_mode}' with NME threshold {args.nme_threshold}.")
            return
        
        print(f"Found {len(final_indices_to_plot)} suitable '{args.display_mode}' samples after checking {attempted_count} images.")

    actual_num_samples_to_display = len(final_indices_to_plot)

    if actual_num_samples_to_display == 0:
        print("No samples selected for display.")
        return

    display_cols = min(original_cols, actual_num_samples_to_display)
    display_rows = (actual_num_samples_to_display + display_cols - 1) // display_cols
    
    fig, axes = plt.subplots(display_rows, display_cols, figsize=(display_cols * 4, display_rows * 4))
    if actual_num_samples_to_display == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    print(f"Displaying {actual_num_samples_to_display} samples in {display_rows}x{display_cols} grid (Mode: {args.display_mode})...")

    for i in range(actual_num_samples_to_display):
        sample_idx_in_full_dataset = final_indices_to_plot[i]
        try:
            sample_transformed = full_dataset[sample_idx_in_full_dataset]
        except Exception as e:
            print(f"Error fetching sample at index {sample_idx_in_full_dataset}: {e}")
            axes[i].text(0.5, 0.5, "Error loading sample", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].axis('off')
            continue

        # Get original image path to load its dimensions for scaling landmarks
        try:
            raw_sample_metadata = full_dataset.samples[sample_idx_in_full_dataset]
            original_image_path = raw_sample_metadata['image_path']
            with Image.open(original_image_path) as img_original:
                original_width, original_height = img_original.size
        except Exception as e:
            print(f"Error getting original image dimensions for sample {sample_idx_in_full_dataset}: {e}")
            axes[i].text(0.5, 0.5, "Error scaling GT", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].axis('off')
            continue
        
        image_tensor_resized = sample_transformed['image'].to(device)
        true_landmarks_from_dataset_normalized = sample_transformed[landmark_key].clone().float()
        
        image_batch_for_model = image_tensor_resized.unsqueeze(0)
        with torch.no_grad():
            model_out = model(image_batch_for_model)

        # Determine model output structure
        if isinstance(model_out, tuple):
            pred_landmarks_from_model_normalized_flat = model_out[0].squeeze(0).cpu()

            # Gaze predictions
            if 'gaze' in tasks_list:
                yaw_logits = model_out[1].squeeze(0)
                pitch_logits = model_out[2].squeeze(0)

                yaw_probs = torch.softmax(yaw_logits, dim=0)
                pitch_probs = torch.softmax(pitch_logits, dim=0)

                offsets = torch.arange(args.num_angle_bins, device=yaw_probs.device).float() - (args.num_angle_bins - 1) / 2.0

                yaw_pred_deg = args.angle_bin_width * torch.sum(yaw_probs * offsets).item()
                pitch_pred_deg = args.angle_bin_width * torch.sum(pitch_probs * offsets).item()

                yaw_pred_rad = np.deg2rad(yaw_pred_deg)
                pitch_pred_rad = np.deg2rad(pitch_pred_deg)
        else:
            pred_landmarks_from_model_normalized_flat = model_out.squeeze(0).cpu()
            if 'gaze' in tasks_list:
                yaw_pred_rad = None
                pitch_pred_rad = None
                print("Warning: Model does not output gaze predictions.")

        pred_landmarks_nme = pred_landmarks_from_model_normalized_flat.view(1, args.num_landmarks, 2).to(device)

        nme_value_for_display = -1.0
        try:
            true_lmks_nme_shape = true_landmarks_from_dataset_normalized.view(1, args.num_landmarks, 2).to(device)
            pred_lmks_nme_shape = pred_landmarks_nme.view(1, args.num_landmarks, 2).to(device)
            
            if args.num_landmarks > max(args.left_eye_idx, args.right_eye_idx):
                nme_value_for_display = NME(predictions=pred_lmks_nme_shape, 
                                            ground_truth=true_lmks_nme_shape,
                                            left_eye_idx=args.left_eye_idx,
                                            right_eye_idx=args.right_eye_idx).item()

        except Exception as e_nme:
            print(f"Error calculating NME for sample {sample_idx_in_full_dataset}: {e_nme}")

        scaled_true_landmarks = true_landmarks_from_dataset_normalized.view(-1, 2).clone()
        scaled_true_landmarks[:, 0] *= args.img_size
        scaled_true_landmarks[:, 1] *= args.img_size
        true_landmarks_for_plot = scaled_true_landmarks.view(-1) 

        scaled_pred_landmarks = pred_landmarks_from_model_normalized_flat.view(-1, 2).clone()
        scaled_pred_landmarks[:, 0] *= args.img_size
        scaled_pred_landmarks[:, 1] *= args.img_size
        pred_landmarks_for_plot = scaled_pred_landmarks.view(-1)
        
        # Draw image
        if 'landmarks' in tasks_list:
            plot_image_and_landmarks(
                axes[i],
                image_tensor_resized.cpu(),
                true_landmarks_for_plot,
                pred_landmarks_for_plot,
                CURRENT_MEAN,
                CURRENT_STD,
                show_gt=show_gt_flag,
                show_pred=show_pred_flag,
            )
        else:
            # Just show image
            plot_image_and_landmarks(
                axes[i],
                image_tensor_resized.cpu(),
                torch.empty(0),
                torch.empty(0),
                CURRENT_MEAN,
                CURRENT_STD,
                show_gt=False,
                show_pred=False,
            )

        # Overlay gaze arrows
        if 'gaze' in tasks_list and ('gaze_2d_angles' in sample_transformed):
            origin = (args.img_size / 2.0, args.img_size / 2.0)
            if show_gt_flag:
                gt_pitch_rad = sample_transformed['gaze_2d_angles'][0].item()
                gt_yaw_rad = sample_transformed['gaze_2d_angles'][1].item()
                g = np.array([-np.sin(gt_yaw_rad), np.sin(gt_pitch_rad), -np.cos(gt_yaw_rad)])
                g = g / np.linalg.norm(g)
                gt_pitch_rad = np.arcsin(g[1])
                gt_yaw_rad = np.arctan2(-g[0], -g[2])
                _draw_gaze_arrow(axes[i], gt_pitch_rad, gt_yaw_rad, origin, length=40.0, color='lime')
            if show_pred_flag and 'gaze' in tasks_list and isinstance(model_out, tuple):
                if yaw_pred_rad is not None and pitch_pred_rad is not None:
                    _draw_gaze_arrow(axes[i], pitch_pred_rad, yaw_pred_rad, origin, length=40.0, color='red')

        # Angular error (gaze)
        angular_error_val = None
        if 'gaze' in tasks_list and yaw_pred_rad is not None and ('gaze_2d_angles' in sample_transformed):
            gt_pitch_rad = sample_transformed['gaze_2d_angles'][0].item()
            gt_yaw_rad = sample_transformed['gaze_2d_angles'][1].item()

            try:
                ae_tensor = angular_error(
                    pred_pitch_deg=torch.tensor([math.degrees(pitch_pred_rad)]),
                    pred_yaw_deg=torch.tensor([math.degrees(yaw_pred_rad)]),
                    gt_pitch_deg=torch.tensor([math.degrees(gt_pitch_rad)]),
                    gt_yaw_deg=torch.tensor([math.degrees(gt_yaw_rad)]),
                )
                angular_error_val = ae_tensor.item()
            except Exception as e_ae:
                print(f"Error computing angular error for sample {sample_idx_in_full_dataset}: {e_ae}")

        title_str = f"ID: {sample_idx_in_full_dataset}"
        if angular_error_val is not None:
            title_str += f" AE: {angular_error_val:.2f}°"
        if nme_value_for_display >= 0:
            title_str += f" NME: {nme_value_for_display:.4f}"
        axes[i].set_title(title_str)

    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles and labels:
         fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(f"Landmark Predictions - {args.dataset.upper()} ({args.grid_size} grid)", fontsize=16)
    
    output_filename = f"landmark_visualizations_{args.dataset}.png"
    plt.savefig(output_filename)
    print(f"Visualization saved to {output_filename}")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
        print(f"Please check the saved file: {output_filename}")

if __name__ == '__main__':
    main() 