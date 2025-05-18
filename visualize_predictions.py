import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import random
from PIL import Image

from nn.modules.facial_landmark_estimator import FacialLandmarkEstimator
from data.dataset import MPIIFaceGazeDataset
from utils.misc import set_seed

# ImageNet mean and std for unnormalization
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def get_args():
    parser = argparse.ArgumentParser(description='Visualize predictions from Facial Landmark Estimator')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Root directory for MPIIFaceGaze dataset (containing p00, p01, etc.)')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--grid_size', type=str, default="2x3", 
                        help='Grid size for display, e.g., "2x3" for 2 rows, 3 columns')
    parser.add_argument('--participant_ids', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14", 
                        help='Comma-separated list of participant IDs to sample from (e.g., "0,1,2")')
    parser.add_argument('--num_landmarks', type=int, default=6, 
                        help='Number of facial landmarks (MPIIFaceGaze has 6)')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Input image size used during training (e.g., 224)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility of sample selection')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    
    return parser.parse_args()

def unnormalize_image(tensor):
    """Unnormalize a tensor image with ImageNet mean and std."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    return tensor

def load_model_for_visualization(checkpoint_path, num_landmarks, device):
    model = FacialLandmarkEstimator(num_landmarks=num_landmarks, pretrained_backbone=False)
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle potential keys like 'model_state_dict' or direct state_dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    else:
        model_state_dict = checkpoint

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def plot_image_and_landmarks(ax, image_tensor, true_landmarks, pred_landmarks, img_size):
    """Plots the image with true and predicted landmarks."""
    # Unnormalize and prepare image for display
    img_display = unnormalize_image(image_tensor.cpu())
    img_display = img_display.numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
    img_display = np.clip(img_display, 0, 1) # Ensure values are in [0,1] for display

    ax.imshow(img_display)
    
    # Reshape to (num_landmarks, 2) for plotting
    true_lmks = true_landmarks.cpu().numpy().reshape(-1, 2)
    pred_lmks = pred_landmarks.cpu().numpy().reshape(-1, 2)

    ax.plot(true_lmks[:, 0], true_lmks[:, 1], 'o', color='lime', markersize=3, label='Ground Truth')
    ax.plot(pred_lmks[:, 0], pred_lmks[:, 1], 'x', color='red', markersize=3, label='Predicted')
    
    ax.axis('off') # Hide axes ticks

def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Parse grid size
    try:
        rows, cols = map(int, args.grid_size.split('x'))
        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive.")
    except ValueError as e:
        print(f"Error: Invalid grid_size format '{args.grid_size}'. Expected 'rowsxcols', e.g., '2x3'. {e}")
        return
    
    num_samples_to_display = rows * cols

    # Load Model
    model = load_model_for_visualization(args.checkpoint_path, args.num_landmarks, device)
    if model is None:
        return

    # Data Transformations
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    vis_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Dataset and DataLoader
    try:
        p_ids_str = args.participant_ids.split(',')
        if not all(p.strip().isdigit() for p in p_ids_str if p.strip()): # check if all are digits
             raise ValueError("Participant IDs must be integers.")
        vis_participant_ids = [int(p_id) for p_id in p_ids_str if p_id.strip()]
        if not vis_participant_ids:
            raise ValueError("No participant IDs provided or all are empty.")
    except ValueError as e:
        print(f"Error: Invalid participant_ids format: '{args.participant_ids}'. {e}")
        return

    print(f"Attempting to load data for participant IDs: {vis_participant_ids}")
    
    try:
        full_dataset = MPIIFaceGazeDataset(
            dataset_path=args.data_dir,
            participant_ids=vis_participant_ids,
            transform=vis_transform
        )
    except Exception as e:
        print(f"Error initializing MPIIFaceGazeDataset: {e}")
        print("Please ensure the dataset path and participant IDs are correct, and the dataset structure is as expected.")
        return

    if not full_dataset.samples:
        print(f"Error: No samples found for participants {vis_participant_ids} in {args.data_dir}.")
        print("Check --data_dir and --participant_ids. Ensure participant subfolders (e.g., p00, p01) exist and contain data.")
        return
    
    print(f"Loaded {len(full_dataset)} samples from {len(vis_participant_ids)} participant(s).")

    # Select random samples
    if len(full_dataset) < num_samples_to_display:
        print(f"Warning: Requested {num_samples_to_display} samples, but dataset only has {len(full_dataset)}. Displaying all available.")
        num_samples_to_display = len(full_dataset)
        # Adjust grid size if fewer samples than cells
        if num_samples_to_display == 0:
            print("No samples to display.")
            return
        cols = min(cols, num_samples_to_display)
        rows = (num_samples_to_display + cols -1) // cols


    random_indices = random.sample(range(len(full_dataset)), num_samples_to_display)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows * cols == 1: # if single image, axes is not an array
        axes = np.array([axes])
    axes = axes.flatten() # Flatten to easily iterate

    print(f"Displaying {num_samples_to_display} random samples in a {rows}x{cols} grid...")

    for i in range(num_samples_to_display):
        sample_idx = random_indices[i]
        try:
            # This sample contains the transformed image and original landmarks
            sample_transformed = full_dataset[sample_idx]
        except Exception as e:
            print(f"Error fetching sample at index {sample_idx}: {e}")
            axes[i].text(0.5, 0.5, "Error loading sample", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].axis('off')
            continue

        # Get original image path to load its dimensions for scaling landmarks
        try:
            raw_sample_metadata = full_dataset.samples[sample_idx]
            original_image_path = raw_sample_metadata['image_path']
            with Image.open(original_image_path) as img_original:
                original_width, original_height = img_original.size
        except Exception as e:
            print(f"Error getting original image dimensions for sample {sample_idx} ({original_image_path}): {e}")
            axes[i].text(0.5, 0.5, "Error scaling GT", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].axis('off')
            continue

        image_tensor_resized = sample_transformed['image'].to(device)

        # Define the normalization factor used by MPIIFaceGazeDataset
        # MPIIFaceGazeDataset normalizes original pixel coordinates by its self.img_size (default 224)
        # DATASET_NORMALIZATION_FACTOR = 224.0 # This was incorrect. Dataset normalizes by original image dimensions.

        # --- Ground Truth Landmarks --- 
        # sample_transformed['facial_landmarks'] are normalized by original image dimensions (range [0,1])
        true_landmarks_from_dataset = sample_transformed['facial_landmarks'].clone().float().view(-1, 2)
        
        scaled_true_landmarks = true_landmarks_from_dataset.clone()
        # Scale normalized [0,1] coordinates to the displayed image size (args.img_size)
        scaled_true_landmarks[:, 0] = true_landmarks_from_dataset[:, 0] * args.img_size
        scaled_true_landmarks[:, 1] = true_landmarks_from_dataset[:, 1] * args.img_size
        true_landmarks_for_plot = scaled_true_landmarks.view(-1) # Flatten for plotting function

        # --- Predicted Landmarks --- 
        # Model expects batch dimension.
        # Assume model outputs landmarks normalized to its input image space (range [0,1]), relative to args.img_size.
        image_batch = image_tensor_resized.unsqueeze(0)
        with torch.no_grad():
            pred_landmarks_from_model_flat = model(image_batch).squeeze(0).cpu() # Shape (num_landmarks * 2)
        
        pred_landmarks_from_model = pred_landmarks_from_model_flat.view(-1, 2) # Reshape to (num_landmarks, 2)
        
        scaled_pred_landmarks = pred_landmarks_from_model.clone()
        # Scale normalized [0,1] coordinates to the displayed image size (args.img_size)
        scaled_pred_landmarks[:, 0] = pred_landmarks_from_model[:, 0] * args.img_size
        scaled_pred_landmarks[:, 1] = pred_landmarks_from_model[:, 1] * args.img_size
        pred_landmarks_for_plot = scaled_pred_landmarks.view(-1) # Flatten for plotting function
        
        # plot_image_and_landmarks expects landmarks in pixel coordinates of the displayed image
        plot_image_and_landmarks(axes[i], image_tensor_resized.cpu(), true_landmarks_for_plot, pred_landmarks_for_plot, args.img_size)
        axes[i].set_title(f"Sample {sample_idx} (Orig: {original_width}x{original_height})")

    # Add a single legend for the entire figure
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles and labels:
         fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01)) # Adjust position


    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.suptitle(f"Landmark Predictions ({args.grid_size} grid)", fontsize=16)
    
    # Save or show
    output_filename = "landmark_visualizations.png"
    plt.savefig(output_filename)
    print(f"Visualization saved to {output_filename}")
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot (possibly running in a headless environment): {e}")
        print(f"Please check the saved file: {output_filename}")

if __name__ == '__main__':
    main() 