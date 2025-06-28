import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
import time
from tqdm import tqdm

from nn.modules.model import MHModel
from nn.loss import SmoothWingLoss
from nn.metrics import NME
from data.dataset import MPIIFaceGazeDataset, WFLWDataset, Face300WDataset
from utils.visualization import plot_training_history
from utils.misc import set_seed, setup_device
from utils.checkpoint import save_checkpoint, load_checkpoint, save_history, load_history


# Configuration
def get_args():
    parser = argparse.ArgumentParser(description='Facial Landmark Estimation Training')
    parser.add_argument('--dataset', type=str, required=True, choices=['mpii', 'wflw', '300w'],
                        help='Dataset to use: mpii for MPIIFaceGaze, wflw for WFLW, 300w for Face300W')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for dataset')
    parser.add_argument('--annotation_file', type=str, 
                        help='Annotation file path (required for WFLW)')
    parser.add_argument('--num_landmarks', type=int, default=6, help='Number of facial landmarks')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size for ConvNeXt')
    parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3], help='Number of input image channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_final', type=float, default=1e-6, help='Final learning rate for linear scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_mh', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Frequency (in epochs) to save checkpoints. Best model is always saved.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval for logging training status (batches)')
    
    # Optimized DataLoader settings for reduced CPU overhead
    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for DataLoader')
    parser.add_argument('--prefetch_factor', type=int, default=6, help='Number of batches to prefetch per worker')
    parser.add_argument('--persistent_workers', action='store_true', help='Keep DataLoader workers alive between epochs')
    
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')

    # Model arguments
    parser.add_argument('--no_pretrained_backbone', action='store_true', help='Do not use pretrained backbone weights')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the model')

    # Augmentation arguments
    parser.add_argument('--affine_aug', action='store_true', help='Use affine augmentation')
    parser.add_argument('--flip_aug', action='store_true', help='Use flip augmentation')
    parser.add_argument('--label_smoothing', type=float, default=0.01, help='Label smoothing for the dataset')
    parser.add_argument('--use_cache', action='store_true', help='Use cached images and landmarks')

    # Warmup arguments
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of epochs for warmup phase.')
    parser.add_argument('--warmup_lr', type=float, default=1e-3, help='Learning rate during warmup phase.')
    parser.add_argument('--freeze_backbone_warmup', action='store_true', help='Freeze backbone weights during warmup phase.')
    
    # Training Mode arguments
    parser.add_argument('--training_mode', type=str, default='landmarks,gaze_2d', help='Comma-separated training modes. Available modes: landmarks, gaze_2d, gaze_3d. Examples: "landmarks", "gaze_2d", "landmarks,gaze_2d", "landmarks,gaze_2d,gaze_3d"')
    parser.add_argument('--landmark_loss_weight', type=float, default=1.0, help='Weight for landmark loss component.')
    parser.add_argument('--gaze_2d_loss_weight', type=float, default=1.0, help='Weight for 2D gaze loss component.')
    parser.add_argument('--gaze_3d_loss_weight', type=float, default=1.0, help='Weight for 3D gaze loss component.')

    # MPII-specific arguments
    parser.add_argument('--train_participant_ids', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", 
                        help='Comma-separated list of participant IDs for training (MPII only)')
    parser.add_argument('--val_participant_ids', type=str, default="12,13,14", 
                        help='Comma-separated list of participant IDs for validation (MPII only)')

    # WFLW-specific arguments
    parser.add_argument('--train_annotation_file', type=str,
                        help='Training annotation file path (WFLW only)')
    parser.add_argument('--val_annotation_file', type=str,
                        help='Validation annotation file path (WFLW only)')
    parser.add_argument('--mpii_landmarks', action='store_true',
                        help='Extract MPII-style 6 landmarks from WFLW (WFLW only)')

    # 300W-specific arguments
    parser.add_argument('--subset', type=str, choices=['indoor', 'outdoor'], 
                        help='Subset to load for 300W dataset (indoor/outdoor, default: both)')
    parser.add_argument('--padding_ratio', type=float, default=0.3,
                        help='Padding ratio for landmark-based cropping in 300W dataset (default: 0.3)')
    parser.add_argument('--translation_ratio', type=float, default=0.2,
                        help='Random translation ratio for landmark-based cropping in 300W dataset (default: 0.2)')
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='Train/test split ratio for 300W dataset (default: 0.8 = 80% train, 20% test)')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for reproducible train/test splits in 300W dataset (default: 42)')

    # NME calculation indices
    parser.add_argument('--left_eye_idx', type=int, default=0,
                        help='Index of left eye landmark for NME calculation')
    parser.add_argument('--right_eye_idx', type=int, default=3,
                        help='Index of right eye landmark for NME calculation')

    return parser.parse_args()


# Training and Validation
def train_one_epoch(model, dataloader, landmark_criterion, gaze_2d_criterion, gaze_3d_criterion, optimizer, scaler, device, epoch, args, landmark_key):
    model.train()
    total_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_loss_accumulator = torch.tensor(0.0, device=device)
    gaze_2d_loss_accumulator = torch.tensor(0.0, device=device)
    gaze_3d_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_nme_accumulator = torch.tensor(0.0, device=device)
    gaze_2d_mse_accumulator = torch.tensor(0.0, device=device)
    gaze_2d_mae_accumulator = torch.tensor(0.0, device=device)
    gaze_3d_mse_accumulator = torch.tensor(0.0, device=device)
    gaze_3d_mae_accumulator = torch.tensor(0.0, device=device)
    start_time = time.time()

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        images = batch_data['image'].to(device, non_blocking=True)
        gt_landmarks_flat = batch_data[landmark_key].to(device, non_blocking=True).view(images.size(0), -1)
        gt_landmarks_reshaped = batch_data[landmark_key].to(device, non_blocking=True)
        
        gt_gaze_2d = None
        gt_gaze_3d = None
        if 'gaze_2d' in args.training_modes:
            gt_gaze_2d = batch_data['gaze_2d_angles'].to(device, non_blocking=True)
        if 'gaze_3d' in args.training_modes:
            gt_gaze_3d = batch_data['gaze_direction_cam_3d'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            pred_landmarks, pred_gaze_2d, pred_gaze_3d = model(images)

            # Calculate total loss based on training mode
            total_loss = torch.tensor(0.0, device=device)

            landmark_loss = torch.tensor(0.0, device=device)
            if 'landmarks' in args.training_modes:
                landmark_loss = landmark_criterion(pred_landmarks, gt_landmarks_flat)
                total_loss += args.landmark_loss_weight * landmark_loss

            gaze_2d_loss = torch.tensor(0.0, device=device)
            if 'gaze_2d' in args.training_modes:
                gaze_2d_loss = gaze_2d_criterion(pred_gaze_2d, gt_gaze_2d)
                total_loss += args.gaze_2d_loss_weight * gaze_2d_loss

            gaze_3d_loss = torch.tensor(0.0, device=device)
            if 'gaze_3d' in args.training_modes:
                gaze_3d_loss = gaze_3d_criterion(pred_gaze_3d, gt_gaze_3d)
                total_loss += args.gaze_3d_loss_weight * gaze_3d_loss

        if args.amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        total_loss_accumulator += total_loss.detach()
        landmark_loss_accumulator += landmark_loss.detach()
        gaze_2d_loss_accumulator += gaze_2d_loss.detach()
        gaze_3d_loss_accumulator += gaze_3d_loss.detach()

        # Calculate metrics
        if 'landmarks' in args.training_modes:
            pred_landmarks_reshaped = pred_landmarks.detach().view(pred_landmarks.size(0), -1, 2)
            current_nme = NME(pred_landmarks_reshaped, gt_landmarks_reshaped, left_eye_idx=0, right_eye_idx=3)
            landmark_nme_accumulator += current_nme.detach()
        else:
            landmark_nme_accumulator += torch.tensor(0.0, device=device)

        # 2D Gaze metrics
        current_gaze_2d_mse = torch.tensor(0.0, device=device)
        current_gaze_2d_mae = torch.tensor(0.0, device=device)
        if 'gaze_2d' in args.training_modes:
            current_gaze_2d_mse = torch.nn.functional.mse_loss(pred_gaze_2d.detach(), gt_gaze_2d)
            current_gaze_2d_mae = torch.nn.functional.l1_loss(pred_gaze_2d.detach(), gt_gaze_2d)
        gaze_2d_mse_accumulator += current_gaze_2d_mse.detach()
        gaze_2d_mae_accumulator += current_gaze_2d_mae.detach()
        
        # 3D Gaze metrics
        current_gaze_3d_mse = torch.tensor(0.0, device=device)
        current_gaze_3d_mae = torch.tensor(0.0, device=device)
        if 'gaze_3d' in args.training_modes:
            current_gaze_3d_mse = torch.nn.functional.mse_loss(pred_gaze_3d.detach(), gt_gaze_3d)
            current_gaze_3d_mae = torch.nn.functional.l1_loss(pred_gaze_3d.detach(), gt_gaze_3d)
        gaze_3d_mse_accumulator += current_gaze_3d_mse.detach()
        gaze_3d_mae_accumulator += current_gaze_3d_mae.detach()

        # Log training progress
        if (batch_idx + 1) % args.log_interval == 0 or batch_idx == len(dataloader) -1:
            avg_total_loss = total_loss_accumulator.item() / (batch_idx + 1)
            avg_landmark_loss = landmark_loss_accumulator.item() / (batch_idx + 1)
            avg_gaze_2d_loss = gaze_2d_loss_accumulator.item() / (batch_idx + 1)
            avg_gaze_3d_loss = gaze_3d_loss_accumulator.item() / (batch_idx + 1)
            avg_nme = landmark_nme_accumulator.item() / (batch_idx + 1)
            avg_gaze_2d_mse = gaze_2d_mse_accumulator.item() / (batch_idx + 1)
            avg_gaze_2d_mae = gaze_2d_mae_accumulator.item() / (batch_idx + 1)
            avg_gaze_3d_mse = gaze_3d_mse_accumulator.item() / (batch_idx + 1)
            avg_gaze_3d_mae = gaze_3d_mae_accumulator.item() / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
                  f'Total Loss: {total_loss.item():.4f} (Avg: {avg_total_loss:.4f}), '
                  f'Lmk Loss: {landmark_loss.item():.4f} (Avg: {avg_landmark_loss:.4f}), '
                  f'2D Gaze Loss: {gaze_2d_loss.item():.4f} (Avg: {avg_gaze_2d_loss:.4f}), '
                  f'3D Gaze Loss: {gaze_3d_loss.item():.4f} (Avg: {avg_gaze_3d_loss:.4f}), '
                  f'Lmk NME: {landmark_nme_accumulator.item() / (batch_idx + 1):.4f}, '
                  f'2D Gaze MSE: {current_gaze_2d_mse.item():.4f} (Avg: {avg_gaze_2d_mse:.4f}), '
                  f'2D Gaze MAE: {current_gaze_2d_mae.item():.4f} (Avg: {avg_gaze_2d_mae:.4f}), '
                  f'3D Gaze MSE: {current_gaze_3d_mse.item():.4f} (Avg: {avg_gaze_3d_mse:.4f}), '
                  f'3D Gaze MAE: {current_gaze_3d_mae.item():.4f} (Avg: {avg_gaze_3d_mae:.4f}), '
                  f'Time: {elapsed_time:.2f}s')
            start_time = time.time() # Reset timer for next log interval reporting
            
    avg_total_loss = (total_loss_accumulator / len(dataloader)).item()
    avg_landmark_loss = (landmark_loss_accumulator / len(dataloader)).item()
    avg_gaze_2d_loss = (gaze_2d_loss_accumulator / len(dataloader)).item()
    avg_gaze_3d_loss = (gaze_3d_loss_accumulator / len(dataloader)).item()
    avg_nme = (landmark_nme_accumulator / len(dataloader)).item()
    avg_gaze_2d_mse = (gaze_2d_mse_accumulator / len(dataloader)).item()
    avg_gaze_2d_mae = (gaze_2d_mae_accumulator / len(dataloader)).item()
    avg_gaze_3d_mse = (gaze_3d_mse_accumulator / len(dataloader)).item()
    avg_gaze_3d_mae = (gaze_3d_mae_accumulator / len(dataloader)).item()

    return avg_total_loss, avg_landmark_loss, avg_gaze_2d_loss, avg_gaze_3d_loss, avg_nme, avg_gaze_2d_mse, avg_gaze_2d_mae, avg_gaze_3d_mse, avg_gaze_3d_mae

def validate(model, dataloader, landmark_criterion, gaze_2d_criterion, gaze_3d_criterion, device, args, landmark_key):
    model.eval()
    total_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_loss_accumulator = torch.tensor(0.0, device=device)
    gaze_2d_loss_accumulator = torch.tensor(0.0, device=device)
    gaze_3d_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_nme_accumulator = torch.tensor(0.0, device=device)
    gaze_2d_mse_accumulator = torch.tensor(0.0, device=device)
    gaze_2d_mae_accumulator = torch.tensor(0.0, device=device)
    gaze_3d_mse_accumulator = torch.tensor(0.0, device=device)
    gaze_3d_mae_accumulator = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            images = batch_data['image'].to(device, non_blocking=True)
            gt_landmarks_flat = batch_data[landmark_key].to(device, non_blocking=True).view(images.size(0), -1)
            gt_landmarks_reshaped = batch_data[landmark_key].to(device, non_blocking=True)

            gt_gaze_2d = None
            gt_gaze_3d = None
            if 'gaze_2d' in args.training_modes:
                gt_gaze_2d = batch_data['gaze_2d_angles'].to(device, non_blocking=True)
            if 'gaze_3d' in args.training_modes:
                gt_gaze_3d = batch_data['gaze_direction_cam_3d'].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                pred_landmarks, pred_gaze_2d, pred_gaze_3d = model(images)
                                    
                # Calculate total loss based on training mode
                total_loss = torch.tensor(0.0, device=device)

                landmark_loss = torch.tensor(0.0, device=device)
                if 'landmarks' in args.training_modes:
                    landmark_loss = landmark_criterion(pred_landmarks, gt_landmarks_flat)
                    total_loss += args.landmark_loss_weight * landmark_loss

                gaze_2d_loss = torch.tensor(0.0, device=device)
                if 'gaze_2d' in args.training_modes:
                    gaze_2d_loss = gaze_2d_criterion(pred_gaze_2d, gt_gaze_2d)
                    total_loss += args.gaze_2d_loss_weight * gaze_2d_loss

                gaze_3d_loss = torch.tensor(0.0, device=device)
                if 'gaze_3d' in args.training_modes:
                    gaze_3d_loss = gaze_3d_criterion(pred_gaze_3d, gt_gaze_3d)
                    total_loss += args.gaze_3d_loss_weight * gaze_3d_loss
                
            total_loss_accumulator += total_loss.detach()
            landmark_loss_accumulator += landmark_loss.detach()
            gaze_2d_loss_accumulator += gaze_2d_loss.detach()
            gaze_3d_loss_accumulator += gaze_3d_loss.detach()

            # Calculate metrics
            if 'landmarks' in args.training_modes:
                pred_landmarks_reshaped = pred_landmarks.detach().view(pred_landmarks.size(0), -1, 2)
                current_nme = NME(pred_landmarks_reshaped, gt_landmarks_reshaped, left_eye_idx=0, right_eye_idx=3)
                landmark_nme_accumulator += current_nme.detach()
            else:
                landmark_nme_accumulator += torch.tensor(0.0, device=device)

            # 2D Gaze metrics
            if 'gaze_2d' in args.training_modes:
                current_gaze_2d_mse = torch.nn.functional.mse_loss(pred_gaze_2d.detach(), gt_gaze_2d)
                current_gaze_2d_mae = torch.nn.functional.l1_loss(pred_gaze_2d.detach(), gt_gaze_2d)
                gaze_2d_mse_accumulator += current_gaze_2d_mse.detach()
                gaze_2d_mae_accumulator += current_gaze_2d_mae.detach()

            # 3D Gaze metrics
            if 'gaze_3d' in args.training_modes:
                current_gaze_3d_mse = torch.nn.functional.mse_loss(pred_gaze_3d.detach(), gt_gaze_3d)
                current_gaze_3d_mae = torch.nn.functional.l1_loss(pred_gaze_3d.detach(), gt_gaze_3d)
                gaze_3d_mse_accumulator += current_gaze_3d_mse.detach()
                gaze_3d_mae_accumulator += current_gaze_3d_mae.detach()
            
    avg_total_loss = (total_loss_accumulator / len(dataloader)).item()
    avg_landmark_loss = (landmark_loss_accumulator / len(dataloader)).item()
    avg_gaze_2d_loss = (gaze_2d_loss_accumulator / len(dataloader)).item()
    avg_gaze_3d_loss = (gaze_3d_loss_accumulator / len(dataloader)).item()
    avg_nme = (landmark_nme_accumulator / len(dataloader)).item()
    avg_gaze_2d_mse = (gaze_2d_mse_accumulator / len(dataloader)).item()
    avg_gaze_2d_mae = (gaze_2d_mae_accumulator / len(dataloader)).item()
    avg_gaze_3d_mse = (gaze_3d_mse_accumulator / len(dataloader)).item()
    avg_gaze_3d_mae = (gaze_3d_mae_accumulator / len(dataloader)).item()
    
    print(f'Validation: Avg Total Loss: {avg_total_loss:.4f}, '
          f'Avg Lmk Loss: {avg_landmark_loss:.4f}, '
          f'Avg 2D Gaze Loss: {avg_gaze_2d_loss:.4f}, Avg 3D Gaze Loss: {avg_gaze_3d_loss:.4f}, '
          f'Avg NME: {avg_nme:.4f}, '
          f'Avg 2D Gaze MSE: {avg_gaze_2d_mse:.4f}, Avg 2D Gaze MAE: {avg_gaze_2d_mae:.4f}, '
          f'Avg 3D Gaze MSE: {avg_gaze_3d_mse:.4f}, Avg 3D Gaze MAE: {avg_gaze_3d_mae:.4f}\n')

    return avg_total_loss, avg_landmark_loss, avg_gaze_2d_loss, avg_gaze_3d_loss, avg_nme, avg_gaze_2d_mse, avg_gaze_2d_mae, avg_gaze_3d_mse, avg_gaze_3d_mae

# Main Function
def main():
    args = get_args()
    
    if args.dataset == 'wflw':
        if not args.annotation_file and not (args.train_annotation_file and args.val_annotation_file):
            print("Error: For WFLW dataset, either --annotation_file or both --train_annotation_file and --val_annotation_file must be provided")
            return
    
    if args.dataset == '300w':
        if not (0.0 < args.train_test_split < 1.0):
            print(f"Error: train_test_split must be between 0.0 and 1.0, got {args.train_test_split}")
            return
    
    # Parse comma-separated training modes
    args.training_modes = [mode.strip() for mode in args.training_mode.split(',')]
    
    # Validate training modes
    valid_modes = {'landmarks', 'gaze_2d', 'gaze_3d'}
    invalid_modes = set(args.training_modes) - valid_modes
    if invalid_modes:
        print(f"Error: Invalid training modes: {invalid_modes}. Valid modes are: {valid_modes}")
        return
    
    # Check for gaze training with non-MPII datasets
    if args.dataset != 'mpii' and any(mode in args.training_modes for mode in ['gaze_2d', 'gaze_3d']):
        print(f"Warning: Gaze training is only supported for the MPII dataset. Removing gaze modes.")
        args.training_modes = [mode for mode in args.training_modes if mode not in ['gaze_2d', 'gaze_3d']]
        if not args.training_modes:
            args.training_modes = ['landmarks']
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    setup_device(device)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Image Transforms
    if args.input_channels == 1:
        normalize = transforms.Normalize(mean=[0.449], std=[0.226]) # Grayscale normalization (adjust as needed)
        train_transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ]
        val_transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    else: # Default to 3 channels (RGB)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        train_transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            normalize,
        ]
        val_transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]

    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose(val_transform_list)

    # Dataset creation based on dataset type
    if args.dataset == 'mpii':
        try:
            train_ids = [int(p_id) for p_id in args.train_participant_ids.split(',')]
            val_ids = [int(p_id) for p_id in args.val_participant_ids.split(',')]
        except ValueError:
            print("Error: Participant IDs must be comma-separated integers.")
            return

        print(f"Training with MPII participant IDs: {train_ids}")
        print(f"Validating with MPII participant IDs: {val_ids}")

        train_dataset = MPIIFaceGazeDataset(
            dataset_path=args.data_dir, 
            participant_ids=train_ids, 
            transform=train_transform,
            is_train=True,
            affine_aug=args.affine_aug,
            flip_aug=args.flip_aug,
            use_cache=args.use_cache,
            label_smoothing=args.label_smoothing
        )
        val_dataset = MPIIFaceGazeDataset(
            dataset_path=args.data_dir, 
            participant_ids=val_ids, 
            transform=val_transform,
            is_train=False,
            use_cache=args.use_cache
        )
        landmark_key = 'facial_landmarks'
        
    elif args.dataset == 'wflw':
        if args.train_annotation_file and args.val_annotation_file:
            train_annotation = args.train_annotation_file
            val_annotation = args.val_annotation_file
        else:
            train_annotation = val_annotation = args.annotation_file
            
        print(f"Training with WFLW annotation: {train_annotation}")
        print(f"Validating with WFLW annotation: {val_annotation}")

        train_dataset = WFLWDataset(
            annotation_file=train_annotation,
            images_dir=args.data_dir,
            transform=train_transform,
            is_train=True,
            affine_aug=args.affine_aug,
            flip_aug=args.flip_aug,
            use_cache=args.use_cache,
            label_smoothing=args.label_smoothing,
            mpii_landmarks=args.mpii_landmarks
        )
        val_dataset = WFLWDataset(
            annotation_file=val_annotation,
            images_dir=args.data_dir,
            transform=val_transform,
            is_train=False,
            use_cache=args.use_cache,
            mpii_landmarks=args.mpii_landmarks
        )
        landmark_key = 'landmarks'
        
    else:  # 300w
        subset_str = f" ({args.subset} subset)" if args.subset else " (both subsets)"
        print(f"Training with 300W dataset from {args.data_dir}{subset_str}")
        print(f"Using padding_ratio={args.padding_ratio}, translation_ratio={args.translation_ratio}")
        print(f"Train/test split: {args.train_test_split:.2f}/{1-args.train_test_split:.2f} (seed: {args.split_seed})")

        train_dataset = Face300WDataset(
            root_dir=args.data_dir,
            subset=args.subset,
            transform=train_transform,
            is_train=True,
            affine_aug=args.affine_aug,
            flip_aug=args.flip_aug,
            use_cache=args.use_cache,
            label_smoothing=args.label_smoothing,
            mpii_landmarks=args.mpii_landmarks,
            padding_ratio=args.padding_ratio,
            translation_ratio=args.translation_ratio,
            train_test_split=args.train_test_split,
            split='train',
            split_seed=args.split_seed
        )
        val_dataset = Face300WDataset(
            root_dir=args.data_dir,
            subset=args.subset,
            transform=val_transform,
            is_train=False,
            use_cache=args.use_cache,
            mpii_landmarks=args.mpii_landmarks,
            padding_ratio=args.padding_ratio,
            translation_ratio=0.0,  # No translation during validation
            train_test_split=args.train_test_split,
            split='test',
            split_seed=args.split_seed
        )
        landmark_key = 'landmarks'

    if not train_dataset.samples:
        print(f"Error: No training samples found. Check data paths and configurations.")
        return
    if not val_dataset.samples:
        print(f"Error: No validation samples found. Check data paths and configurations.")
        return

    # Optimized DataLoader settings for reduced CPU overhead
    persistent_workers = args.persistent_workers and args.num_workers > 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        drop_last=True,
        prefetch_factor=args.prefetch_factor, 
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        prefetch_factor=args.prefetch_factor, 
        persistent_workers=persistent_workers
    )

    print(f"DataLoader settings: num_workers={args.num_workers}, prefetch_factor={args.prefetch_factor}, persistent_workers={persistent_workers}")

    # Model
    model = MHModel(
        num_landmarks=args.num_landmarks, 
        pretrained_backbone=not args.no_pretrained_backbone,
        in_channels=args.input_channels,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    print(f"Model: MHModel initialized with {args.num_landmarks} landmarks and {args.input_channels} input channel(s).")
    print(f"Backbone pretrained: {not args.no_pretrained_backbone}")
    print(f"Training Modes: {', '.join(args.training_modes).upper()} (Landmark Weight: {args.landmark_loss_weight}, 2D Gaze Weight: {args.gaze_2d_loss_weight}, 3D Gaze Weight: {args.gaze_3d_loss_weight})")

    # Loss and Optimizer
    landmark_criterion = SmoothWingLoss() # Smooth Wing Loss for landmark regression
    gaze_2d_criterion = nn.MSELoss() # MSE for 2D gaze regression
    gaze_3d_criterion = nn.MSELoss() # MSE for 3D gaze regression
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning Rate Scheduler - will be initialized considering warmup
    scheduler = None # Initialize later
    main_training_epochs = args.epochs - args.warmup_epochs

    if main_training_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=1.0, # Scheduler starts from the LR set in optimizer (will be args.lr at main phase start)
            end_factor=args.lr_final / args.lr if args.lr > 0 else 0.0, # Handle args.lr = 0
            total_iters=main_training_epochs
        )
        print(f"Main LR scheduler configured for {main_training_epochs} epochs, from {args.lr} to {args.lr_final}")
    else:
        print("No main training epochs after warmup, or main_training_epochs is zero. Scheduler not configured.")
    
    # AMP Scaler
    scaler = torch.amp.GradScaler(device=device, enabled=args.amp)
    if args.amp:
        print("Automatic Mixed Precision (AMP) enabled.")

    # Resume from Checkpoint
    start_epoch = 0
    history = {
        'train_total_loss': [], 'val_total_loss': [],
        'train_landmark_loss': [], 'val_landmark_loss': [],
        'train_gaze_2d_loss': [], 'val_gaze_2d_loss': [],
        'train_gaze_3d_loss': [], 'val_gaze_3d_loss': [],
        'train_landmark_nme': [], 'val_landmark_nme': [],
        'train_gaze_2d_mse': [], 'val_gaze_2d_mse': [],
        'train_gaze_2d_mae': [], 'val_gaze_2d_mae': [],
        'train_gaze_3d_mse': [], 'val_gaze_3d_mse': [],
        'train_gaze_3d_mae': [], 'val_gaze_3d_mae': [],
        'lr': []
    }
    history_filepath = os.path.join(args.checkpoint_dir, 'training_history.json')

    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            # Pass scheduler to load_checkpoint (it might be None if no main_training_epochs)
            print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            start_epoch = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, scaler if args.amp else None)
            print(f"Resumed from epoch {start_epoch}. Optimizer LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Load history if resuming
            loaded_history = load_history(history_filepath)
            if loaded_history:
                history = loaded_history
                print("Resumed training history.")
            else:
                print("No existing history file found or error loading it. Starting with fresh history.")
        else:
            print(f"Warning: Resume checkpoint not found at {args.resume_checkpoint}")

    # Training Loop
    best_val_loss = float('inf')

    print(f"Starting training for {args.epochs} total epochs. Warmup epochs: {args.warmup_epochs}.")

    for epoch in range(start_epoch, args.epochs):
        print(f"--- Overall Epoch {epoch+1}/{args.epochs} ---")
        
        is_warmup_epoch = epoch < args.warmup_epochs

        # Handle Backbone Freezing/Unfreezing
        if args.freeze_backbone_warmup and hasattr(model, 'backbone') and isinstance(model.backbone, nn.Module):
            if is_warmup_epoch:
                # Freeze backbone if it's the first warmup epoch or if it was previously unfrozen
                if not all(not p.requires_grad for p in model.backbone.parameters()):
                    print(f"Epoch {epoch+1}: Freezing backbone for warmup phase (epoch < {args.warmup_epochs}).")
                    for param in model.backbone.parameters():
                        param.requires_grad = False
            else: # Main training phase, i.e., epoch >= args.warmup_epochs
                # Unfreeze backbone if it was frozen
                if any(not p.requires_grad for p in model.backbone.parameters()):
                    print(f"Epoch {epoch+1}: Unfreezing backbone for main training phase (epoch >= {args.warmup_epochs}).")
                    for param in model.backbone.parameters():
                        param.requires_grad = True
        elif not args.freeze_backbone_warmup and hasattr(model, 'backbone') and isinstance(model.backbone, nn.Module):
            # Ensure backbone is trainable if freeze_backbone_warmup is false
            if any(not p.requires_grad for p in model.backbone.parameters()):
                print(f"Epoch {epoch+1}: Ensuring backbone is trainable (freeze_backbone_warmup=False).")
                for param in model.backbone.parameters():
                    param.requires_grad = True
        
        # Set Learning Rate for the current epoch
        if is_warmup_epoch:
            if args.warmup_epochs > 0: # Ensure warmup_lr is used only if there are warmup epochs
                for g in optimizer.param_groups:
                    g['lr'] = args.warmup_lr
                print(f"Warmup Epoch {epoch+1}/{args.warmup_epochs}. LR explicitly set to: {args.warmup_lr:.2e}")
        else: # Main training phase
            if epoch == args.warmup_epochs: # First main training epoch
                # Set optimizer's LR to args.lr so scheduler starts from the correct base
                for g in optimizer.param_groups:
                    g['lr'] = args.lr
                print(f"Main Training (Epoch {epoch+1-args.warmup_epochs}/{main_training_epochs}). LR reset to: {args.lr:.2e} for scheduler.")
        
        history['lr'].append(optimizer.param_groups[0]['lr'])  # Record LR for the epoch

        train_total_loss, train_lmk_loss, train_gaze_2d_loss, train_gaze_3d_loss, train_nme, train_gaze_2d_mse, train_gaze_2d_mae, train_gaze_3d_mse, train_gaze_3d_mae = train_one_epoch(model, train_loader, landmark_criterion, gaze_2d_criterion, gaze_3d_criterion, optimizer, scaler, device, epoch, args, landmark_key)
        val_total_loss, val_lmk_loss, val_gaze_2d_loss, val_gaze_3d_loss, val_nme, val_gaze_2d_mse, val_gaze_2d_mae, val_gaze_3d_mse, val_gaze_3d_mae = validate(model, val_loader, landmark_criterion, gaze_2d_criterion, gaze_3d_criterion, device, args, landmark_key)
        
        # Step the scheduler if in main training phase (and scheduler exists)
        if not is_warmup_epoch and scheduler is not None:
            scheduler.step()
            # Log the LR after scheduler step for clarity
            print(f"Main Training (Epoch {epoch + 1 - args.warmup_epochs}/{main_training_epochs}). Scheduler stepped. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        elif not is_warmup_epoch: # Main training but no scheduler (e.g. main_training_epochs <=0)
             print(f"Main Training (Epoch {epoch + 1 - args.warmup_epochs}/{main_training_epochs}). No scheduler. LR: {optimizer.param_groups[0]['lr']:.2e}")

        history['train_total_loss'].append(train_total_loss)
        history['val_total_loss'].append(val_total_loss)
        history['train_landmark_loss'].append(train_lmk_loss)
        history['val_landmark_loss'].append(val_lmk_loss)
        history['train_gaze_2d_loss'].append(train_gaze_2d_loss)
        history['val_gaze_2d_loss'].append(val_gaze_2d_loss)
        history['train_gaze_3d_loss'].append(train_gaze_3d_loss)
        history['val_gaze_3d_loss'].append(val_gaze_3d_loss)
        history['train_landmark_nme'].append(train_nme)
        history['val_landmark_nme'].append(val_nme)
        history['train_gaze_2d_mse'].append(train_gaze_2d_mse)
        history['val_gaze_2d_mse'].append(val_gaze_2d_mse)
        history['train_gaze_2d_mae'].append(train_gaze_2d_mae)
        history['val_gaze_2d_mae'].append(val_gaze_2d_mae)
        history['train_gaze_3d_mse'].append(train_gaze_3d_mse)
        history['val_gaze_3d_mse'].append(val_gaze_3d_mse)
        history['train_gaze_3d_mae'].append(train_gaze_3d_mae)
        history['val_gaze_3d_mae'].append(val_gaze_3d_mae)
        
        # Save checkpoint based on frequency or if it's the last epoch
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, checkpoint_path, args.num_landmarks, args.dataset)
            save_history(history, history_filepath)
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, best_checkpoint_path, args.num_landmarks, args.dataset)
            save_history(history, history_filepath)
            print(f"New best validation loss: {best_val_loss:.6f}. Saved best model to {best_checkpoint_path}")
        
        print(f"Epoch {epoch+1} Summary: Train Total Loss: {train_total_loss:.4f}, Val Total Loss: {val_total_loss:.4f}, Train NME: {train_nme:.4f}, Val NME: {val_nme:.4f}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Find the best model at: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")

    # Save final history at the end of training
    save_history(history, history_filepath)
    print(f"Final training history saved to {history_filepath}")

    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    print(f"Training history plot saved to {plot_path}")


if __name__ == '__main__':
    main()
