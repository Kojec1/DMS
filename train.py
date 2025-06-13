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
from data.dataset import MPIIFaceGazeDataset
from utils.visualization import plot_training_history
from utils.misc import set_seed, setup_device
from utils.checkpoint import save_checkpoint, load_checkpoint, save_history, load_history


# Configuration
def get_args():
    parser = argparse.ArgumentParser(description='Multi-Head Model Training for Facial Landmarks and 2D Gaze Estimation')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for MPIIFaceGaze dataset (containing p00, p01, etc.)')
    parser.add_argument('--num_landmarks', type=int, default=6, help='Number of facial landmarks (MPIIFaceGaze has 6)')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (height and width) for ConvNeXt, e.g., 224')
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
    parser.add_argument('--training_mode', type=str, default='both', choices=['landmarks', 'gaze', 'both'], help='Training mode: landmarks, gaze, or both.')
    parser.add_argument('--landmark_loss_weight', type=float, default=1.0, help='Weight for landmark loss component.')
    parser.add_argument('--gaze_loss_weight', type=float, default=1.0, help='Weight for gaze loss component.')


    # Participant IDs for splitting data
    parser.add_argument('--train_participant_ids', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", 
                        help='Comma-separated list of participant IDs for training (e.g., from 0 to 14 for MPIIFaceGaze)')
    parser.add_argument('--val_participant_ids', type=str, default="12,13,14", 
                        help='Comma-separated list of participant IDs for validation')

    return parser.parse_args()


# Training and Validation
def train_one_epoch(model, dataloader, landmark_criterion, gaze_criterion, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_loss_accumulator = torch.tensor(0.0, device=device)
    gaze_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_nme_accumulator = torch.tensor(0.0, device=device)
    gaze_mse_accumulator = torch.tensor(0.0, device=device)
    gaze_mae_accumulator = torch.tensor(0.0, device=device)
    start_time = time.time()

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        images = batch_data['image'].to(device, non_blocking=True)
        gt_landmarks_flat = batch_data['facial_landmarks'].to(device, non_blocking=True).view(images.size(0), -1)
        gt_landmarks_reshaped = batch_data['facial_landmarks'].to(device, non_blocking=True)
        gt_gaze = batch_data['gaze_2d_angles'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            pred_landmarks, pred_gaze = model(images)
            
            # Calculate individual losses
            landmark_loss = landmark_criterion(pred_landmarks, gt_landmarks_flat)
            gaze_loss = gaze_criterion(pred_gaze, gt_gaze)
            
            # Calculate total loss based on training mode
            total_loss = torch.tensor(0.0, device=device)
            if args.training_mode in ['landmarks', 'both']:
                total_loss += args.landmark_loss_weight * landmark_loss
            if args.training_mode in ['gaze', 'both']:
                total_loss += args.gaze_loss_weight * gaze_loss

        if args.amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        total_loss_accumulator += total_loss.detach()
        landmark_loss_accumulator += landmark_loss.detach()
        gaze_loss_accumulator += gaze_loss.detach()

        # Calculate metrics
        pred_landmarks_reshaped = pred_landmarks.detach().view(pred_landmarks.size(0), -1, 2)
        
        # Landmark NME - outer eye corner landmarks have indices 0 and 3
        current_nme = NME(pred_landmarks_reshaped, gt_landmarks_reshaped, left_eye_idx=0, right_eye_idx=3)
        landmark_nme_accumulator += current_nme.detach()

        # Gaze MSE
        current_gaze_mse = torch.nn.functional.mse_loss(pred_gaze.detach(), gt_gaze)
        gaze_mse_accumulator += current_gaze_mse.detach()
        
        # Gaze MAE
        current_gaze_mae = torch.nn.functional.l1_loss(pred_gaze.detach(), gt_gaze)
        gaze_mae_accumulator += current_gaze_mae.detach()

        # Log training progress
        if (batch_idx + 1) % args.log_interval == 0 or batch_idx == len(dataloader) -1:
            avg_total_loss = total_loss_accumulator.item() / (batch_idx + 1)
            avg_landmark_loss = landmark_loss_accumulator.item() / (batch_idx + 1)
            avg_gaze_loss = gaze_loss_accumulator.item() / (batch_idx + 1)
            avg_nme = landmark_nme_accumulator.item() / (batch_idx + 1)
            avg_gaze_mse = gaze_mse_accumulator.item() / (batch_idx + 1)
            avg_gaze_mae = gaze_mae_accumulator.item() / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
                  f'Total Loss: {total_loss.item():.4f} (Avg: {avg_total_loss:.4f}), '
                  f'Lmk Loss: {landmark_loss.item():.4f} (Avg: {avg_landmark_loss:.4f}), '
                  f'Gaze Loss: {gaze_loss.item():.4f} (Avg: {avg_gaze_loss:.4f}), '
                  f'Lmk NME: {current_nme.item():.4f} (Avg: {avg_nme:.4f}), '
                  f'Gaze MSE: {current_gaze_mse.item():.4f} (Avg: {avg_gaze_mse:.4f}), '
                  f'Gaze MAE: {current_gaze_mae.item():.4f} (Avg: {avg_gaze_mae:.4f}), '
                  f'Time: {elapsed_time:.2f}s')
            start_time = time.time() # Reset timer for next log interval reporting
            
    avg_total_loss = (total_loss_accumulator / len(dataloader)).item()
    avg_landmark_loss = (landmark_loss_accumulator / len(dataloader)).item()
    avg_gaze_loss = (gaze_loss_accumulator / len(dataloader)).item()
    avg_nme = (landmark_nme_accumulator / len(dataloader)).item()
    avg_gaze_mse = (gaze_mse_accumulator / len(dataloader)).item()
    avg_gaze_mae = (gaze_mae_accumulator / len(dataloader)).item()

    return avg_total_loss, avg_landmark_loss, avg_gaze_loss, avg_nme, avg_gaze_mse, avg_gaze_mae

def validate(model, dataloader, landmark_criterion, gaze_criterion, device, args):
    model.eval()
    total_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_loss_accumulator = torch.tensor(0.0, device=device)
    gaze_loss_accumulator = torch.tensor(0.0, device=device)
    landmark_nme_accumulator = torch.tensor(0.0, device=device)
    gaze_mse_accumulator = torch.tensor(0.0, device=device)
    gaze_mae_accumulator = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            images = batch_data['image'].to(device, non_blocking=True)
            gt_landmarks_flat = batch_data['facial_landmarks'].to(device, non_blocking=True).view(images.size(0), -1)
            gt_landmarks_reshaped = batch_data['facial_landmarks'].to(device, non_blocking=True)
            gt_gaze = batch_data['gaze_2d_angles'].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                pred_landmarks, pred_gaze = model(images)
                
                # Calculate individual losses
                landmark_loss = landmark_criterion(pred_landmarks, gt_landmarks_flat)
                gaze_loss = gaze_criterion(pred_gaze, gt_gaze)
            
                # Calculate total loss based on training mode
                total_loss = torch.tensor(0.0, device=device)
                if args.training_mode in ['landmarks', 'both']:
                    total_loss += args.landmark_loss_weight * landmark_loss
                if args.training_mode in ['gaze', 'both']:
                    total_loss += args.gaze_loss_weight * gaze_loss
                
            total_loss_accumulator += total_loss.detach()
            landmark_loss_accumulator += landmark_loss.detach()
            gaze_loss_accumulator += gaze_loss.detach()

            # Calculate metrics
            pred_landmarks_reshaped = pred_landmarks.detach().view(pred_landmarks.size(0), -1, 2)
            
            # Landmark NME - outer eye corner landmarks have indices 0 and 3
            current_nme = NME(pred_landmarks_reshaped, gt_landmarks_reshaped, left_eye_idx=0, right_eye_idx=3)
            landmark_nme_accumulator += current_nme.detach()

            # Gaze MSE
            current_gaze_mse = torch.nn.functional.mse_loss(pred_gaze.detach(), gt_gaze)
            gaze_mse_accumulator += current_gaze_mse.detach()
            
            # Gaze MAE
            current_gaze_mae = torch.nn.functional.l1_loss(pred_gaze.detach(), gt_gaze)
            gaze_mae_accumulator += current_gaze_mae.detach()
            
    avg_total_loss = (total_loss_accumulator / len(dataloader)).item()
    avg_landmark_loss = (landmark_loss_accumulator / len(dataloader)).item()
    avg_gaze_loss = (gaze_loss_accumulator / len(dataloader)).item()
    avg_nme = (landmark_nme_accumulator / len(dataloader)).item()
    avg_gaze_mse = (gaze_mse_accumulator / len(dataloader)).item()
    avg_gaze_mae = (gaze_mae_accumulator / len(dataloader)).item()
    
    print(f'Validation: Avg Total Loss: {avg_total_loss:.4f}, '
          f'Avg Lmk Loss: {avg_landmark_loss:.4f}, Avg Gaze Loss: {avg_gaze_loss:.4f}, '
          f'Avg NME: {avg_nme:.4f}, Avg Gaze MSE: {avg_gaze_mse:.4f}, Avg Gaze MAE: {avg_gaze_mae:.4f}\n')

    return avg_total_loss, avg_landmark_loss, avg_gaze_loss, avg_nme, avg_gaze_mse, avg_gaze_mae

# Main Function
def main():
    args = get_args()
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

    # DataLoaders
    try:
        train_ids = [int(p_id) for p_id in args.train_participant_ids.split(',')]
        val_ids = [int(p_id) for p_id in args.val_participant_ids.split(',')]
    except ValueError:
        print("Error: Participant IDs must be comma-separated integers.")
        return

    print(f"Training with participant IDs: {train_ids}")
    print(f"Validating with participant IDs: {val_ids}")

    train_dataset = MPIIFaceGazeDataset(dataset_path=args.data_dir, 
                                        participant_ids=train_ids, 
                                        transform=train_transform,
                                        is_train=True,
                                        affine_aug=args.affine_aug,
                                        flip_aug=args.flip_aug,
                                        use_cache=args.use_cache,
                                        label_smoothing=args.label_smoothing)
    val_dataset = MPIIFaceGazeDataset(dataset_path=args.data_dir, 
                                      participant_ids=val_ids, 
                                      transform=val_transform,
                                      is_train=False,
                                      use_cache=args.use_cache)
    
    if not train_dataset.samples:
        print(f"Error: No training samples found. Check data_dir ('{args.data_dir}') and train_participant_ids ('{args.train_participant_ids}').")
        return
    if not val_dataset.samples:
        print(f"Error: No validation samples found. Check data_dir ('{args.data_dir}') and val_participant_ids ('{args.val_participant_ids}').")
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
    print(f"Training Mode: {args.training_mode.upper()} (Landmark Weight: {args.landmark_loss_weight}, Gaze Weight: {args.gaze_loss_weight})")

    # Loss and Optimizer
    landmark_criterion = SmoothWingLoss() # Smooth Wing Loss for landmark regression
    gaze_criterion = nn.MSELoss() # MSE for gaze regression
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
        'train_gaze_loss': [], 'val_gaze_loss': [],
        'train_landmark_nme': [], 'val_landmark_nme': [],
        'train_gaze_mse': [], 'val_gaze_mse': [],
        'train_gaze_mae': [], 'val_gaze_mae': [],
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

        train_total_loss, train_lmk_loss, train_gaze_loss, train_nme, train_gaze_mse, train_gaze_mae = train_one_epoch(model, train_loader, landmark_criterion, gaze_criterion, optimizer, scaler, device, epoch, args)
        val_total_loss, val_lmk_loss, val_gaze_loss, val_nme, val_gaze_mse, val_gaze_mae = validate(model, val_loader, landmark_criterion, gaze_criterion, device, args)
        
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
        history['train_gaze_loss'].append(train_gaze_loss)
        history['val_gaze_loss'].append(val_gaze_loss)
        history['train_landmark_nme'].append(train_nme)
        history['val_landmark_nme'].append(val_nme)
        history['train_gaze_mse'].append(train_gaze_mse)
        history['val_gaze_mse'].append(val_gaze_mse)
        history['train_gaze_mae'].append(train_gaze_mae)
        history['val_gaze_mae'].append(val_gaze_mae)
        
        # Save checkpoint based on frequency or if it's the last epoch
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, checkpoint_path)
            save_history(history, history_filepath)
        
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, best_checkpoint_path)
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
