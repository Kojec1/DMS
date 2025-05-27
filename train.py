import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
import time
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

from nn.modules.facial_landmark_estimator import FacialLandmarkEstimator
from nn.loss import SmoothWingLoss
from nn.metrics import NME
from data.dataset import MPIIFaceGazeDataset
from utils.visualization import plot_training_history
from utils.misc import set_seed, setup_device
from utils.checkpoint import save_checkpoint, load_checkpoint, save_history, load_history


# Configuration
def get_args():
    parser = argparse.ArgumentParser(description='Facial Landmark Estimation Training using MPIIFaceGazeDataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for MPIIFaceGaze dataset (containing p00, p01, etc.)')
    parser.add_argument('--num_landmarks', type=int, default=6, help='Number of facial landmarks (MPIIFaceGaze has 6)')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (height and width) for ConvNeXt, e.g., 224')
    parser.add_argument('--input_channels', type=int, default=3, choices=[1, 3], help='Number of input image channels (1 for grayscale, 3 for RGB)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_final', type=float, default=1e-6, help='Final learning rate for linear scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_fpe', help='Directory to save checkpoints')
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

    # Participant IDs for splitting data
    parser.add_argument('--train_participant_ids', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11", 
                        help='Comma-separated list of participant IDs for training (e.g., from 0 to 14 for MPIIFaceGaze)')
    parser.add_argument('--val_participant_ids', type=str, default="12,13,14", 
                        help='Comma-separated list of participant IDs for validation')

    return parser.parse_args()


# Training and Validation
def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    loss_accumulator = torch.tensor(0.0, device=device)
    nme_accumulator = torch.tensor(0.0, device=device)
    mse_accumulator = torch.tensor(0.0, device=device)
    start_time = time.time()

    # --- PyTorch Profiler Integration ---
    # Profile a few batches (e.g., first 5) in the first epoch for detailed analysis
    # To avoid overhead, don't profile every epoch or every batch unless specifically needed for debugging.
    profile_batches = 5 
    if epoch == 0: # Profile only during the first epoch, for example
        profiler_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            record_shapes=True, # Records input shapes
            profile_memory=True, # Tracks memory usage
            with_stack=True # Records call stacks, might add overhead
        )
    else:
        # Create a dummy context manager if not profiling to avoid changing loop structure
        class DummyContextManager:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc_value, traceback): pass
        profiler_context = DummyContextManager()
    # --- End Profiler Integration ---

    with profiler_context as prof: # Use the selected profiler context
        for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
            if epoch == 0 and batch_idx >= profile_batches and prof is not None: # Stop profiling after 'profile_batches' in the first epoch
                 if hasattr(prof, 'stop'): # Check if it's the actual profiler
                    break # Exit the loop early for this profiling run to get results faster


            with record_function("data_loading_and_preprocessing"): # Custom label for this block
                images = batch_data['image'].to(device, non_blocking=True)
                gt_landmarks_flat = batch_data['facial_landmarks'].to(device, non_blocking=True).view(images.size(0), -1)
                gt_landmarks_reshaped = batch_data['facial_landmarks'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if args.amp:
                with record_function("forward_backward_amp"): # Custom label
                    with torch.amp.autocast(device_type='cuda'):
                        outputs_flat = model(images)
                        loss = criterion(outputs_flat, gt_landmarks_flat)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                with record_function("forward_backward_no_amp"): # Custom label
                    outputs_flat = model(images)
                    loss = criterion(outputs_flat, gt_landmarks_flat)
                    loss.backward()
                    optimizer.step()

            loss_accumulator += loss.detach()

            # Calculate NME and MSE
            outputs_reshaped = outputs_flat.detach().view(outputs_flat.size(0), -1, 2)
            
            # NME - outer eye corner landmarks have indices 0 and 3
            current_nme = NME(outputs_reshaped, gt_landmarks_reshaped, left_eye_idx=0, right_eye_idx=3)
            nme_accumulator += current_nme.detach()

            # MSE
            current_mse = torch.nn.functional.mse_loss(outputs_flat.detach(), gt_landmarks_flat)
            mse_accumulator += current_mse.detach()
                        
            # Log training progress
            if (batch_idx + 1) % args.log_interval == 0 or batch_idx == len(dataloader) -1:
                current_loss = loss.item()
                avg_epoch_loss = loss_accumulator.item() / (batch_idx + 1)
                avg_epoch_nme = nme_accumulator.item() / (batch_idx + 1)
                avg_epoch_mse = mse_accumulator.item() / (batch_idx + 1)
                elapsed_time = time.time() - start_time
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
                    f'Loss: {current_loss:.6f} (Avg Epoch: {avg_epoch_loss:.6f}), '
                    f'NME: {current_nme.item():.6f} (Avg Epoch: {avg_epoch_nme:.6f}), '
                    f'MSE: {current_mse.item():.6f} (Avg Epoch: {avg_epoch_mse:.6f}), Time: {elapsed_time:.2f}s')
                start_time = time.time() # Reset timer for next log interval reporting
        
                    
    avg_loss = (loss_accumulator / len(dataloader)).item()
    avg_nme = (nme_accumulator / len(dataloader)).item()
    avg_mse = (mse_accumulator / len(dataloader)).item()

    if epoch == 0 and prof is not None and hasattr(prof, 'key_averages'): # Check if it's the actual profiler
        print("--- Profiler Results (First Epoch, First Few Batches) ---")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        print("--- End Profiler Results ---")
            
    return avg_loss, avg_nme, avg_mse

def validate(model, dataloader, criterion, device, args):
    model.eval()
    loss_accumulator = torch.tensor(0.0, device=device)
    nme_accumulator = torch.tensor(0.0, device=device)
    mse_accumulator = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            images = batch_data['image'].to(device, non_blocking=True)
            gt_landmarks_flat = batch_data['facial_landmarks'].to(device, non_blocking=True).view(images.size(0), -1)
            gt_landmarks_reshaped = batch_data['facial_landmarks'].to(device, non_blocking=True)
            
            # Use autocast consistently with enabled flag
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                outputs_flat = model(images)
                loss = criterion(outputs_flat, gt_landmarks_flat)
                
            loss_accumulator += loss.detach()

            # Calculate NME and MSE
            outputs_reshaped = outputs_flat.detach().view(outputs_flat.size(0), -1, 2)
            
            # NME - outer eye corner landmarks have indices 0 and 3
            current_nme = NME(outputs_reshaped, gt_landmarks_reshaped, left_eye_idx=0, right_eye_idx=3)
            nme_accumulator += current_nme.detach()

            # MSE
            current_mse = torch.nn.functional.mse_loss(outputs_flat.detach(), gt_landmarks_flat)
            mse_accumulator += current_mse.detach()
            
    avg_loss = (loss_accumulator / len(dataloader)).item()
    avg_nme = (nme_accumulator / len(dataloader)).item()
    avg_mse = (mse_accumulator / len(dataloader)).item()
    print(f'Validation: Avg Loss: {avg_loss:.6f}, Avg NME: {avg_nme:.6f}, Avg MSE: {avg_mse:.6f}\n')

    return avg_loss, avg_nme, avg_mse

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
    model = FacialLandmarkEstimator(
        num_landmarks=args.num_landmarks, 
        pretrained_backbone=not args.no_pretrained_backbone,
        in_channels=args.input_channels,
        dropout_rate=args.dropout_rate
    ).to(device)
    print(f"Model: FacialLandmarkEstimator initialized with {args.num_landmarks} landmarks and {args.input_channels} input channel(s).")
    print(f"Backbone pretrained: {not args.no_pretrained_backbone}")

    # Attempt to compile the model with torch.compile
    # if hasattr(torch, 'compile'):
    #     try:
    #         print("Attempting to compile the model with torch.compile()...")
    #         model = torch.compile(model)
    #         print("Model compiled successfully.")
    #     except Exception as e:
    #         print(f"Failed to compile model: {e}. Proceeding without compilation.")
    # else:
    #     print("torch.compile not available. Proceeding without compilation (requires PyTorch 2.0+ for this feature).")

    # Loss and Optimizer
    criterion = SmoothWingLoss() # Smooth Wing Loss for landmark regression
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
        'train_loss': [],
        'val_loss': [],
        'train_nme': [],
        'val_nme': [],
        'train_mse': [],
        'val_mse': [],
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

        train_loss, train_nme, train_mse = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        
        # If train_one_epoch exited early due to profiling, we might not want to proceed further for this run
        if epoch == 0 and len(train_loader) > 5 and train_loss == 0 : # Heuristic: if profiling broke early and loss is 0
            print("Exiting after profiling run.")
            break

        val_loss, val_nme, val_mse = validate(model, val_loader, criterion, device, args)
        
        # Step the scheduler if in main training phase (and scheduler exists)
        if not is_warmup_epoch and scheduler is not None:
            scheduler.step()
            # Log the LR after scheduler step for clarity
            print(f"Main Training (Epoch {epoch + 1 - args.warmup_epochs}/{main_training_epochs}). Scheduler stepped. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        elif not is_warmup_epoch: # Main training but no scheduler (e.g. main_training_epochs <=0)
             print(f"Main Training (Epoch {epoch + 1 - args.warmup_epochs}/{main_training_epochs}). No scheduler. LR: {optimizer.param_groups[0]['lr']:.2e}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_nme'].append(train_nme)
        history['val_nme'].append(val_nme)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Save checkpoint based on frequency or if it's the last epoch
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, checkpoint_path)
            save_history(history, history_filepath)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, best_checkpoint_path)
            save_history(history, history_filepath)
            print(f"New best validation loss: {best_val_loss:.6f}. Saved best model to {best_checkpoint_path}")
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train NME: {train_nme:.6f}, Val NME: {val_nme:.6f}, Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Find the best model at: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")

    # Save final history at the end of training
    save_history(history, history_filepath)
    print(f"Final training history saved to {history_filepath}")

    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(history['train_loss'], history['val_loss'], 
                          history['train_nme'], history['val_nme'],
                          history['train_mse'], history['val_mse'],
                          history['lr'], plot_path)
    print(f"Training history plot saved to {plot_path}")


if __name__ == '__main__':
    main()
