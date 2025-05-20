import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
import time
from tqdm import tqdm

from nn.modules.facial_landmark_estimator import FacialLandmarkEstimator
from data.dataset import MPIIFaceGazeDataset
from utils.visualization import plot_training_history
from utils.misc import set_seed
from utils.checkpoint import save_checkpoint, load_checkpoint


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
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--no_pretrained_backbone', action='store_true', help='Do not use pretrained backbone weights')
    
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
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        images = batch_data['image'].to(device)
        # MPIIFaceGazeDataset returns landmarks as (N, 6, 2).
        # Model outputs (N, 12). Flatten targets.
        targets = batch_data['facial_landmarks'].to(device).view(images.size(0), -1)
        
        optimizer.zero_grad()
        
        if args.amp:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % args.log_interval == 0 or batch_idx == len(dataloader) -1:
            current_loss = loss.item()
            avg_epoch_loss = total_loss / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(dataloader)}], ' \
                  f'Loss: {current_loss:.6f} (Avg Epoch: {avg_epoch_loss:.6f}), Time: {elapsed_time:.2f}s')
            start_time = time.time() # Reset timer for next log interval reporting
            
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            images = batch_data['image'].to(device)
            targets = batch_data['facial_landmarks'].to(device).view(images.size(0), -1)
            
            # Use autocast consistently with enabled flag
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
                
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    print(f'Validation: Avg Loss: {avg_loss:.6f}\n')
    return avg_loss

# Main Function
def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Image Transforms
    if args.input_channels == 1:
        normalize = transforms.Normalize(mean=[0.449], std=[0.226]) # Grayscale normalization (adjust as needed)
        train_transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Saturation/Hue removed for grayscale
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
                                        is_train=True)
    val_dataset = MPIIFaceGazeDataset(dataset_path=args.data_dir, 
                                      participant_ids=val_ids, 
                                      transform=val_transform,
                                      is_train=False)
    
    if not train_dataset.samples:
        print(f"Error: No training samples found. Check data_dir ('{args.data_dir}') and train_participant_ids ('{args.train_participant_ids}').")
        return
    if not val_dataset.samples:
        print(f"Error: No validation samples found. Check data_dir ('{args.data_dir}') and val_participant_ids ('{args.val_participant_ids}').")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    model = FacialLandmarkEstimator(
        num_landmarks=args.num_landmarks, 
        pretrained_backbone=not args.no_pretrained_backbone,
        in_channels=args.input_channels  # Pass in_channels to the model
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
    criterion = nn.MSELoss() # Mean Squared Error for landmark regression
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
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            # Pass scheduler to load_checkpoint (it might be None if no main_training_epochs)
            print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
            start_epoch = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, scaler if args.amp else None)
            print(f"Resumed from epoch {start_epoch}. Optimizer LR: {optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"Warning: Resume checkpoint not found at {args.resume_checkpoint}")

    # Training Loop
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    lr_history = []
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
        lr_history.append(optimizer.param_groups[0]['lr']) # Record LR for the epoch

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        val_loss = validate(model, val_loader, criterion, device, args)
        
        # Step the scheduler if in main training phase (and scheduler exists)
        if not is_warmup_epoch and scheduler is not None:
            scheduler.step()
            # Log the LR after scheduler step for clarity
            print(f"Main Training (Epoch {epoch + 1 - args.warmup_epochs}/{main_training_epochs}). Scheduler stepped. Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        elif not is_warmup_epoch: # Main training but no scheduler (e.g. main_training_epochs <=0)
             print(f"Main Training (Epoch {epoch + 1 - args.warmup_epochs}/{main_training_epochs}). No scheduler. LR: {optimizer.param_groups[0]['lr']:.2e}")


        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # Save checkpoint based on frequency or if it's the last epoch
        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            # Pass scheduler to save_checkpoint
            save_checkpoint(epoch, model, optimizer, scheduler, scaler if args.amp else None, best_checkpoint_path)
            print(f"New best validation loss: {best_val_loss:.6f}. Saved best model to {best_checkpoint_path}")
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Find the best model at: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")

    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(train_loss_history, val_loss_history, lr_history, plot_path)
    print(f"Training history plot saved to {plot_path}")

if __name__ == '__main__':
    main()
