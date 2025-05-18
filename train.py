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
    # num_landmarks is fixed by the dataset (6 landmarks, 2 coords each)
    parser.add_argument('--num_landmarks', type=int, default=6, help='Number of facial landmarks (MPIIFaceGaze has 6)')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (height and width) for ConvNeXt, e.g., 224')
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
                  f'Loss: {current_loss:.4f} (Avg Epoch: {avg_epoch_loss:.4f}), Time: {elapsed_time:.2f}s')
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
    print(f'Validation: Avg Loss: {avg_loss:.4f}\n')
    return avg_loss

# Main Function
def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Image Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize,
    ])

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
        pretrained_backbone=not args.no_pretrained_backbone
    ).to(device)
    print(f"Model: FacialLandmarkEstimator initialized with {args.num_landmarks} landmarks.")
    print(f"Backbone pretrained: {not args.no_pretrained_backbone}")

    # Attempt to compile the model with torch.compile() if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            print("Attempting to compile the model with torch.compile()...")
            model = torch.compile(model)
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Failed to compile model: {e}. Proceeding without compilation.")
    else:
        print("torch.compile not available. Proceeding without compilation (requires PyTorch 2.0+ for this feature).")

    # Loss and Optimizer
    criterion = nn.MSELoss() # Mean Squared Error for landmark regression
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=args.lr_final / args.lr,
        total_iters=args.epochs
    )
    
    # AMP Scaler
    scaler = torch.amp.GradScaler(device=device, enabled=args.amp)
    if args.amp:
        print("Automatic Mixed Precision (AMP) enabled.")

    # Resume from Checkpoint
    start_epoch = 0
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            # Pass scheduler to load_checkpoint
            start_epoch = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler, scaler if args.amp else None)
        else:
            print(f"Warning: Resume checkpoint not found at {args.resume_checkpoint}")

    # Training Loop
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args)
        val_loss = validate(model, val_loader, criterion, device, args)
        
        scheduler.step() # Step the scheduler after each epoch

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
            print(f"New best validation loss: {best_val_loss:.4f}. Saved best model to {best_checkpoint_path}")
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Find the best model at: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")

    # Plot training history
    plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
    plot_training_history(train_loss_history, val_loss_history, plot_path)
    print(f"Training history plot saved to {plot_path}")

if __name__ == '__main__':
    main()
