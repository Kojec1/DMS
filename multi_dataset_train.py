import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import os
import argparse
import time
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from nn.modules.facial_landmark_estimator import MHFacialLandmarkEstimator, FacialLandmarkEstimator
from nn.loss import MultiTaskLoss
from nn.metrics import NME
from data.dataset import MPIIFaceGazeDataset, WFLWDataset, Face300WDataset, MultiDatasetWrapper
from utils.visualization import plot_multi_task_training_history, create_metrics_summary
from utils.misc import set_seed, setup_device
from utils.checkpoint import save_checkpoint, save_history, load_history


def get_args():
    parser = argparse.ArgumentParser(description='Multi-Task Facial Landmark Estimation Training')
    
    # Dataset paths
    parser.add_argument('--mpii_data_dir', type=str, help='MPII dataset directory')
    parser.add_argument('--wflw_data_dir', type=str, help='WFLW images directory')
    parser.add_argument('--wflw_train_annotation', type=str, help='WFLW training annotation file')
    parser.add_argument('--wflw_val_annotation', type=str, help='WFLW validation annotation file')
    parser.add_argument('--face300w_data_dir', type=str, help='300W dataset directory')
    
    # Training parameters
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3], help='Number of input channels')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    
    # Multi-task specific parameters
    parser.add_argument('--balancing_strategy', type=str, default='weighted_sampling',
                        choices=['weighted_sampling', 'equal_sampling', 'natural'],
                        help='Dataset balancing strategy')
    parser.add_argument('--task_weights', type=str, default='1.0,1.0,1.0',
                        help='Comma-separated task weights for loss (mpii,wflw,300w)')
    parser.add_argument('--use_mpii_landmarks', action='store_true',
                        help='Extract MPII-style 6 landmarks from WFLW and 300W datasets')
    parser.add_argument('--single_head', action='store_true',
                        help='Use single-head FacialLandmarkEstimator instead of multi-head model')
    
    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_multi', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of DataLoader workers')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    
    # MPII specific
    parser.add_argument('--mpii_train_participants', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument('--mpii_val_participants', type=str, default="12,13,14")
    
    return parser.parse_args()


def create_multi_task_batch(batch_data):
    """
    Reorganize batch data by task for multi-task processing.
    """
    task_data = defaultdict(lambda: defaultdict(list))
    
    for sample in batch_data:
        task = sample['task']
        for key, value in sample.items():
            if key != 'task':
                task_data[task][key].append(value)
    
    # Convert lists to tensors
    for task in task_data:
        for key in task_data[task]:
            if isinstance(task_data[task][key][0], torch.Tensor):
                task_data[task][key] = torch.stack(task_data[task][key])
            else:
                task_data[task][key] = task_data[task][key]  # Keep as list for strings
    
    return dict(task_data)


def multi_task_collate_fn(batch):
    """
    Custom collate function for multi-task batches.
    Groups samples by task before collating to handle different tensor sizes.
    """
    if not batch:
        return {}
    
    # Group samples by task
    task_batches = defaultdict(list)
    for sample in batch:
        task = sample['task']
        task_batches[task].append(sample)
    
    # Collate each task separately
    collated_tasks = {}
    for task, task_samples in task_batches.items():
        # For each task, collate all samples
        collated_task = {}
        
        # Get all keys except 'task'
        sample_keys = set()
        for sample in task_samples:
            sample_keys.update(sample.keys())
        sample_keys.discard('task')
        
        for key in sample_keys:
            values = [sample[key] for sample in task_samples]
            
            if isinstance(values[0], torch.Tensor):
                # Stack tensors
                collated_task[key] = torch.stack(values)
            else:
                # Keep as list for non-tensors (strings, etc.)
                collated_task[key] = values
        
        collated_tasks[task] = collated_task
    
    return collated_tasks


def get_eye_indices_for_task(task, use_mpii_landmarks=False):
    """Get eye landmark indices for NME calculation based on task"""
    if task == 'mpii' or use_mpii_landmarks:
        return 0, 3  # Left outer eye, right outer eye
    elif task == 'wflw':
        return 60, 72  # Left outer eye, right outer eye
    elif task == '300w':
        return 36, 45  # Left outer eye, right outer eye
    else:
        raise ValueError(f"Unknown task: {task}")


def train_one_epoch_multi_task(model, dataloader, criterion, optimizer, scaler, device, epoch, args):
    model.train()
    loss_accumulator = torch.tensor(0.0, device=device)
    task_loss_accumulators = defaultdict(lambda: torch.tensor(0.0, device=device))
    task_nme_accumulators = defaultdict(lambda: torch.tensor(0.0, device=device))
    task_sample_counts = defaultdict(int)
    start_time = time.time()
    
    for batch_idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        # batch_data is already organized by task from our custom collate function
        task_data = batch_data
        
        optimizer.zero_grad(set_to_none=True)
        
        predictions_dict = {}
        targets_dict = {}
        
        # Forward pass for each task in the batch
        for task, data in task_data.items():
            images = data['image'].to(device, non_blocking=True)
            landmarks = data['landmarks'].to(device, non_blocking=True)
            
            # Flatten landmarks for loss calculation
            targets_flat = landmarks.view(landmarks.size(0), -1)
            targets_dict[task] = targets_flat
            
            if args.amp:
                with torch.amp.autocast(device_type='cuda'):
                    # Check if model is single-head or multi-head
                    if hasattr(args, 'single_head') and args.single_head and args.use_mpii_landmarks:
                        outputs = model(images)  # Single-head model doesn't take task parameter
                    else:
                        outputs = model(images, task=task)  # Multi-head model takes task parameter
                    predictions_dict[task] = outputs
            else:
                # Check if model is single-head or multi-head
                if hasattr(args, 'single_head') and args.single_head and args.use_mpii_landmarks:
                    outputs = model(images)  # Single-head model doesn't take task parameter
                else:
                    outputs = model(images, task=task)  # Multi-head model takes task parameter
                predictions_dict[task] = outputs
            
            task_sample_counts[task] += landmarks.size(0)
        
        # Calculate multi-task loss
        if args.amp:
            with torch.amp.autocast(device_type='cuda'):
                total_loss, task_losses = criterion(predictions_dict, targets_dict)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss, task_losses = criterion(predictions_dict, targets_dict)
            total_loss.backward()
            optimizer.step()
        
        loss_accumulator += total_loss.detach()
        
        # Accumulate task-specific metrics
        for task in task_losses:
            task_loss_accumulators[task] += task_losses[task]
            
            # Calculate NME for this task
            outputs_reshaped = predictions_dict[task].detach().view(predictions_dict[task].size(0), -1, 2)
            landmarks_reshaped = targets_dict[task].detach().view(targets_dict[task].size(0), -1, 2)
            
            # Use task-specific eye indices
            left_eye_idx, right_eye_idx = get_eye_indices_for_task(task, args.use_mpii_landmarks)
            current_nme = NME(outputs_reshaped, landmarks_reshaped, 
                            left_eye_idx=left_eye_idx, right_eye_idx=right_eye_idx)
            task_nme_accumulators[task] += current_nme.detach()
        
        # Logging
        if (batch_idx + 1) % args.log_interval == 0 or batch_idx == len(dataloader) - 1:
            avg_total_loss = loss_accumulator.item() / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            
            log_str = f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
            log_str += f'Total Loss: {total_loss.item():.6f} (Avg: {avg_total_loss:.6f}), '
            
            for task in task_losses:
                avg_task_loss = task_loss_accumulators[task].item() / (batch_idx + 1)
                log_str += f'{task.upper()}: {task_losses[task]:.6f} (Avg: {avg_task_loss:.6f}), '
            
            log_str += f'Time: {elapsed_time:.2f}s'
            print(log_str)
            start_time = time.time()
    
    # Calculate final averages
    avg_total_loss = (loss_accumulator / len(dataloader)).item()
    
    task_metrics = {}
    for task in task_loss_accumulators:
        avg_task_loss = (task_loss_accumulators[task] / len(dataloader)).item()
        avg_task_nme = (task_nme_accumulators[task] / len(dataloader)).item()
        task_metrics[task] = {
            'loss': avg_task_loss,
            'nme': avg_task_nme,
            'samples': task_sample_counts[task]
        }
    
    return avg_total_loss, task_metrics


def validate_multi_task(model, dataloader, criterion, device, args):
    model.eval()
    loss_accumulator = torch.tensor(0.0, device=device)
    task_loss_accumulators = defaultdict(lambda: torch.tensor(0.0, device=device))
    task_nme_accumulators = defaultdict(lambda: torch.tensor(0.0, device=device))
    task_sample_counts = defaultdict(int)
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, total=len(dataloader), desc="Validating"):
            # batch_data is already organized by task from our custom collate function
            task_data = batch_data
            
            predictions_dict = {}
            targets_dict = {}
            
            for task, data in task_data.items():
                images = data['image'].to(device, non_blocking=True)
                landmarks = data['landmarks'].to(device, non_blocking=True)
                
                targets_flat = landmarks.view(landmarks.size(0), -1)
                targets_dict[task] = targets_flat
                
                with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                    # Check if model is single-head or multi-head
                    if hasattr(args, 'single_head') and args.single_head and args.use_mpii_landmarks:
                        outputs = model(images)  # Single-head model doesn't take task parameter
                    else:
                        outputs = model(images, task=task)  # Multi-head model takes task parameter
                    predictions_dict[task] = outputs
                
                task_sample_counts[task] += landmarks.size(0)
            
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                total_loss, task_losses = criterion(predictions_dict, targets_dict)
            
            loss_accumulator += total_loss.detach()
            
            for task in task_losses:
                task_loss_accumulators[task] += task_losses[task]
                
                outputs_reshaped = predictions_dict[task].detach().view(predictions_dict[task].size(0), -1, 2)
                landmarks_reshaped = targets_dict[task].detach().view(targets_dict[task].size(0), -1, 2)
                
                left_eye_idx, right_eye_idx = get_eye_indices_for_task(task, args.use_mpii_landmarks)
                current_nme = NME(outputs_reshaped, landmarks_reshaped,
                                left_eye_idx=left_eye_idx, right_eye_idx=right_eye_idx)
                task_nme_accumulators[task] += current_nme.detach()
    
    avg_total_loss = (loss_accumulator / len(dataloader)).item()
    
    task_metrics = {}
    for task in task_loss_accumulators:
        avg_task_loss = (task_loss_accumulators[task] / len(dataloader)).item()
        avg_task_nme = (task_nme_accumulators[task] / len(dataloader)).item()
        task_metrics[task] = {
            'loss': avg_task_loss,
            'nme': avg_task_nme,
            'samples': task_sample_counts[task]
        }
    
    print(f'Validation Summary - Total Loss: {avg_total_loss:.6f}')
    for task, metrics in task_metrics.items():
        print(f'  {task.upper()}: Loss: {metrics["loss"]:.6f}, NME: {metrics["nme"]:.6f}, Samples: {metrics["samples"]}')
    print()
    
    return avg_total_loss, task_metrics


def main():
    args = get_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    setup_device(device)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Parse task weights
    task_weight_values = [float(w) for w in args.task_weights.split(',')]
    task_weights = {'mpii': task_weight_values[0], 'wflw': task_weight_values[1], '300w': task_weight_values[2]}
    
    # Validate arguments
    if args.single_head and not args.use_mpii_landmarks:
        raise ValueError("single_head can only be used with use_mpii_landmarks=True")
    
    # Image transforms
    if args.input_channels == 1:
        normalize = transforms.Normalize(mean=[0.449], std=[0.226])
        transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_list = [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    
    train_transform = transforms.Compose(transform_list + [
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    val_transform = transforms.Compose(transform_list)
    
    # Create datasets
    datasets_train = {}
    datasets_val = {}
    
    # MPII Dataset
    if args.mpii_data_dir:
        train_ids = [int(p_id) for p_id in args.mpii_train_participants.split(',')]
        val_ids = [int(p_id) for p_id in args.mpii_val_participants.split(',')]
        
        datasets_train['mpii'] = MPIIFaceGazeDataset(
            dataset_path=args.mpii_data_dir,
            participant_ids=train_ids,
            transform=train_transform,
            is_train=True,
            affine_aug=True,
            flip_aug=True
        )
        datasets_val['mpii'] = MPIIFaceGazeDataset(
            dataset_path=args.mpii_data_dir,
            participant_ids=val_ids,
            transform=val_transform,
            is_train=False
        )
    
    # WFLW Dataset
    if args.wflw_data_dir and args.wflw_train_annotation and args.wflw_val_annotation:
        datasets_train['wflw'] = WFLWDataset(
            annotation_file=args.wflw_train_annotation,
            images_dir=args.wflw_data_dir,
            transform=train_transform,
            is_train=True,
            affine_aug=True,
            flip_aug=True,
            mpii_landmarks=args.use_mpii_landmarks
        )
        datasets_val['wflw'] = WFLWDataset(
            annotation_file=args.wflw_val_annotation,
            images_dir=args.wflw_data_dir,
            transform=val_transform,
            is_train=False,
            mpii_landmarks=args.use_mpii_landmarks
        )
    
    # 300W Dataset
    if args.face300w_data_dir:
        datasets_train['300w'] = Face300WDataset(
            root_dir=args.face300w_data_dir,
            transform=train_transform,
            is_train=True,
            affine_aug=True,
            flip_aug=True,
            mpii_landmarks=args.use_mpii_landmarks,
            split='train'
        )
        datasets_val['300w'] = Face300WDataset(
            root_dir=args.face300w_data_dir,
            transform=val_transform,
            is_train=False,
            mpii_landmarks=args.use_mpii_landmarks,
            split='test'
        )
    
    if not datasets_train:
        print("Error: No datasets specified. Please provide at least one dataset.")
        return
    
    print(f"Training on datasets: {list(datasets_train.keys())}")
    
    # Create multi-dataset wrappers
    train_multi_dataset = MultiDatasetWrapper(
        datasets_train, 
        balancing_strategy=args.balancing_strategy
    )
    val_multi_dataset = MultiDatasetWrapper(
        datasets_val, 
        balancing_strategy='natural'  # Use natural proportions for validation
    )
    
    # Create data loaders
    if args.balancing_strategy == 'weighted_sampling':
        sampler = WeightedRandomSampler(
            weights=train_multi_dataset.sample_weights,
            num_samples=len(train_multi_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_multi_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=multi_task_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_multi_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=multi_task_collate_fn
        )
    
    val_loader = DataLoader(
        val_multi_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Determine output dimensions based on landmarks used
    if args.use_mpii_landmarks:
        output_dims = {'mpii': 6, 'wflw': 6, '300w': 6}
    else:
        output_dims = {'mpii': 6, 'wflw': 98, '300w': 68}
    
    # Create model
    if args.single_head and args.use_mpii_landmarks:
        # Use single-head model for MPII-style landmarks
        model = FacialLandmarkEstimator(
            num_landmarks=6,  # MPII landmarks
            pretrained_backbone=True,
            in_channels=args.input_channels,
            dropout_rate=args.dropout_rate
        ).to(device)
        print("Using single-head FacialLandmarkEstimator with 6 MPII landmarks")
    else:
        # Use multi-head model
        model = MHFacialLandmarkEstimator(
            output_dims=output_dims,
            pretrained_backbone=True,
            in_channels=args.input_channels,
            dropout_rate=args.dropout_rate
        ).to(device)
        print(f"Using multi-head MHFacialLandmarkEstimator with output dimensions: {output_dims}")
    
    # Add model type flag for training functions
    is_single_head = args.single_head and args.use_mpii_landmarks
    
    # Loss and optimizer
    criterion = MultiTaskLoss(task_weights=task_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device=device, enabled=args.amp)
    
    # Training loop
    best_val_loss = float('inf')
    
    # Initialize history with task-specific metrics
    available_tasks = list(datasets_train.keys())
    history = {
        'train_loss_total': [],
        'val_loss_total': [],
        'epochs': []
    }
    
    # Add task-specific metric tracking
    for task in available_tasks:
        history[f'train_loss_{task}'] = []
        history[f'val_loss_{task}'] = []
        history[f'train_nme_{task}'] = []
        history[f'val_nme_{task}'] = []
    
    history_filepath = os.path.join(args.checkpoint_dir, 'multi_task_history.json')
    
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Training
        train_loss, train_task_metrics = train_one_epoch_multi_task(
            model, train_loader, criterion, optimizer, scaler, device, epoch, args
        )
        
        # Validation
        val_loss, val_task_metrics = validate_multi_task(
            model, val_loader, criterion, device, args
        )
        
        # Save history
        history['train_loss_total'].append(train_loss)
        history['val_loss_total'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Save task-specific metrics
        for task in available_tasks:
            if task in train_task_metrics:
                history[f'train_loss_{task}'].append(train_task_metrics[task]['loss'])
                history[f'train_nme_{task}'].append(train_task_metrics[task]['nme'])
            else:
                # If task not present in this epoch, append NaN or previous value
                history[f'train_loss_{task}'].append(float('nan'))
                history[f'train_nme_{task}'].append(float('nan'))
            
            if task in val_task_metrics:
                history[f'val_loss_{task}'].append(val_task_metrics[task]['loss'])
                history[f'val_nme_{task}'].append(val_task_metrics[task]['nme'])
            else:
                history[f'val_loss_{task}'].append(float('nan'))
                history[f'val_nme_{task}'].append(float('nan'))
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            
            # Use appropriate model info for checkpoint
            if args.single_head and args.use_mpii_landmarks:
                save_checkpoint(epoch, model, optimizer, None, scaler if args.amp else None, 
                              best_checkpoint_path, 6, 'single_head_mpii')
            else:
                save_checkpoint(epoch, model, optimizer, None, scaler if args.amp else None, 
                              best_checkpoint_path, output_dims, 'multi_task')
            
            save_history(history, history_filepath)
            print(f"New best validation loss: {best_val_loss:.6f}")
        
        # Save history periodically
        if (epoch + 1) % 10 == 0:
            save_history(history, history_filepath)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Save final history
    save_history(history, history_filepath)
    
    # Create visualizations
    plot_path = os.path.join(args.checkpoint_dir, 'multi_task_training_history.png')
    plot_multi_task_training_history(history, available_tasks, plot_path)
    print(f"Multi-task training history plot saved to {plot_path}")
    
    # Create metrics summary
    summary_path = os.path.join(args.checkpoint_dir, 'training_summary.txt')
    create_metrics_summary(history, available_tasks, summary_path)
    
    print("Training finished!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main() 