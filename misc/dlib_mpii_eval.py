import torch
import numpy as np
import dlib
import time
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import urllib.request

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import MPIIFaceGazeDataset
from nn.metrics.landmark_metrics import NME
from utils.misc import set_seed

# Mapping from dlib's 68-point landmarks to MPII's 6-point format
DLIB_TO_MPII_MAPPING = {
    0: 36,  # Left outer eye
    1: 39,  # Left inner eye  
    2: 42,  # Right inner eye
    3: 45,  # Right outer eye
    4: 48,  # Left mouth corner
    5: 54   # Right mouth corner
}

def download_dlib_model(model_path):
    """Download dlib's facial landmark predictor model if it doesn't exist."""
    if os.path.exists(model_path):
        print(f"Dlib model already exists at: {model_path}")
        return
    
    print(f"Downloading dlib facial landmark predictor model...")
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = model_path + ".bz2"
    
    try:
        urllib.request.urlretrieve(model_url, compressed_path)
        print(f"Downloaded compressed model to: {compressed_path}")
        
        # Decompress the file
        import bz2
        with bz2.BZ2File(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        # Remove compressed file
        os.remove(compressed_path)
        print(f"Decompressed and saved model to: {model_path}")
        
    except Exception as e:
        print(f"Error downloading dlib model: {e}")
        print("Please manually download the model from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(f"Extract it and place it at: {model_path}")
        raise

def tensor_to_cv2(tensor_image):
    """Convert tensor image to OpenCV format."""
    # Convert from tensor (C, H, W) to numpy (H, W, C) if needed
    if tensor_image.dim() == 3:
        if tensor_image.shape[0] == 1:  # Grayscale (1, H, W)
            image_np = tensor_image.squeeze(0).numpy()
        else:  # RGB (3, H, W)
            image_np = tensor_image.permute(1, 2, 0).numpy()
    else:  # Already (H, W)
        image_np = tensor_image.numpy()
    
    # Convert to uint8 if needed
    if image_np.dtype != np.uint8:
        # Assuming tensor is normalized [0, 1], convert to [0, 255]
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
    
    return image_np

def extract_mpii_landmarks(dlib_landmarks):
    """Extract 6 MPII-style landmarks from dlib's 68-point landmarks."""
    mpii_landmarks = np.zeros((6, 2), dtype=np.float32)
    
    for mpii_idx, dlib_idx in DLIB_TO_MPII_MAPPING.items():
        point = dlib_landmarks.part(dlib_idx)
        mpii_landmarks[mpii_idx] = [point.x, point.y]
    
    return mpii_landmarks

def normalize_landmarks_to_image_size(landmarks, img_width, img_height):
    """Normalize landmarks to [0, 1] range based on image dimensions."""
    normalized = landmarks.copy().astype(np.float32)
    normalized[:, 0] /= img_width
    normalized[:, 1] /= img_height
    return normalized

def evaluate_dlib_on_mpii(data_dir, participant_ids, dlib_model_path, device='cpu', 
                         batch_size=1, num_workers=0, max_samples=None):
    """Evaluate dlib facial landmark detector on MPII dataset."""
    
    # Initialize dlib predictor
    predictor = dlib.shape_predictor(dlib_model_path)
    
    # Create dataset
    dataset = MPIIFaceGazeDataset(
        dataset_path=data_dir,
        participant_ids=participant_ids,
        transform=None,
        is_train=False,
        affine_aug=False,
        flip_aug=False,
        use_cache=False,
        label_smoothing=0.0  # No label smoothing
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    
    print(f"Evaluating dlib on {len(dataset)} MPII samples from participants {participant_ids}")
    
    # Evaluation metrics
    total_samples = 0
    nme_accumulator = 0.0
    mse_accumulator = 0.0
    mae_accumulator = 0.0
    inference_times = []
    
    # Process samples
    sample_count = 0
    
    for _, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating dlib"):
        if max_samples and sample_count >= max_samples:
            break
            
        # Get batch data
        images = batch_data['image']
        gt_landmarks = batch_data['facial_landmarks'].numpy()
        image_paths = batch_data['image_path']
        
        batch_size_actual = len(image_paths)
        
        for i in range(batch_size_actual):
            if max_samples and sample_count >= max_samples:
                break
                
            sample_count += 1
            total_samples += 1
            
            image_tensor = images[i]
            gt_landmark = gt_landmarks[i]
            image_path = image_paths[i]
            
            try:
                cv2_image = tensor_to_cv2(image_tensor)
                img_height, img_width = cv2_image.shape[:2]
                
                # Create a dummy rectangle that covers the whole image
                face_rect = dlib.rectangle(0, 0, img_width, img_height)
                
                # Measure inference time
                start_time = time.time()
                
                # Predict landmarks
                landmarks = predictor(cv2_image, face_rect)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Extract MPII-style landmarks
                pred_landmarks = extract_mpii_landmarks(landmarks)
                
                # Normalize predicted landmarks to [0, 1] range
                pred_landmarks_normalized = normalize_landmarks_to_image_size(pred_landmarks, img_width, img_height)
                
                # Calculate metrics using the same approach as for the neural network
                pred_tensor = torch.from_numpy(pred_landmarks_normalized).unsqueeze(0).to(device)
                gt_tensor = torch.from_numpy(gt_landmark).unsqueeze(0).to(device)
                
                # Calculate NME (using outer eye corners as normalizing factor)
                nme_value = NME(pred_tensor, gt_tensor, left_eye_idx=0, right_eye_idx=3)
                nme_accumulator += nme_value.item()
                
                # Calculate MSE (Mean Squared Error)
                mse_value = torch.nn.functional.mse_loss(pred_tensor, gt_tensor)
                mse_accumulator += mse_value.item()
                
                # Calculate MAE (Mean Absolute Error)
                mae_value = torch.nn.functional.l1_loss(pred_tensor, gt_tensor)
                mae_accumulator += mae_value.item()
                
            except Exception as e:
                print(f"Error processing sample {sample_count} ({image_path}): {e}")
                continue
    
    # Calculate final metrics
    avg_nme = nme_accumulator / total_samples if total_samples > 0 else float('inf')
    avg_mse = mse_accumulator / total_samples if total_samples > 0 else float('inf')
    avg_mae = mae_accumulator / total_samples if total_samples > 0 else float('inf')
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    
    results = {
        'total_samples': total_samples,
        'avg_nme': avg_nme,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'avg_inference_time_ms': avg_inference_time * 1000,  # Convert to milliseconds
        'inference_times': inference_times
    }
    
    return results

def print_results(results, split_name):
    """Print evaluation results in a formatted manner."""
    print(f"\n=== Dlib Evaluation Results ({split_name}) ===")
    print(f"Total samples: {results['total_samples']}")
    print(f"Average NME: {results['avg_nme']:.6f}")
    print(f"Average MSE: {results['avg_mse']:.6f}")
    print(f"Average MAE: {results['avg_mae']:.6f}")
    print(f"Average inference time: {results['avg_inference_time_ms']:.2f} ms")
    
    if results['inference_times']:
        inference_times_ms = np.array(results['inference_times']) * 1000
        print(f"Inference time stats (ms):")
        print(f"  Min: {np.min(inference_times_ms):.2f}")
        print(f"  Max: {np.max(inference_times_ms):.2f}")
        print(f"  Std: {np.std(inference_times_ms):.2f}")
        print(f"  Median: {np.median(inference_times_ms):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate dlib facial landmark detector on MPII dataset')
    
    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Root directory for MPII dataset')
    parser.add_argument('--train_participant_ids', type=str, default="0,1,2,3,4,5,6,7,8,9,10,11",
                        help='Comma-separated list of participant IDs for training evaluation')
    parser.add_argument('--val_participant_ids', type=str, default="12,13,14",
                        help='Comma-separated list of participant IDs for validation evaluation')
    
    # Dlib model arguments
    parser.add_argument('--dlib_model_path', type=str, default='./models/shape_predictor_68_face_landmarks.dat',
                        help='Path to dlib facial landmark predictor model')
    parser.add_argument('--download_model', action='store_true',
                        help='Download dlib model if it does not exist')
    
    # Evaluation arguments
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                        help='Which split to evaluate: train, val, or both')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for data loading (recommended: 1 for dlib)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for DataLoader')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate per split (for quick testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for metric calculations (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Parse participant IDs
    try:
        train_ids = [int(p_id) for p_id in args.train_participant_ids.split(',')]
        val_ids = [int(p_id) for p_id in args.val_participant_ids.split(',')]
    except ValueError:
        print("Error: Participant IDs must be comma-separated integers.")
        return
    
    # Setup dlib model
    os.makedirs(os.path.dirname(args.dlib_model_path), exist_ok=True)
    
    if args.download_model or not os.path.exists(args.dlib_model_path):
        download_dlib_model(args.dlib_model_path)
    
    if not os.path.exists(args.dlib_model_path):
        print(f"Error: Dlib model not found at {args.dlib_model_path}")
        print("Please use --download_model flag or manually download the model.")
        return
    
    print(f"Using dlib model: {args.dlib_model_path}")
    print(f"Evaluating on MPII dataset: {args.data_dir}")
    print(f"Train participant IDs: {train_ids}")
    print(f"Val participant IDs: {val_ids}")
    
    # Evaluate on requested splits
    all_results = {}
    
    if args.split in ['train', 'both']:
        print(f"\n--- Evaluating on Training Split ---")
        train_results = evaluate_dlib_on_mpii(
            data_dir=args.data_dir,
            participant_ids=train_ids,
            dlib_model_path=args.dlib_model_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )
        all_results['train'] = train_results
        print_results(train_results, 'Training')
    
    if args.split in ['val', 'both']:
        print(f"\n--- Evaluating on Validation Split ---")
        val_results = evaluate_dlib_on_mpii(
            data_dir=args.data_dir,
            participant_ids=val_ids,
            dlib_model_path=args.dlib_model_path,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples
        )
        all_results['val'] = val_results
        print_results(val_results, 'Validation')
    
    # Print summary comparison
    print(f"\n=== Summary ===")
    for split_name, results in all_results.items():
        print(f"{split_name.capitalize()}: NME={results['avg_nme']:.6f}, "
              f"MSE={results['avg_mse']:.6f}, MAE={results['avg_mae']:.6f}, "
              f"Inference Time={results['avg_inference_time_ms']:.2f}ms")
    
    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()
