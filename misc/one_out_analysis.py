import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def find_training_history_files(root_dir: str) -> List[str]:
    """Recursively find all training_history.json files in the given directory."""
    history_files = []
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob('training_history.json'):
        history_files.append(str(file_path))
    
    return history_files

def load_training_history(file_path: str) -> Dict[str, Any]:
    """Load training history from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_metrics_at_best_val_loss(history: Dict[str, Any]) -> Dict[str, float]:
    """Extract metrics at the epoch with the lowest validation total loss."""
    if not history or 'val_total_loss' not in history:
        return {}
    
    val_losses = history['val_total_loss']
    if not val_losses:
        return {}
    
    # Find index of minimum validation loss
    best_idx = np.argmin(val_losses)
    
    metrics = {}
    for key in ['train_landmark_nme', 'val_landmark_nme', 'train_ang_error', 'val_ang_error']:
        if key in history and len(history[key]) > best_idx:
            metrics[key] = history[key][best_idx]
    
    return metrics

def create_visualization(root_dir: str, output_path: str = None):
    """Create comprehensive visualization of training histories."""
    # Find all training history files
    history_files = find_training_history_files(root_dir)
    
    if not history_files:
        print(f"No training_history.json files found in {root_dir}")
        return
    
    print(f"Found {len(history_files)} training history files")
    
    # Load all training histories
    all_histories = []
    for file_path in history_files:
        history = load_training_history(file_path)
        if history:
            all_histories.append((file_path, history))
    
    if not all_histories:
        print("No valid training histories found")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    fig.suptitle(f'Training Analysis - {len(all_histories)} Runs', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    # Collect metrics for histograms
    best_metrics = {
        'train_landmark_nme': [],
        'val_landmark_nme': [],
        'train_ang_error': [],
        'val_ang_error': []
    }
    
    # Plot 1-6: Loss curves
    loss_keys = [
        ('train_total_loss', 'Train Total Loss'),
        ('val_total_loss', 'Val Total Loss'),
        ('train_landmark_loss', 'Train Landmark Loss'),
        ('val_landmark_loss', 'Val Landmark Loss'),
        ('train_gaze_loss', 'Train Gaze Loss'),
        ('val_gaze_loss', 'Val Gaze Loss')
    ]
    
    for i, (key, title) in enumerate(loss_keys):
        ax = axes[i]
        
        for j, (file_path, history) in enumerate(all_histories):
            if key in history and history[key]:
                epochs = range(1, len(history[key]) + 1)
                ax.plot(epochs, history[key], color=colors[j], alpha=0.7, 
                       label=f'Run {j+1}' if i == 0 else "")
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Add legend only to first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Extract metrics at best validation loss for each run
    for file_path, history in all_histories:
        metrics = extract_metrics_at_best_val_loss(history)
        for key in best_metrics:
            if key in metrics:
                best_metrics[key].append(metrics[key])
    
    # Plot 7-10: Histograms
    histogram_configs = [
        ('train_landmark_nme', 'Train Landmark NME at Best Val Loss', 6),
        ('val_landmark_nme', 'Val Landmark NME at Best Val Loss', 7),
        ('train_ang_error', 'Train Angular Error at Best Val Loss', 8),
        ('val_ang_error', 'Val Angular Error at Best Val Loss', 9)
    ]
    
    for key, title, idx in histogram_configs:
        ax = axes[idx]
        
        if best_metrics[key]:
            data = best_metrics[key]
            ax.hist(data, bins=max(5, len(data)//2), alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add average line
            avg_val = np.mean(data)
            ax.axvline(avg_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Avg: {avg_val:.4f}')
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total runs analyzed: {len(all_histories)}")
    
    for key, title in [
        ('train_landmark_nme', 'Train Landmark NME'),
        ('val_landmark_nme', 'Val Landmark NME'),
        ('train_ang_error', 'Train Angular Error'),
        ('val_ang_error', 'Val Angular Error')
    ]:
        if best_metrics[key]:
            data = best_metrics[key]
            print(f"\n{title} (at best val loss):")
            print(f"  Mean: {np.mean(data):.4f}")
            print(f"  Std:  {np.std(data):.4f}")
            print(f"  Min:  {np.min(data):.4f}")
            print(f"  Max:  {np.max(data):.4f}")

def main():
    """Main function to run the visualization script."""
    parser = argparse.ArgumentParser(description='Visualize training histories from multiple runs')
    parser.add_argument('--dir', help='Root directory to search for training_history.json files')
    parser.add_argument('--output', '-o', help='Output path for the plot (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist")
        return
    
    create_visualization(args.dir, args.output)

if __name__ == "__main__":
    main()
