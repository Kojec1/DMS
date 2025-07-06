import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import warnings
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def extract_run_index(file_path: str) -> int:
    """Extract the numeric run index from a path containing 'run_val_<idx>'."""
    match = re.search(r"run_val_(\d+)", str(file_path))
    return int(match.group(1)) if match else -1

def find_training_history_files(root_dir: str) -> List[str]:
    """Recursively find all training_history.json files in the given directory."""
    history_files = []
    root_path = Path(root_dir)
    
    for file_path in root_path.rglob('training_history.json'):
        history_files.append(str(file_path))
    
    history_files.sort(key=extract_run_index)
    
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
    
    # Create figure with subplots (5 rows, 2 columns)
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle(f'Training Analysis - {len(all_histories)} Participants', fontsize=16, fontweight='bold')
    
    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
    
    # Collect metrics for histograms
    best_metrics = {
        'train_landmark_nme': [],
        'val_landmark_nme': [],
        'train_ang_error': [],
        'val_ang_error': []
    }
    
    # Plot 1-6: Loss curves (arranged in pairs)
    loss_pairs = [
        (('train_total_loss', 'Train Total Loss'), ('val_total_loss', 'Val Total Loss')),
        (('train_landmark_loss', 'Train Landmark Loss'), ('val_landmark_loss', 'Val Landmark Loss')),
        (('train_gaze_loss', 'Train Gaze Loss'), ('val_gaze_loss', 'Val Gaze Loss'))
    ]
    
    for row, ((train_key, train_title), (val_key, val_title)) in enumerate(loss_pairs):
        # Train loss (left column)
        ax_train = axes[row, 0]
        for j, (file_path, history) in enumerate(all_histories):
            if train_key in history and history[train_key]:
                epochs = range(1, len(history[train_key]) + 1)
                run_idx = extract_run_index(file_path)
                ax_train.plot(
                    epochs,
                    history[train_key],
                    color=colors[j],
                    alpha=0.7,
                    label=f'P{run_idx}' if row == 0 else "",
                )
        
        ax_train.set_title(train_title, fontweight='bold')
        ax_train.set_xlabel('Epoch')
        ax_train.set_ylabel('Loss')
        ax_train.grid(True, alpha=0.3)
        
        if row == 0:  # Add legend only to first subplot
            ax_train.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Val loss (right column)
        ax_val = axes[row, 1]
        for j, (file_path, history) in enumerate(all_histories):
            if val_key in history and history[val_key]:
                epochs = range(1, len(history[val_key]) + 1)
                ax_val.plot(epochs, history[val_key], color=colors[j], alpha=0.7)
        
        ax_val.set_title(val_title, fontweight='bold')
        ax_val.set_xlabel('Epoch')
        ax_val.set_ylabel('Loss')
        ax_val.grid(True, alpha=0.3)
    
    # Extract metrics at best validation loss for each run
    for file_path, history in all_histories:
        metrics = extract_metrics_at_best_val_loss(history)
        for key in best_metrics:
            if key in metrics:
                best_metrics[key].append(metrics[key])
    
    # Plot 7-10: Bar charts
    bar_configs = [
        ('train_landmark_nme', 'Train Landmark NME', 3, 0),
        ('val_landmark_nme', 'Val Landmark NME', 3, 1),
        ('train_ang_error', 'Train Angular Error', 4, 0),
        ('val_ang_error', 'Val Angular Error', 4, 1)
    ]
    
    for key, title, row, col in bar_configs:
        ax = axes[row, col]
        
        if best_metrics[key]:
            data = best_metrics[key]
            n_runs = len(data)
            avg_val = np.mean(data)
            
            # Create bar chart with run indices + average
            x_positions = list(range(1, n_runs + 1)) + [n_runs + 1.5]  # Gap before average
            y_values = data + [avg_val]
            bar_colors = ['skyblue'] * n_runs + ['red']
            
            bars = ax.bar(x_positions, y_values, color=bar_colors, alpha=0.7, edgecolor='black')
            
            # Add value labels on bars
            for i, (x, y) in enumerate(zip(x_positions, y_values)):
                ax.text(x, y + max(y_values) * 0.01, f'{y:.4f}', 
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Participant Index')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set x-axis labels
            first_run_idx = extract_run_index(history_files[0]) if history_files else 0
            x_labels = [f'P{extract_run_index(history_files[i])}' for i in range(n_runs)] + ['Avg']
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
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
