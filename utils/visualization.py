import matplotlib.pyplot as plt
import os
import numpy as np


def plot_training_history(train_loss_history, val_loss_history, 
                          train_nme_history, val_nme_history,
                          train_mse_history, val_mse_history,
                          lr_history, save_path):
    """
    Plots the training/validation loss, NME, MSE, and learning rate history using subplots, then saves it.
    """
    epochs = range(1, len(train_loss_history) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12)) # 2 rows, 2 columns

    # Subplot 1: Training and Validation Loss
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    ax1.tick_params(axis='y')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')

    # Subplot 2: Learning Rate
    ax2.set_ylabel('Learning Rate')
    ax2.plot(epochs, lr_history, 'bo-', label='Learning Rate')
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--')
    ax2.set_title('Learning Rate')
    ax2.set_xlabel('Epochs')

    # Subplot 3: Training and Validation NME
    ax3.set_ylabel('NME')
    ax3.plot(epochs, train_nme_history, 'bo-', label='Training NME')
    ax3.plot(epochs, val_nme_history, 'ro-', label='Validation NME')
    ax3.tick_params(axis='y')
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--')
    ax3.set_title('Training and Validation NME')
    ax3.set_xlabel('Epochs')

    # Subplot 4: Training and Validation MSE
    ax4.set_ylabel('MSE')
    ax4.plot(epochs, train_mse_history, 'bo-', label='Training MSE')
    ax4.plot(epochs, val_mse_history, 'ro-', label='Validation MSE')
    ax4.tick_params(axis='y')
    ax4.set_yscale('log')
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--')
    ax4.set_title('Training and Validation MSE')
    ax4.set_xlabel('Epochs')

    fig.suptitle('Training Metrics', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()


def plot_multi_task_training_history(history, tasks, save_path):
    """
    Plots multi-task training history with task-specific metrics.
    Each metric type is plotted in a separate subplot with different colors for each task.
    
    Args:
        history (dict): Dictionary containing training history
        tasks (list): List of task names (e.g., ['mpii', 'wflw', '300w'])
        save_path (str): Path to save the plot
    """
    epochs = history['epochs']
    
    # Define colors for each task
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    task_colors = {task: colors[i % len(colors)] for i, task in enumerate(tasks)}
    
    # Create subplots: 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    
    # Subplot 1: Total Loss (Training and Validation)
    ax1.plot(epochs, history['train_loss_total'], 'b-', linewidth=2, label='Total Training Loss')
    ax1.plot(epochs, history['val_loss_total'], 'r-', linewidth=2, label='Total Validation Loss')
    ax1.set_ylabel('Total Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_title('Total Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_yscale('log')
    
    # Subplot 2: Task-specific Training Loss
    for task in tasks:
        train_loss_key = f'train_loss_{task}'
        if train_loss_key in history:
            # Filter out NaN values for plotting
            task_epochs = []
            task_losses = []
            for i, loss in enumerate(history[train_loss_key]):
                if not np.isnan(loss):
                    task_epochs.append(epochs[i])
                    task_losses.append(loss)
            
            if task_losses:  # Only plot if we have data
                ax2.plot(task_epochs, task_losses, color=task_colors[task], 
                        marker='o', markersize=3, linewidth=1.5, label=f'{task.upper()} Training')
    
    ax2.set_ylabel('Training Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_title('Task-specific Training Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yscale('log')
    
    # Subplot 3: Task-specific Validation Loss
    for task in tasks:
        val_loss_key = f'val_loss_{task}'
        if val_loss_key in history:
            # Filter out NaN values for plotting
            task_epochs = []
            task_losses = []
            for i, loss in enumerate(history[val_loss_key]):
                if not np.isnan(loss):
                    task_epochs.append(epochs[i])
                    task_losses.append(loss)
            
            if task_losses:  # Only plot if we have data
                ax3.plot(task_epochs, task_losses, color=task_colors[task], 
                        marker='s', markersize=3, linewidth=1.5, label=f'{task.upper()} Validation')
    
    ax3.set_ylabel('Validation Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_title('Task-specific Validation Loss')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_yscale('log')
    
    # Subplot 4: Task-specific NME (both training and validation)
    for task in tasks:
        train_nme_key = f'train_nme_{task}'
        val_nme_key = f'val_nme_{task}'
        
        # Plot training NME
        if train_nme_key in history:
            task_epochs = []
            task_nmes = []
            for i, nme in enumerate(history[train_nme_key]):
                if not np.isnan(nme):
                    task_epochs.append(epochs[i])
                    task_nmes.append(nme)
            
            if task_nmes:
                ax4.plot(task_epochs, task_nmes, color=task_colors[task], 
                        linestyle='-', marker='o', markersize=2, linewidth=1, 
                        alpha=0.7, label=f'{task.upper()} Train NME')
        
        # Plot validation NME
        if val_nme_key in history:
            task_epochs = []
            task_nmes = []
            for i, nme in enumerate(history[val_nme_key]):
                if not np.isnan(nme):
                    task_epochs.append(epochs[i])
                    task_nmes.append(nme)
            
            if task_nmes:
                ax4.plot(task_epochs, task_nmes, color=task_colors[task], 
                        linestyle='--', marker='s', markersize=2, linewidth=1.5, 
                        label=f'{task.upper()} Val NME')
    
    ax4.set_ylabel('NME')
    ax4.set_xlabel('Epochs')
    ax4.set_title('Task-specific Normalized Mean Error (NME)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_yscale('log')
    
    # Overall title and layout
    fig.suptitle('Multi-Task Facial Landmark Estimation Training History', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-task training history visualization saved to: {save_path}")


def create_metrics_summary(history, tasks, save_path):
    """
    Creates a text summary of the final metrics for each task.
    
    Args:
        history (dict): Dictionary containing training history
        tasks (list): List of task names
        save_path (str): Path to save the summary text file
    """
    summary_lines = []
    summary_lines.append("=== Multi-Task Training Summary ===\n")
    summary_lines.append(f"Total epochs trained: {len(history['epochs'])}\n")
    summary_lines.append(f"Tasks: {', '.join([task.upper() for task in tasks])}\n\n")
    
    # Final epoch metrics
    if history['epochs']:
        final_epoch = history['epochs'][-1]
        summary_lines.append(f"Final Epoch ({final_epoch}) Metrics:\n")
        summary_lines.append("-" * 50 + "\n")
        
        # Total loss
        if history['train_loss_total'] and history['val_loss_total']:
            final_train_total = history['train_loss_total'][-1]
            final_val_total = history['val_loss_total'][-1]
            summary_lines.append(f"Total Loss - Train: {final_train_total:.6f}, Val: {final_val_total:.6f}\n\n")
        
        # Task-specific metrics
        for task in tasks:
            summary_lines.append(f"{task.upper()} Task:\n")
            
            # Get final values (last non-NaN value)
            train_loss_key = f'train_loss_{task}'
            val_loss_key = f'val_loss_{task}'
            train_nme_key = f'train_nme_{task}'
            val_nme_key = f'val_nme_{task}'
            
            def get_last_valid_value(values):
                for val in reversed(values):
                    if not np.isnan(val):
                        return val
                return None
            
            if train_loss_key in history:
                final_train_loss = get_last_valid_value(history[train_loss_key])
                if final_train_loss is not None:
                    summary_lines.append(f"  Training Loss: {final_train_loss:.6f}\n")
            
            if val_loss_key in history:
                final_val_loss = get_last_valid_value(history[val_loss_key])
                if final_val_loss is not None:
                    summary_lines.append(f"  Validation Loss: {final_val_loss:.6f}\n")
            
            if train_nme_key in history:
                final_train_nme = get_last_valid_value(history[train_nme_key])
                if final_train_nme is not None:
                    summary_lines.append(f"  Training NME: {final_train_nme:.6f}\n")
            
            if val_nme_key in history:
                final_val_nme = get_last_valid_value(history[val_nme_key])
                if final_val_nme is not None:
                    summary_lines.append(f"  Validation NME: {final_val_nme:.6f}\n")
            
            summary_lines.append("\n")
        
        # Best validation performance
        summary_lines.append("Best Validation Performance:\n")
        summary_lines.append("-" * 30 + "\n")
        
        if history['val_loss_total']:
            best_val_total_idx = np.argmin(history['val_loss_total'])
            best_val_total = history['val_loss_total'][best_val_total_idx]
            best_epoch = history['epochs'][best_val_total_idx]
            summary_lines.append(f"Best Total Val Loss: {best_val_total:.6f} (Epoch {best_epoch})\n\n")
        
        # Task-specific best performance
        for task in tasks:
            val_loss_key = f'val_loss_{task}'
            val_nme_key = f'val_nme_{task}'
            
            if val_loss_key in history:
                valid_losses = [(i, loss) for i, loss in enumerate(history[val_loss_key]) if not np.isnan(loss)]
                if valid_losses:
                    best_idx, best_loss = min(valid_losses, key=lambda x: x[1])
                    best_epoch = history['epochs'][best_idx]
                    summary_lines.append(f"{task.upper()} Best Val Loss: {best_loss:.6f} (Epoch {best_epoch})\n")
            
            if val_nme_key in history:
                valid_nmes = [(i, nme) for i, nme in enumerate(history[val_nme_key]) if not np.isnan(nme)]
                if valid_nmes:
                    best_idx, best_nme = min(valid_nmes, key=lambda x: x[1])
                    best_epoch = history['epochs'][best_idx]
                    summary_lines.append(f"{task.upper()} Best Val NME: {best_nme:.6f} (Epoch {best_epoch})\n")
    
    # Save summary to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.writelines(summary_lines)
    
    print(f"Training summary saved to: {save_path}")
    
    # Also print to console
    print("\n" + "".join(summary_lines))