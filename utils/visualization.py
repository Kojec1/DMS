import matplotlib.pyplot as plt
import os


def plot_training_history(history, save_path):
    """Plots the training/validation history for the multi-head model using subplots, then saves it."""
    # Determine the number of epochs from one of the history lists
    if not history or not history.get('train_total_loss'):
        print("Warning: History is empty or missing key 'train_total_loss'. Cannot generate plot.")
        return
        
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Determine metric availability
    has_new_gaze = 'train_gaze_loss' in history and history['train_gaze_loss']
    has_ang_err = 'train_ang_error' in history and history['train_ang_error']
    
    # Determine grid size based on available metrics
    if has_new_gaze:
        # Format with new gaze metrics - use larger grid
        fig, axes = plt.subplots(4, 3, figsize=(24, 24))
        fig.suptitle('Training & Validation Metrics (Gaze)', fontsize=20)
    else:
        # Landmarks-only - use smaller grid
        fig, axes = plt.subplots(3, 2, figsize=(20, 18))
        fig.suptitle('Training & Validation Metrics', fontsize=20)
    
    axes = axes.flatten()

    # Plotting function
    def plot_metric(ax, train_key, val_key, title, y_label, log_scale=True):
        if train_key in history and val_key in history and history[train_key] and history[val_key]:
            ax.plot(epochs, history[train_key], 'o-', color='royalblue', label=f'Training {y_label}')
            ax.plot(epochs, history[val_key], 'o-', color='orangered', label=f'Validation {y_label}')
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Epochs', fontsize=10)
            ax.set_ylabel(y_label, fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            if log_scale:
                ax.set_yscale('log')
            ax.legend()
        else:
            ax.set_visible(False)

    plot_idx = 0
    
    # 1. Total Loss (always present)
    plot_metric(axes[plot_idx], 'train_total_loss', 'val_total_loss', 'Total Loss', 'Loss')
    plot_idx += 1

    # 2. Landmark Loss
    plot_metric(axes[plot_idx], 'train_landmark_loss', 'val_landmark_loss', 'Landmark Loss', 'Loss')
    plot_idx += 1

    if has_new_gaze:
        # 3. Gaze Loss
        plot_metric(axes[plot_idx], 'train_gaze_loss', 'val_gaze_loss', 'Gaze Loss (RCS)', 'Loss')
        plot_idx += 1

        # 4. Landmark NME
        plot_metric(axes[plot_idx], 'train_landmark_nme', 'val_landmark_nme', 'Landmark NME', 'NME')
        plot_idx += 1

        # 5. Angular Error
        if has_ang_err:
            plot_metric(axes[plot_idx], 'train_ang_error', 'val_ang_error', 'Gaze Angular Error (°)', 'Error', log_scale=False)
            plot_idx += 1

    else:
        # Landmarks-only format
        
        # 3. Landmark NME
        plot_metric(axes[plot_idx], 'train_landmark_nme', 'val_landmark_nme', 'Landmark NME', 'NME')
        plot_idx += 1

    # Learning Rate (always last)
    if 'lr' in history and history['lr']:
        axes[plot_idx].plot(epochs, history['lr'], 'o-', color='forestgreen', label='Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule', fontsize=14)
        axes[plot_idx].set_xlabel('Epochs', fontsize=10)
        axes[plot_idx].set_ylabel('Learning Rate', fontsize=10)
        axes[plot_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[plot_idx].legend()
        plot_idx += 1

    # Hide any remaining unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()