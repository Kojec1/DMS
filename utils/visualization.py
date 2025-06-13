import matplotlib.pyplot as plt
import os


def plot_training_history(history, save_path):
    """
    Plots the training/validation history for the multi-head model using subplots, then saves it.
    """
    # Determine the number of epochs from one of the history lists
    if not history or not history.get('train_total_loss'):
        print("Warning: History is empty or missing key 'train_total_loss'. Cannot generate plot.")
        return
        
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Training & Validation Metrics', fontsize=20)
    axes = axes.flatten()

    # Plotting function
    def plot_metric(ax, train_key, val_key, title, y_label, log_scale=True):
        ax.plot(epochs, history[train_key], 'o-', color='royalblue', label=f'Training {y_label}')
        ax.plot(epochs, history[val_key], 'o-', color='orangered', label=f'Validation {y_label}')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if log_scale:
            ax.set_yscale('log')
        ax.legend()

    # 1. Total Loss
    plot_metric(axes[0], 'train_total_loss', 'val_total_loss', 'Total Loss', 'Loss')

    # 2. Landmark Loss
    plot_metric(axes[1], 'train_landmark_loss', 'val_landmark_loss', 'Landmark Loss', 'Loss')

    # 3. Gaze Loss (MSE)
    plot_metric(axes[2], 'train_gaze_loss', 'val_gaze_loss', 'Gaze Loss (MSE)', 'Loss')

    # 4. Landmark NME
    plot_metric(axes[3], 'train_landmark_nme', 'val_landmark_nme', 'Landmark NME', 'NME')

    # 5. Gaze MAE (as a metric)
    plot_metric(axes[4], 'train_gaze_mae', 'val_gaze_mae', 'Gaze MAE (Metric)', 'MAE')

    # 6. Learning Rate
    if 'lr' in history and history['lr']:
        axes[5].plot(epochs, history['lr'], 'o-', color='forestgreen', label='Learning Rate')
        axes[5].set_title('Learning Rate Schedule', fontsize=14)
        axes[5].set_xlabel('Epochs', fontsize=10)
        axes[5].set_ylabel('Learning Rate', fontsize=10)
        axes[5].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[5].legend()
    else:
        axes[5].set_visible(False) # Hide if no LR history

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()