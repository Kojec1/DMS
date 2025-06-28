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
    
    # Check if we have 2D/3D gaze metrics
    has_2d_gaze = 'train_gaze_2d_loss' in history and history['train_gaze_2d_loss']
    has_3d_gaze = 'train_gaze_3d_loss' in history and history['train_gaze_3d_loss']
    
    # Determine grid size based on available metrics
    if has_2d_gaze or has_3d_gaze:
        # Format with 2D/3D gaze metrics - use larger grid
        fig, axes = plt.subplots(4, 3, figsize=(24, 24))
        fig.suptitle('Training & Validation Metrics (2D/3D Gaze)', fontsize=20)
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

    if has_2d_gaze or has_3d_gaze:
        # Format with separate 2D/3D gaze metrics
        
        # 3. 2D Gaze Loss
        if has_2d_gaze:
            plot_metric(axes[plot_idx], 'train_gaze_2d_loss', 'val_gaze_2d_loss', '2D Gaze Loss (MSE)', 'Loss')
            plot_idx += 1
        
        # 4. 3D Gaze Loss
        if has_3d_gaze:
            plot_metric(axes[plot_idx], 'train_gaze_3d_loss', 'val_gaze_3d_loss', '3D Gaze Loss (MSE)', 'Loss')
            plot_idx += 1

        # 5. Landmark NME
        plot_metric(axes[plot_idx], 'train_landmark_nme', 'val_landmark_nme', 'Landmark NME', 'NME')
        plot_idx += 1

        # 6. 2D Gaze MSE
        if has_2d_gaze:
            plot_metric(axes[plot_idx], 'train_gaze_2d_mse', 'val_gaze_2d_mse', '2D Gaze MSE (Metric)', 'MSE', log_scale=False)
            plot_idx += 1

        # 7. 2D Gaze MAE
        if has_2d_gaze:
            plot_metric(axes[plot_idx], 'train_gaze_2d_mae', 'val_gaze_2d_mae', '2D Gaze MAE (Metric)', 'MAE', log_scale=False)
            plot_idx += 1

        # 8. 3D Gaze MSE
        if has_3d_gaze:
            plot_metric(axes[plot_idx], 'train_gaze_3d_mse', 'val_gaze_3d_mse', '3D Gaze MSE (Metric)', 'MSE', log_scale=False)
            plot_idx += 1

        # 9. 3D Gaze MAE
        if has_3d_gaze:
            plot_metric(axes[plot_idx], 'train_gaze_3d_mae', 'val_gaze_3d_mae', '3D Gaze MAE (Metric)', 'MAE', log_scale=False)
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