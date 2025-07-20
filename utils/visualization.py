import matplotlib.pyplot as plt
import os
import math


def plot_training_history(history, save_path):
    """Plots the training/validation history for the multi-head model using subplots, then saves it."""
    if not history or not history.get('train_total_loss'):
        print("Warning: History is empty or missing key 'train_total_loss'. Cannot generate plot.")
        return
        
    epochs = range(1, len(history['train_total_loss']) + 1)
    
    # Determine metric availability
    has_landmark = 'train_landmark_loss' in history and history['train_landmark_loss']
    has_gaze = 'train_gaze_loss' in history and history['train_gaze_loss']
    has_head_pose = 'train_head_pose_loss' in history and history['train_head_pose_loss']
    has_awl = 'awl_weights' in history and history['awl_weights']

    task_list = []
    if has_landmark:
        task_list.append('landmark')
    if has_gaze:
        task_list.append('gaze')
    if has_head_pose:
        task_list.append('head_pose')
    
    # Calculate number of plots needed
    num_plots = 3  # Total Loss, Landmark Loss, Landmark NME
    if has_gaze:
        num_plots += 2  # Gaze Loss, Gaze Angular Error
    if has_head_pose:
        num_plots += 2  # Head Pose Loss, Head Pose Angular Error
    if has_awl:
        num_plots += 1  # AWL weights
    num_plots += 1  # Learning Rate
    
    # Determine grid size
    nrows = math.ceil(num_plots / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(16, 4 * nrows))
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
            return False
        return True

    plot_idx = 0
    
    # 1. Total Loss (always present)
    if plot_metric(axes[plot_idx], 'train_total_loss', 'val_total_loss', 'Total Loss', 'Loss'):
        plot_idx += 1

    # 2. Landmark Loss
    if has_landmark and plot_metric(axes[plot_idx], 'train_landmark_loss', 'val_landmark_loss', 'Landmark Loss', 'Loss'):
        plot_idx += 1

    # 3. Gaze Loss (if present)
    if has_gaze and plot_metric(axes[plot_idx], 'train_gaze_loss', 'val_gaze_loss', 'Gaze Loss (RCS)', 'Loss'):
        plot_idx += 1

    # 4. Head Pose Loss (if present)
    if has_head_pose and plot_metric(axes[plot_idx], 'train_head_pose_loss', 'val_head_pose_loss', 'Head Pose Loss (RCS)', 'Loss'):
        plot_idx += 1

    # 5. Landmark NME
    if has_landmark and plot_metric(axes[plot_idx], 'train_landmark_nme', 'val_landmark_nme', 'Landmark NME', 'NME'):
        plot_idx += 1

    # 6. Gaze Angular Error (if present)
    if has_gaze and plot_metric(axes[plot_idx], 'train_ang_error', 'val_ang_error', 'Gaze Angular Error (°)', 'Error', log_scale=False):
        plot_idx += 1

    # 7. Head Pose Angular Error (if present)
    if has_head_pose and plot_metric(axes[plot_idx], 'train_head_ang_error', 'val_head_ang_error', 'Head Pose Angular Error (°)', 'Error', log_scale=False):
        plot_idx += 1

    # 8. AWL weights (if present)
    if has_awl:
        n_tasks = len(history['awl_weights'][0])
        for i in range(n_tasks):
            axes[plot_idx].plot(epochs, [weights[i] for weights in history['awl_weights']], 'o-', color='purple', label=f'{task_list[i].capitalize()}')
        axes[plot_idx].set_title('AWL Weights', fontsize=14)
        axes[plot_idx].set_xlabel('Epochs', fontsize=10)
        axes[plot_idx].set_ylabel('Weights', fontsize=10)
        axes[plot_idx].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[plot_idx].legend()

        plot_idx += 1

    # Learning Rates
    for lr_key, color in zip(['lr_backbone', 'lr_landmark', 'lr_gaze', 'lr_head_pose'], ['forestgreen', 'skyblue', 'coral', 'purple']):
        if lr_key in history and history[lr_key]:
            axes[plot_idx].plot(epochs, history[lr_key], 'o-', color=color, label=f'{lr_key.split("_")[-1].capitalize()}')
    axes[plot_idx].set_title('Learning Rate Schedule', fontsize=14)
    axes[plot_idx].set_xlabel('Epochs', fontsize=10)
    axes[plot_idx].set_ylabel('Learning Rate', fontsize=10)
    axes[plot_idx].set_yscale('log')
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