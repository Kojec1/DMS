import matplotlib.pyplot as plt
import os


def plot_training_history(train_loss_history, val_loss_history, lr_history, save_path):
    """
    Plots the training/validation loss and learning rate history, then saves it.

    Args:
        train_loss_history (list): List of training losses for each epoch.
        val_loss_history (list): List of validation losses for each epoch.
        lr_history (list): List of learning rates for each epoch.
        save_path (str): Path to save the plot image.
    """
    epochs = range(1, len(train_loss_history) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Training and Validation Loss on ax1 (left y-axis)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # Often useful for loss
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--')

    # Create a second y-axis (ax2) for the Learning Rate, sharing the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, lr_history, 'go-', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log') # LR often changes over orders of magnitude
    ax2.legend(loc='upper right')

    fig.suptitle('Training Metrics', fontsize=16)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # Adjust layout to make space for suptitle
    plt.subplots_adjust(top=0.92)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()