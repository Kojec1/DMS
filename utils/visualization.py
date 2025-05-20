import matplotlib.pyplot as plt
import os


def plot_training_history(train_loss_history, val_loss_history, lr_history, save_path):
    """
    Plots the training/validation loss and learning rate history using subplots, then saves it.

    Args:
        train_loss_history (list): List of training losses for each epoch.
        val_loss_history (list): List of validation losses for each epoch.
        lr_history (list): List of learning rates for each epoch.
        save_path (str): Path to save the plot image.
    """
    epochs = range(1, len(train_loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # 2 rows, 1 column

    # Subplot 1: Training and Validation Loss
    color_loss = 'tab:red'
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_yscale('log') # Often useful for loss
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--')
    ax1.set_title('Training and Validation Loss')

    # Subplot 2: Learning Rate
    color_lr = 'tab:blue'
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate', color=color_lr)
    ax2.plot(epochs, lr_history, 'go-', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color_lr)
    ax2.set_yscale('log') # LR often changes over orders of magnitude
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--')
    ax2.set_title('Learning Rate')

    fig.suptitle('Training Metrics', fontsize=16)
    fig.tight_layout()
    # Adjust layout to make space for suptitle and prevent overlap
    plt.subplots_adjust(top=0.90, hspace=0.3)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()