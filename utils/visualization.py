import matplotlib.pyplot as plt
import os


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
    color_loss = 'tab:red'
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_yscale('log')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')

    # Subplot 2: Learning Rate
    color_lr = 'tab:blue'
    ax2.set_ylabel('Learning Rate', color=color_lr)
    ax2.plot(epochs, lr_history, 'go-', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color_lr)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--')
    ax2.set_title('Learning Rate')
    ax2.set_xlabel('Epochs')

    # Subplot 3: Training and Validation NME
    color_nme = 'tab:green'
    ax3.set_ylabel('NME', color=color_nme)
    ax3.plot(epochs, train_nme_history, 'co-', label='Training NME')
    ax3.plot(epochs, val_nme_history, 'mo-', label='Validation NME')
    ax3.tick_params(axis='y', labelcolor=color_nme)
    ax3.set_yscale('log')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--')
    ax3.set_title('Training and Validation NME')
    ax3.set_xlabel('Epochs')

    # Subplot 4: Training and Validation MSE
    color_mse = 'tab:purple'
    ax4.set_ylabel('MSE', color=color_mse)
    ax4.plot(epochs, train_mse_history, 'yo-', label='Training MSE')
    ax4.plot(epochs, val_mse_history, 'ko-', label='Validation MSE')
    ax4.tick_params(axis='y', labelcolor=color_mse)
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