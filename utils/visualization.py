import matplotlib.pyplot as plt
import os


def plot_training_history(train_loss_history, val_loss_history, save_path):
    """
    Plots the training and validation loss history and saves it to a file.

    Args:
        train_loss_history (list): List of training losses for each epoch.
        val_loss_history (list): List of validation losses for each epoch.
        save_path (str): Path to save the plot image.
    """
    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_history, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss_history, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path)
    plt.close()