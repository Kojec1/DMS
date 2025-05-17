import torch
import os


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, filepath):
    """
    Saves a checkpoint of the model, optimizer, scheduler, and scaler.
    """
    print(f"=> Saving checkpoint to {filepath}")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    if scaler is not None: # For AMP
        state['scaler_state_dict'] = scaler.state_dict()
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer, scheduler, scaler):
    """
    Loads a checkpoint of the model, optimizer, scheduler, and scaler.
    """
    if os.path.isfile(filepath):
        print(f"=> Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"=> Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
        return start_epoch
    else:
        print(f"=> No checkpoint found at '{filepath}'")
        return 0