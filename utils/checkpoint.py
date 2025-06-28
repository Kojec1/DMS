import torch
import os
import json


def save_checkpoint(epoch, model, optimizer, scheduler, scaler, filepath, num_landmarks=None, dataset=None):
    """
    Saves a checkpoint of the model, optimizer, scheduler, and scaler with model metadata.
    """    
    print(f"Saving checkpoint to {filepath}")
    state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
    }

    # Get the state_dict from the original model if it was compiled
    unwrapped_model = getattr(model, '_orig_mod', model)
    state['model_state_dict'] = unwrapped_model.state_dict()

    if scaler is not None: # For AMP
        state['scaler_state_dict'] = scaler.state_dict()
    
    # Store model metadata
    if num_landmarks is not None:
        state['num_landmarks'] = num_landmarks
    if dataset is not None:
        state['dataset'] = dataset
        
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer, scheduler, scaler):
    """
    Loads a checkpoint of the model, optimizer, scheduler, and scaler.
    """
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
        return start_epoch
    else:
        print(f"No checkpoint found at '{filepath}'")  
        return 0

def save_history(history, filepath):
    """
    Saves the history dictionary to a file using torch.save.
    """
    print(f"Saving training history to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)

def load_history(filepath):
    """
    Loads the history dictionary from a file using torch.load.
    """
    if os.path.isfile(filepath):
        try:
            print(f"Loading training history from {filepath}")
            with open(filepath, 'r') as f:
                history = json.load(f)
            return history
        except Exception as e:
            print(f"Error loading history file {filepath}: {e}")
            return None
    return None

