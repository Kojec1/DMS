import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_device(device: torch.device):
    if device.type == 'cuda':
        # Set memory allocation strategy for better performance
        torch.backends.cudnn.benchmark = True
        print("CUDA optimizations enabled: cudnn.benchmark=True")

        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        print("CUDA optimizations enabled: cudnn.deterministic=False")

        # Enable optimized attention for better GPU utilization (PyTorch 2.0+)
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
            print("CUDA optimizations enabled: enable_math_sdp=True")
