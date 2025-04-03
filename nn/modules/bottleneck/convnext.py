import torch
import torch.nn as nn
from torchvision.models import convnext_tiny as tv_convnext_tiny
from typing import Optional


def convnext_tiny(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    ConvNeXt Tiny model from torchvision, with the classification head removed.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        nn.Module: ConvNeXt Tiny model without classification head
    """
    # Create model from torchvision
    weights = 'DEFAULT' if pretrained else None
    model = tv_convnext_tiny(weights=weights, **kwargs)
    
    # Remove the classification head
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    
    return feature_extractor