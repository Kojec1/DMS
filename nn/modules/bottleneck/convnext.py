import torch
import torch.nn as nn
from torchvision.models import convnext_tiny as tv_convnext_tiny
from torchvision.models.convnext import LayerNorm2d
from typing import Optional
from functools import partial


def convnext_tiny(pretrained: bool = False, **kwargs) -> nn.Module:
    """ConvNeXt Tiny model from torchvision, with the classification head removed."""
    # Create model from torchvision
    weights = 'DEFAULT' if pretrained else None
    model = tv_convnext_tiny(weights=weights, **kwargs)

    norm_layer = partial(LayerNorm2d, eps=1e-6)  # LayerNorm2d 
    
    # Remove the classification head
    feature_extractor = nn.Sequential(*list(model.children())[:-1], norm_layer(768))
    feature_extractor.out_features = 768
    
    return feature_extractor