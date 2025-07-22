import torch.nn as nn
from torchvision.models import mobilenet_v2 as tv_mobilenet_v2


def mobilenet_v2(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    MobileNetV2 model from torchvision, with the classification head removed.
    """
    # Create model from torchvision
    weights = 'DEFAULT' if pretrained else None
    model = tv_mobilenet_v2(weights=weights, **kwargs)
    
    # Remove the classification head and add pooling and flattening
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.add_module('adaptive_avg_pool2d', nn.AdaptiveAvgPool2d((1, 1)))
    feature_extractor.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
    
    return feature_extractor 