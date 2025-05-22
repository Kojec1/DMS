import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2 as tv_mobilenet_v2
from typing import Optional


def mobilenet_v2(pretrained: bool = False, **kwargs) -> nn.Module:
    """
    MobileNetV2 model from torchvision, with the classification head removed.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        nn.Module: MobileNetV2 model without classification head
    """
    # Create model from torchvision
    weights = 'DEFAULT' if pretrained else None
    model = tv_mobilenet_v2(weights=weights, **kwargs)
    
    # Remove the classification head and add pooling and flattening
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.add_module('adaptive_avg_pool2d', nn.AdaptiveAvgPool2d((1, 1)))
    feature_extractor.add_module('flatten', nn.Flatten(start_dim=1, end_dim=-1))
    feature_extractor.out_features = 1280
    
    return feature_extractor 

if __name__ == '__main__':
    model = mobilenet_v2(pretrained=True)
    print(model)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(output.shape)