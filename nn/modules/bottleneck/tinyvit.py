import torch
import torch.nn as nn
import timm


class TinyViTFeatureExtractor(nn.Module):
	"""Wrapper around timm TinyViT."""
	def __init__(self, model_name: str, pretrained: bool, in_chans: int) -> None:
		super().__init__()
		self.backbone = timm.create_model(
			model_name,
			pretrained=pretrained,
			in_chans=in_chans,
			num_classes=0,
			global_pool="",
		)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.out_features = None

		# Infer feature dimensionality with a dummy pass
		with torch.no_grad():
			dummy = torch.zeros(1, in_chans, 224, 224)
			features = self._forward_features(dummy)
			self.out_features = features.shape[1]

	def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone.forward_features(x)
		if isinstance(features, (list, tuple)):
			features = features[-1]
		features = self.pool(features)
		return features

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self._forward_features(x)


def tiny_vit_11m(pretrained: bool = False, in_chans: int = 3, **kwargs) -> nn.Module:
	"""
	TinyViT-11M (224) feature extractor via timm.
	"""
	return TinyViTFeatureExtractor("tiny_vit_11m_224", pretrained=pretrained, in_chans=in_chans)
