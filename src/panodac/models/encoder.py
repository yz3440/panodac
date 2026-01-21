"""Encoder factory functions."""

from functools import partial
from typing import Any

from torch import nn
from timm.models.vision_transformer import _cfg

from .backbones import Bottleneck, SwinTransformer, _resnet


def swin_large_22k(pretrained: bool = True, **kwargs):
    """Swin-L pretrained on ImageNet-22k."""
    if pretrained:
        pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dims=[192 * (2**i) for i in range(4)],
        num_heads=[6, 12, 24, 48],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 18, 2],
        drop_path_rate=0.2,
        pretrained=pretrained,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


def resnet101(pretrained: bool = True, progress: bool = True, **kwargs: Any):
    """ResNet-101 backbone."""
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = True, progress: bool = True, **kwargs: Any):
    """ResNet-50 backbone."""
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
