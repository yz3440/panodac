"""
IDisc - Implicit Discontinuity-aware Depth model (standard version).

Based on the original work by Luigi Piccinelli, modified for DAC.
"""

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .defattn_decoder import MSDeformAttnPixelDecoder
from .fpn_decoder import BasePixelDecoder
from .id_module import AFP, ISD


class IDisc(nn.Module):
    """Implicit Discontinuity-aware Depth model."""

    def __init__(
        self,
        pixel_encoder: nn.Module,
        afp: nn.Module,
        pixel_decoder: nn.Module,
        isd: nn.Module,
        afp_min_resolution: int = 1,
        eps: float = 1e-6,
        **kwargs
    ):
        super().__init__()
        self.eps = eps
        self.pixel_encoder = pixel_encoder
        self.afp = afp
        self.pixel_decoder = pixel_decoder
        self.isd = isd
        self.afp_min_resolution = afp_min_resolution

    def invert_encoder_output_order(
        self, xs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(xs[::-1])

    def filter_decoder_relevant_resolutions(
        self, decoder_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(decoder_outputs[self.afp_min_resolution:])

    def forward(
        self,
        image: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        original_shape = gt.shape[-2:] if gt is not None else image.shape[-2:]

        encoder_outputs = self.pixel_encoder(image)
        encoder_outputs = self.invert_encoder_output_order(encoder_outputs)

        fpn_outputs, decoder_outputs = self.pixel_decoder(encoder_outputs)

        decoder_outputs = self.filter_decoder_relevant_resolutions(decoder_outputs)
        fpn_outputs = self.filter_decoder_relevant_resolutions(fpn_outputs)

        idrs = self.afp(decoder_outputs)
        outs = self.isd(fpn_outputs, idrs)

        out_lst = []
        for out in outs:
            if out.shape[1] == 1:
                out = F.interpolate(
                    torch.exp(out),
                    size=outs[-1].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                out = self.normalize_normals(
                    F.interpolate(
                        out,
                        size=outs[-1].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            out_lst.append(out)

        out = F.interpolate(
            torch.mean(torch.stack(out_lst, dim=0), dim=0),
            original_shape,
            mode="bilinear" if out.shape[1] == 1 else "bicubic",
            align_corners=True,
        )
        return out if out.shape[1] == 1 else out[:, :3]

    def normalize_normals(self, norms):
        min_kappa = 0.01
        norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
        norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
        kappa = F.elu(kappa) + 1.0 + min_kappa
        norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
        return norms

    def load_pretrained(self, model_file):
        """Load pretrained weights."""
        from ..utils import get_device
        device = get_device()
        dict_model = torch.load(model_file, map_location=device, weights_only=False)
        if 'model' in dict_model:
            dict_model = dict_model['model']
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )
        # Remove loss-related keys (not needed for inference)
        new_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith("loss")}
        self.load_state_dict(new_state_dict, strict=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):
        """Build model from configuration."""
        import importlib
        
        pixel_encoder_img_size = config["model"]["pixel_encoder"]["img_size"]
        pixel_encoder_pretrained = config["model"]["pixel_encoder"].get("pretrained", None)
        config_backbone = {"img_size": np.array(pixel_encoder_img_size)}
        if pixel_encoder_pretrained is not None:
            config_backbone["pretrained"] = pixel_encoder_pretrained

        # Import encoder factory
        from . import encoder as encoder_mod
        pixel_encoder_factory = getattr(encoder_mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder = pixel_encoder_factory(**config_backbone)

        pixel_encoder_embed_dims = getattr(pixel_encoder, "embed_dims")
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims

        pixel_decoder = (
            MSDeformAttnPixelDecoder.build(config)
            if config["model"]["attn_dec"]
            else BasePixelDecoder.build(config)
        )
        afp = AFP.build(config)
        isd = ISD.build(config)

        return deepcopy(
            cls(
                pixel_encoder=pixel_encoder,
                pixel_decoder=pixel_decoder,
                afp=afp,
                isd=isd,
                afp_min_resolution=len(pixel_encoder_embed_dims)
                - config["model"]["isd"]["num_resolutions"],
            )
        )
