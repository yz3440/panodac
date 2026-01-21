"""Common layers and utilities for models."""

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _get_clones(module, N):
    return [deepcopy(module) for _ in range(N)]


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "silu":
        return F.silu
    raise RuntimeError(f"activation should be relu/gelu/silu, not {activation}.")


def _get_activation_cls(activation):
    """Return an activation class given a string."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    if activation == "silu":
        return nn.SiLU()
    raise RuntimeError(f"activation should be relu/gelu/silu, not {activation}.")


def get_norm(norm, out_channels):
    """Get normalization layer by name."""
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": nn.BatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "torchSyncBN": nn.SyncBatchNorm,
            "LN": lambda channels: LayerNorm(channels),
            "torchLN": lambda channels: nn.LayerNorm(channels),
        }[norm]
    return norm(out_channels)


def c2_xavier_fill(module: nn.Module) -> None:
    """Xavier initialization for Conv2d layers."""
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module) -> None:
    """MSRA initialization for Conv2d layers."""
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class Conv2d(torch.nn.Conv2d):
    """Conv2d with optional normalization and activation."""
    
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"
        x = F.conv2d(
            x, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm for 2D feature maps (channel-first)."""
    
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal 2D position embedding."""
    
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingSineERP(nn.Module):
    """Sinusoidal position embedding for Equirectangular Projection (ERP)."""
    
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        lat_range: Optional[torch.Tensor] = None,
        long_range: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_embed = torch.zeros(x.size(0), x.size(2), x.size(3), device=x.device)
        y_embed = torch.zeros(x.size(0), x.size(2), x.size(3), device=x.device)
        
        for b in range(lat_range.size(0)):
            x_embed[b] = torch.tile(
                torch.linspace(long_range[b, 0], long_range[b, 1], x.size(3), device=x.device).unsqueeze(0),
                (x.size(2), 1)
            )
            y_embed[b] = torch.tile(
                torch.linspace(lat_range[b, 0], lat_range[b, 1], x.size(2), device=x.device).unsqueeze(0),
                (x.size(3), 1)
            ).T
            
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class AttentionLayer(nn.Module):
    """Standard attention layer."""
    
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
        out_attn: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.pre_norm = pre_norm
        assert (hidden_dim % num_heads) == 0, "hidden_dim and num_heads are not divisible"
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = norm_layer(source_dim, eps=eps) if source_dim is not None else None

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(hidden_dim, sink_dim if output_dim is None else output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition
        self.out_attn = out_attn

    def forward(self, sink: torch.Tensor, source: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pre_norm:
            sink = self.norm(sink)
            if source is not None:
                source = self.norm_context(source)

        q = self.to_q(sink)
        source = source if source is not None else sink
        k, v = self.to_kv(source).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", q, k) * self.scale

        if self.sink_competition:
            attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            attn = attn / torch.sum(attn, dim=(-1,), keepdim=True)
        else:
            attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out)
        if not self.pre_norm:
            out = self.norm(out)

        if self.out_attn:
            return out, attn
        return out


class AttentionLayerIsoPE(nn.Module):
    """Attention layer with isolated positional encoding (used in ERP models)."""
    
    def __init__(
        self,
        sink_dim: int,
        hidden_dim: int,
        source_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
        pre_norm: bool = True,
        norm_layer=nn.LayerNorm,
        sink_competition: bool = False,
        qkv_bias: bool = True,
        eps: float = 1e-6,
        out_attn: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.pre_norm = pre_norm
        assert (hidden_dim % num_heads) == 0, "hidden_dim and num_heads are not divisible"
        self.scale = (hidden_dim // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(sink_dim, eps=eps)
        self.norm_context = norm_layer(source_dim, eps=eps) if source_dim is not None else None

        self.to_q = nn.Linear(sink_dim, hidden_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(
            sink_dim if source_dim is None else source_dim,
            hidden_dim * 2,
            bias=qkv_bias,
        )
        self.to_out = nn.Linear(hidden_dim, sink_dim if output_dim is None else output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sink_competition = sink_competition
        self.out_attn = out_attn

        self.sink_dim = sink_dim
        self.source_dim = source_dim if source_dim is not None else sink_dim

    def forward(
        self,
        sink: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        top_t: Optional[int] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            sink = sink.clone()
            sink[:, :, :self.sink_dim] = self.norm(sink[:, :, :self.sink_dim].clone())
            if source is not None:
                source = source.clone()
                source[:, :, :self.source_dim] = self.norm_context(source[:, :, :self.source_dim].clone())

        q = self.to_q(sink[:, :, :self.sink_dim].clone())
        source = source if source is not None else sink
        k, v = self.to_kv(source[:, :, :self.source_dim].clone()).chunk(2, dim=-1)

        q = torch.cat([q, sink[:, :, self.sink_dim:].clone()], dim=-1)
        k = torch.cat([k, source[:, :, self.source_dim:].clone()], dim=-1)
        v = torch.cat([v, source[:, :, self.source_dim:].clone()], dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=self.num_heads),
            (q, k, v),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", q, k) * self.scale
        
        if attn_mask is not None:
            similarity_matrix = similarity_matrix.masked_fill(
                attn_mask.transpose(-1, -2).repeat(1, similarity_matrix.shape[1], 1) == 0, -1e9
            )

        if top_t is not None:
            _, top_t_indices = torch.topk(similarity_matrix, top_t, dim=2)
            top_t_mask = torch.zeros_like(similarity_matrix)
            top_t_mask.scatter_(2, top_t_indices, 1)
            similarity_matrix = similarity_matrix.masked_fill(top_t_mask == 0, -1e9)

        if self.sink_competition:
            attn = F.softmax(similarity_matrix, dim=-2) + self.eps
            attn = attn / torch.sum(attn, dim=(-1,), keepdim=True)
        else:
            attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(attn)

        out = torch.einsum("bij, bjd -> bid", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.num_heads)

        out_feature = self.to_out(out[:, :, :self.sink_dim].clone())
        out = torch.cat([out_feature, out[:, :, self.sink_dim:].clone()], dim=-1)

        if self.out_attn:
            return out, attn
        return out
