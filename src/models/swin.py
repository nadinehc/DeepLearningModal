import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
import numpy as np


def window_partition(x, window_size):
    """Partition input tensor into non-overlapping windows."""
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0], window_size[0],
        H // window_size[1], window_size[1],
        W // window_size[2], window_size[2],
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """Reverse window partition back to original tensor."""
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0], window_size[1], window_size[2],
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


# ─────────────────────────────────────────────
# Relative Position Bias
# ─────────────────────────────────────────────

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, N

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1) 
        ].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ─────────────────────────────────────────────
# Swin Transformer Block (3D)
# ─────────────────────────────────────────────

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # Pad
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # Cyclic shift
        if any(s > 0 for s in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        if any(s > 0 for s in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        x = shortcut + self.drop_path(self.forward_part1(x, mask_matrix))
        x = x + self.forward_part2(x)
        return x


# ─────────────────────────────────────────────
# Drop Path
# ─────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ─────────────────────────────────────────────
# Patch Merging
# ─────────────────────────────────────────────

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ─────────────────────────────────────────────
# Patch Embedding (3D)
# ─────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=128, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # x: (B, C, T, H, W)
        _, _, T, H, W = x.shape
        # Pad spatial dims to be divisible by patch_size
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, T', H', W', embed_dim)
        x = self.norm(x)
        return x


# ─────────────────────────────────────────────
# Basic Layer (Stage)
# ─────────────────────────────────────────────

class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(s // 2 for s in window_size)
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim) if downsample is not None else None

    def _compute_attn_mask(self, D, H, W, device):
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        d_slices = (slice(0, -window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None))
        h_slices = (slice(0, -window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None))
        w_slices = (slice(0, -window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None))
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x):
        B, D, H, W, C = x.shape
        attn_mask = self._compute_attn_mask(D, H, W, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ─────────────────────────────────────────────
# Video Swin Transformer (Swin-B)
# ─────────────────────────────────────────────

class VideoSwinTransformer(nn.Module):
    """
    Video Swin-B for 33-class action recognition.
    Input: (B, 3, T, H, W) with T=4, H=W=224
    """

    def __init__(
        self,
        num_classes=33,
        patch_size=(2, 4, 4),
        embed_dim=128,              # Swin-B: 128
        depths=(2, 2, 18, 2),       # Swin-B depths
        num_heads=(4, 8, 16, 32),   # Swin-B heads
        window_size=(2, 7, 7),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,         # stochastic depth
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=3, embed_dim=embed_dim,
            norm_layer=nn.LayerNorm
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], # type: ignore
                downsample=PatchMerging if i < self.num_layers - 1 else None,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)), num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.patch_embed(x)    # (B, T', H', W', C)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)           # (B, T', H', W', C')

        x = self.norm(x)
        B, D, H, W, C = x.shape
        x = x.view(B, -1, C)       # (B, N, C)
        x = self.avgpool(x.transpose(1, 2)).squeeze(-1)  # (B, C)
        x = self.head(x)           # (B, num_classes)
        return x


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = VideoSwinTransformer(num_classes=33)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    dummy = torch.randn(2, 3, 4, 224, 224)  # batch=2, RGB, 4 frames, 224x224
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")   # expect (2, 33)