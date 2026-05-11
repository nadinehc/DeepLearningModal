import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels=in_chans, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x comes in as (B, T, C, H, W)
        # should return (B, T', H', W', embed_dim)

        # Permute to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.proj(x)  # (B, embed_dim, T', H', W')
        # Permute back to (B, T', H', W', embed_dim)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x