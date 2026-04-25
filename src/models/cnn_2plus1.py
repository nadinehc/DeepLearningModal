"""
Conv(2+1)D Video Classification Architecture
Dataset: Something-Something action detection
Input:   B x 3 x 4 x 224 x 224  (batch, RGB, 4 frames, H, W)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Core building block: factorised (2+1)D convolution
# ---------------------------------------------------------------------------

class Conv2Plus1D(nn.Module):
    """
    Factorises a 3D conv into:
      1. A 2D spatial conv  (kernel: 1 x kH x kW)
      2. A 1D temporal conv (kernel: kT x 1 x 1)
    with a BN + ReLU between the two for an extra non-linearity.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),   # (T, H, W)
        stride: tuple      = (1, 1, 1),
        padding: tuple     = (1, 1, 1),
    ):
        super().__init__()

        kT, kH, kW = kernel_size
        sT, sH, sW = stride
        pT, pH, pW = padding

        # Mid-channels for the spatial conv (match original 3D conv param count)
        mid_channels = max(
            1,
            int((in_channels * out_channels * kT * kH * kW) /
                (in_channels * kH * kW + out_channels * kT))
        )

        # --- 2D spatial conv (operates on each frame independently) ---
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(
                in_channels, mid_channels,
                kernel_size=(1, kH, kW),
                stride=(1, sH, sW),
                padding=(0, pH, pW),
                bias=False,
            ),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # --- 1D temporal conv (mixes information across frames) ---
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                mid_channels, out_channels,
                kernel_size=(kT, 1, 1),
                stride=(sT, 1, 1),
                padding=(pT, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return x


# ---------------------------------------------------------------------------
# Residual block built from two Conv(2+1)D layers
# ---------------------------------------------------------------------------

class ResBlock2Plus1D(nn.Module):
    """
    Residual block with two factorised (2+1)D convolutions.
    A 1x1x1 projection is used on the skip path whenever
    channels or spatial size change.

    Note: temporal stride is always 1 to preserve the 4-frame resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_stride: int = 1,
    ):
        super().__init__()

        stride = (1, spatial_stride, spatial_stride)
        padding = (1, 1, 1)

        self.conv1 = Conv2Plus1D(in_channels,  out_channels, stride=stride, padding=padding)
        self.conv2 = Conv2Plus1D(out_channels, out_channels, stride=(1,1,1), padding=padding)

        # Skip connection projection (needed when dims change)
        if spatial_stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(1, spatial_stride, spatial_stride),
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out + residual)
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class Conv2Plus1DNet(nn.Module):
    """
    Conv(2+1)D network for video action classification.

    Architecture:
        Stem   : Conv(2+1)D  3 -> 64,  spatial 7x7, temporal 3x1x1
        Stage 1: 2x ResBlock  64 ->  64,  spatial stride 2 on block 1
        Stage 2: 2x ResBlock  64 -> 128,  spatial stride 2 on block 1
        Stage 3: 2x ResBlock 128 -> 256,  spatial stride 2 on block 1
        Pool   : Global spatiotemporal average pool  -> 256-d vector
        Head   : Dropout(0.5) + Linear(256, num_classes)

    Input:  (B, 3, 4, 224, 224)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 174, dropout: float = 0.5):
        super().__init__()

        # --- Stem ---
        self.stem = Conv2Plus1D(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),       # spatial /2, temporal preserved
            padding=(1, 3, 3),
        )

        # --- Residual stages ---
        self.stage1 = self._make_stage( 64,  64, n_blocks=2, spatial_stride=2)
        self.stage2 = self._make_stage( 64, 128, n_blocks=2, spatial_stride=2)
        self.stage3 = self._make_stage(128, 256, n_blocks=2, spatial_stride=2)

        # --- Head ---
        self.pool    = nn.AdaptiveAvgPool3d(1)   # (B, 256, 1, 1, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        spatial_stride: int,
    ) -> nn.Sequential:
        blocks = [ResBlock2Plus1D(in_channels, out_channels, spatial_stride)]
        for _ in range(1, n_blocks):
            blocks.append(ResBlock2Plus1D(out_channels, out_channels, spatial_stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 4, 224, 224)
        x = self.stem(x)      # -> (B,  64, 4, 112, 112)
        x = self.stage1(x)    # -> (B,  64, 4,  56,  56)
        x = self.stage2(x)    # -> (B, 128, 4,  28,  28)
        x = self.stage3(x)    # -> (B, 256, 4,  14,  14)
        x = self.pool(x)      # -> (B, 256,  1,   1,   1)
        x = x.flatten(1)      # -> (B, 256)
        x = self.dropout(x)
        x = self.fc(x)        # -> (B, num_classes)
        return x


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = Conv2Plus1DNet(num_classes=174)
    model.eval()

    dummy = torch.zeros(2, 3, 4, 224, 224)   # batch of 2 videos
    with torch.no_grad():
        out = model(dummy)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(out.shape)}")
    print(f"Params : {total_params:,}  ({trainable:,} trainable)")

    # Intermediate feature map sizes
    with torch.no_grad():
        s = model.stem(dummy)
        s1 = model.stage1(s)
        s2 = model.stage2(s1)
        s3 = model.stage3(s2)
    print(f"\nFeature map sizes:")
    print(f"  After stem   : {tuple(s.shape)}")
    print(f"  After stage1 : {tuple(s1.shape)}")
    print(f"  After stage2 : {tuple(s2.shape)}")
    print(f"  After stage3 : {tuple(s3.shape)}")