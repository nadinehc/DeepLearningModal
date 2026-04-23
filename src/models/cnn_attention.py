"""
CNN + Masked Multi-Head Self-Attention

Forward (conceptually):
    Input:  (batch, T=4, C, H, W)
    Reshape: (batch * T, C, H, W)
    Backbone: ResNet18 -> (batch * T, 512)
    Reshape: (batch, T, 512)
    Add learned positional encoding: (batch, T, 512)
    Masked Multi-Head Self-Attention: (batch, T, 512)
        └─ causal mask: frame i can only attend to frames 0..i
    Take last frame's token: (batch, 512)
    Linear classifier: (batch, num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class CNNAttention(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        num_frames: int = 4,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── 1. Frame encoder (ResNet18 backbone) ──────────────────────────
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  
        backbone.fc = nn.Identity() # type: ignore
        self.backbone = backbone

        # ── 2. Learned positional encoding ────────────────────────────────
        # Shape: (1, T, feature_dim) — broadcast over the batch dimension.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames, self.feature_dim)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # ── 3. Masked multi-head self-attention ───────────────────────────
        # batch_first=True  →  expects (batch, seq, dim)
        self.proj = nn.Linear(self.feature_dim, 256)
        self.attn = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(256)

        # ── 4. Feed-forward sub-layer (standard Transformer block) ────────
        self.ffn = nn.Sequential(
            nn.Linear(256, self.feature_dim * 4),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(self.feature_dim * 4, 256),
            nn.Dropout(attn_dropout),
        )
        self.ffn_norm = nn.LayerNorm(256)

        # ── 5. Classifier head ────────────────────────────────────────────
        self.classifier = nn.Linear(256, num_classes)

        # Register causal mask as a buffer so it moves with .to(device) calls.
        # True entries are *ignored* by nn.MultiheadAttention.
        causal_mask = torch.triu(
            torch.ones(num_frames, num_frames, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        self._init_weights()

    # ── helpers ───────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_frames(self, video_batch: torch.Tensor) -> torch.Tensor:
        """CNN encoder: (B, T, C, H, W) → (B, T, D)."""
        B, T, C, H, W = video_batch.shape
        frames = video_batch.reshape(B * T, C, H, W)          # (B*T, C, H, W)
        features = self.backbone(frames)                        # (B*T, D)
        features = torch.flatten(features, start_dim=1)
        return features.view(B, T, self.feature_dim)           # (B, T, D)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, video_batch: torch.Tensor) -> torch.Tensor:
        """
        video_batch : (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        # 1. Per-frame CNN features
        x = self._encode_frames(video_batch)          # (B, T, D)

        # 2. Add positional encoding
        x = x + self.pos_embedding                    # (B, T, D)

        x = self.proj(x)                              # (B, T, 256)

        # 3. Masked multi-head self-attention (pre-norm style)
        x_norm = self.attn_norm(x)         
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=self.causal_mask,               # (T, T) causal mask
            need_weights=False,
        )
        x = x + attn_out                              # residual

        # 4. Feed-forward sub-layer
        x = x + self.ffn(self.ffn_norm(x))            # residual

        # 5. Use the *last* frame token as the sequence summary
        last_token = x[:, -1, :]                      # (B, D)

        return self.classifier(last_token)             # (B, num_classes)