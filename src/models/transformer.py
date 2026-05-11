import torch
import torch.nn as nn
import math

class CNNTokenizer(nn.Module):
    """
    Replace raw patch embedding with shallow CNN.
    Gives transformer spatially-aware tokens without full ResNet depth.
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,  64, kernel_size=7, stride=2, padding=3),  # 112×112
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 56×56
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=4, padding=1), # 28×28
            nn.BatchNorm2d(embed_dim), nn.ReLU(),
        )
        # 14×14 = 192 tokens — still manageable

    def forward(self, x):
        # x: [B, 3, 224, 224]
        x = self.cnn(x)                       # [B, embed_dim, 28, 28]
        x = x.flatten(2).transpose(1, 2)      # [B, 192, embed_dim]
        return x

class DividedSTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        mlp_dim = int(embed_dim * mlp_ratio)

        # ── Spatial attention ──────────────────────────────
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # ── Temporal attention ─────────────────────────────
        self.norm_temporal = nn.LayerNorm(embed_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # ── MLP ────────────────────────────────────────────
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: [B, T, P, D]
        """
        B, T, P, D = x.shape
         # ── Temporal attention (per patch) ─────────────────
        residual = x
        x = self.norm_temporal(x)
        x = x.permute(0, 2, 1, 3)          # [B, P, T, D]
        x = x.reshape(B * P, T, D)
        x, _ = self.temporal_attn(x, x, x)
        x = x.view(B, P, T, D).permute(0, 2, 1, 3)
        x = residual + x

        # after temporal attention
        x = x * (x[:, 1:] - x[:, :-1]).abs().mean(dim=1, keepdim=True)

        # ── Spatial attention (per frame) ──────────────────
        residual = x
        x = self.norm_spatial(x)
        x = x.view(B * T, P, D)
        x, _ = self.spatial_attn(x, x, x)
        x = x.view(B, T, P, D)
        x = residual + x

        # ── MLP ────────────────────────────────────────────
        residual = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = residual + x

        return x

class SingleBlockVideoTransformer(nn.Module):
    def __init__(
        self,
        num_classes=33,
        num_frames=4,
        img_size=224,
        patch_size=16,
        embed_dim=384,   # small — ViT-Tiny uses 192, ViT-Small uses 384
        num_heads=6,     # embed_dim must be divisible by num_heads
        mlp_ratio=4.0,   # MLP hidden dim = embed_dim * mlp_ratio
        dropout=0.1,
    ):
        super().__init__()
        self.num_frames  = num_frames
        self.embed_dim   = embed_dim
        num_patches      = 784
        total_tokens     = num_frames * num_patches   # e.g. 8 * 196 = 1568

        # ── Patch embedding (shared across frames) ────────────────────────────
        self.patch_embed = CNNTokenizer(embed_dim)

        # ── Positional embeddings ─────────────────────────────────────────────
        # Spatial: one per patch position (shared across frames)
        self.spatial_pos_embed  = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        # Temporal: one per frame (added to all patches of that frame)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )

        # CLS token — aggregates global clip representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_drop = nn.Dropout(dropout)

        # ── Single transformer encoder block ──────────────────────────────────
        mlp_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # ── Final norm + classifier ───────────────────────────────────────────
        self.norm       = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Transformers are sensitive to init — these are the standard ViT inits
        nn.init.trunc_normal_(self.cls_token,           std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed,   std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.classifier.weight,   std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        # ── Patch embed each frame ────────────────────────────────────────────
        x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)                   # [B*T, num_patches, embed_dim]
        x = x.contiguous()
        x = x.view(B, T, -1, self.embed_dim)      # [B, T, num_patches, embed_dim]

        # ── Add spatial positional embedding ──────────────────────────────────
        x = x + self.spatial_pos_embed.unsqueeze(1)   # broadcast over T

        # ── Add temporal positional embedding ─────────────────────────────────
        x = x + self.temporal_pos_embed.unsqueeze(2)  # broadcast over num_patches

        # ── Flatten to sequence of tokens ─────────────────────────────────────
        x = x.view(B, T * x.shape[2], self.embed_dim) # [B, T*num_patches, embed_dim]

        # ── Prepend CLS token ─────────────────────────────────────────────────
        cls = self.cls_token.expand(B, -1, -1)        # [B, 1, embed_dim]
        x   = torch.cat([cls, x], dim=1)              # [B, 1 + T*num_patches, embed_dim]
        x   = self.pos_drop(x)

        # ── Single transformer block ──────────────────────────────────────────
        # Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x= residual + x

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        # ── Classify from CLS token ───────────────────────────────────────────
        x = self.norm(x)
        cls_out = x[:, 0]                             # [B, embed_dim]
        return self.classifier(cls_out)               # [B, num_classes]

class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):

        nt, c, h, w = x.size()
        t = self.n_segment
        b = nt // t

        x = x.view(b, t, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # shift left
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # shift right
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        # not shift
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        out = out.view(nt, c, h, w)

        return out

class DividedSpaceTimeVideoTransformer(nn.Module):
    def __init__(
        self,
        num_classes=33,
        num_frames=4,
        embed_dim=384,
        num_heads=6,
        depth=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.embed_dim  = embed_dim

        self.TSM = TemporalShift(n_segment=num_frames)

        # ── Tokenizer (same as yours) ──────────────────────
        self.patch_embed = CNNTokenizer(embed_dim)
        num_patches = 196   # from CNNTokenizer (14×14)

        # ── Positional embeddings ──────────────────────────
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, num_patches, embed_dim)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, 1, embed_dim)
        )

        self.pos_drop = nn.Dropout(dropout)

        # ── Transformer blocks ─────────────────────────────
        self.blocks = nn.ModuleList([
            DividedSTBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # ── Tokenize frames ────────────────────────────────
        x = x.view(B * T, C, H, W)
        x = self.TSM(x)                          
        x = self.patch_embed(x)                 # [B*T, P, D]
        P = x.shape[1]

        x = x.view(B, T, P, self.embed_dim)     # [B, T, P, D]

        # ── Add positional embeddings ──────────────────────
        x = x + self.spatial_pos_embed
        x = x + self.temporal_pos_embed
        x = self.pos_drop(x)

        # ── Divided space–time transformer ─────────────────
        for blk in self.blocks:
            x = blk(x)

        # ── Pooling (NO CLS TOKEN) ─────────────────────────
        x = self.norm(x)
        x = x.mean(dim=2)   # spatial pool → [B, T, D]
        x = x.mean(dim=1)   # temporal pool → [B, D]

        return self.head(x)

"""
def build_vit_optimizer(cfg, model, steps_per_epoch):
    # Don't apply weight decay to positional embeddings, CLS token, biases, LayerNorm
    decay_params    = []
    no_decay_params = []

    num_epochs = int(cfg.training.epochs)

    for name, param in model.named_parameters():
        if any(nd in name for nd in ["pos_embed", "cls_token", "bias", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg.training.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.training.lr, betas=(0.9, 0.999))

    # Warmup is critical for transformers — without it training is unstable
    def lr_lambda(step):
        warmup_steps = 5 * steps_per_epoch   # 5 epoch warmup
        if step < warmup_steps:
            return step / warmup_steps        # linear warmup
        # cosine decay after warmup
        progress = (step - warmup_steps) / (num_epochs * steps_per_epoch - warmup_steps)
        return max(1e-5, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
"""

def build_vit_optimizer(cfg, model, steps_per_epoch):
    decay = []
    no_decay = []

    num_epochs = int(cfg.training.epochs)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            name.endswith("bias")
            or "norm" in name.lower()
            or "pos_embed" in name.lower()
            or "spatial_pos_embed" in name.lower()
            or "temporal_pos_embed" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.lr,
        betas=(0.9, 0.95),
    )

    # ── LR schedule ──────────────────────────────────────
    warmup_steps = 10 * steps_per_epoch
    total_steps  = num_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler

if __name__ == "__main__":
    vid = torch.rand(2, 4, 3, 224, 224)
    model = DividedSpaceTimeVideoTransformer(depth=2)  # smaller model for testing
    pred = model(vid)
    print(pred.shape)