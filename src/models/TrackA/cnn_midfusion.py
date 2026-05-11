import torch
import torch.nn as nn
from torchvision import models

class TemporalAttentionFusion(nn.Module):
    def __init__(self, channels: int, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        self.channels   = channels

        # global context → per-frame scalar weights
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                        # [B, T*ch, 1, 1]
            nn.Flatten(),                                   # [B, T*ch]
            nn.Linear(channels * num_frames, num_frames),
            nn.Softmax(dim=-1),                             # [B, T]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T*ch, h, w]
        T, ch, h, w = self.num_frames, self.channels, x.shape[2], x.shape[3]
        B = x.shape[0]

        attn_w  = self.attn(x)                              # [B, T]
        x_frames = x.view(B, T, ch, h, w)                  # [B, T, ch, h, w]
        attn_w   = attn_w.view(B, T, 1, 1, 1)
        x_weighted = (x_frames * attn_w).view(B, T * ch, h, w)

        return self.proj(x_weighted) 

class MidFusionResNet18(nn.Module):
    def __init__(self, num_classes=33, num_frames=4, fusion_point=2, dropout=0.3, diff_weight=0.5):
        """
        fusion_point: where to concatenate frame features.
            1 → fuse at 64ch  56×56 
            2 → fuse at 128ch 28×28
            3 → fuse at 256ch 14×14
        """
        super().__init__()
        self.num_frames   = num_frames
        self.fusion_point = fusion_point
        self.diff_weight = diff_weight

        backbone = models.resnet18(weights=None)

        self.per_frame_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,      # always included
        )

        self.mid_layers = nn.Sequential()
        if fusion_point in [2, 3]:
            self.mid_layers.add_module("layer2", backbone.layer2)
        if fusion_point == 3:
            self.mid_layers.add_module("layer3", backbone.layer3)

        channels_at_fusion = {1: 64, 2: 128, 3: 256}[fusion_point]

        self.spatial_drop = nn.Dropout2d(p=0.1)

        self.fusion = TemporalAttentionFusion(channels_at_fusion, num_frames)

        self.post_fusion_layers = nn.Sequential()
        if fusion_point == 1:
            self.post_fusion_layers.add_module("layer2", backbone.layer2)
            self.post_fusion_layers.add_module("layer3", backbone.layer3)
            self.post_fusion_layers.add_module("layer4", backbone.layer4)
        elif fusion_point == 2:
            self.post_fusion_layers.add_module("layer3", backbone.layer3)
            self.post_fusion_layers.add_module("layer4", backbone.layer4)
        elif fusion_point == 3:
            self.post_fusion_layers.add_module("layer4", backbone.layer4)

        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _inject_diff(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C, H, W] → same shape, diffs added in-place"""
        diffs = x[:, 1:] - x[:, :-1]                               # [B, T-1, C, H, W]
        diffs = torch.cat([diffs, torch.zeros_like(x[:, :1])], dim=1)  # pad last frame
        return x + self.diff_weight * diffs

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape

        x = self._inject_diff(x)

        x = x.view(B * T, C, H, W)            # [B*T, C, H, W]
        x = self.per_frame_layers(x)          # [B*T, 64,  56, 56]
        x = self.mid_layers(x)                # [B*T, ch,  h,  w]  depends on fusion_point
        x = self.spatial_drop(x)  

        _, ch, h, w = x.shape
        x = x.view(B, T * ch, h, w)           # [B, T*ch, h, w]

        x = self.fusion(x)                    # [B, ch, h, w]

        x = self.post_fusion_layers(x)        # [B, 512, 7, 7]
        x = self.avgpool(x)                   # [B, 512, 1, 1]
        x = torch.flatten(x, 1)               # [B, 512]
        return self.classifier(x)   

def build_mid_fusion_optimizer(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    return optimizer, scheduler

if __name__ == "__main__":
    vid = torch.rand(8, 4, 3, 224, 224)
    model = MidFusionResNet18()
    logits = model(vid)
    print(logits.shape)         