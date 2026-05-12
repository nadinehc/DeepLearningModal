import torch
import torch.nn as nn
from torchvision import models


class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=4):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = n_div

    def forward(self, x):
        # x: (B*T, C, H, W)
        nt, c, h, w = x.size()
        t = self.n_segment
        b = nt // t

        x   = x.view(b, t, c, h, w)
        fold = c // self.fold_div
        out  = torch.zeros_like(x)

        out[:, :-1, :fold]      = x[:, 1:,  :fold]       # shift left
        out[:, 1:,  fold:2*fold] = x[:, :-1, fold:2*fold] # shift right
        out[:, :,   2*fold:]     = x[:, :,   2*fold:]      # no shift

        return out.view(nt, c, h, w)


class TSMWrapper(nn.Module):
    """Wraps any conv layer with a temporal shift applied before it."""
    def __init__(self, conv, n_segment=4, n_div=4):
        super().__init__()
        self.tsm  = TemporalShift(n_segment=n_segment, n_div=n_div)
        self.conv = conv

    def forward(self, x):
        return self.conv(self.tsm(x))


def inject_tsm_into_efficientnet(model, n_segment=4, n_div=4, inject_every_n=2):
    """
    Replace the first depthwise conv in every Nth MBConv block with a TSM-wrapped version.
    EfficientNet MBConv structure: expand_conv → depthwise_conv → project_conv
    We inject TSM before the depthwise conv (most effective position).
    """
    block_idx = 0
    for module in model.features.modules():
        # MBConv blocks in torchvision EfficientNet contain a 'block' Sequential
        classname = module.__class__.__name__
        if classname == "MBConv" or classname == "FusedMBConv":
            if block_idx % inject_every_n == 0:
                # find the depthwise conv inside the block
                for name, child in module.block.named_children():
                    # depthwise conv is the one with groups == in_channels
                    if isinstance(child, nn.Sequential):
                        for subname, subchild in child.named_children():
                            if (isinstance(subchild, nn.Conv2d) and
                                    subchild.groups == subchild.in_channels):
                                wrapped = TSMWrapper(subchild, n_segment, n_div)
                                setattr(child, subname, wrapped)
                                break
            block_idx += 1
    return model


class TSMEfficientNet(nn.Module):
    def __init__(self, num_classes=33, n_segment=4, n_div=4, inject_every_n=2):
        super().__init__()
        self.n_segment = n_segment

        # build backbone
        backbone = models.efficientnet_b1(weights=None)

        # inject TSM into depthwise convs
        self.backbone = inject_tsm_into_efficientnet(
            backbone, n_segment=n_segment, n_div=n_div, inject_every_n=inject_every_n
        )

        # replace classifier
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)   # (B*T, C, H, W) — TSM operates on this
        x = self.backbone(x)          # (B*T, num_classes)
        x = x.view(b, t, -1)         # (B, T, num_classes)
        return x.mean(dim=1)          # (B, num_classes)


if __name__ == "__main__":
    model  = TSMEfficientNet(num_classes=33, n_segment=4)
    vid    = torch.rand(2, 4, 3, 224, 224)
    output = model(vid)
    print(f"Output shape: {output.shape}")  # [2, 33]

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n_params:.1f}M")