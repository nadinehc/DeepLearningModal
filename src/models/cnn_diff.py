import torch
import torch.nn as nn
from torchvision import models

class TwoStreamResNet18(nn.Module):
    def __init__(self, num_classes=33, num_frames=4, fusion_weight=0.7):
        """
        fusion_weight: how much to weight the diff stream.
            0.5 = equal weighting (good start)
            0.7 = trust diff stream more (good for SSv2 — motion is the signal)
        """
        super().__init__()
        self.num_frames    = num_frames
        self.fusion_weight = fusion_weight  # α in: logits = (1-α)*rgb + α*diff

        def make_backbone():
            net = models.resnet18(weights=None)
            net.fc = nn.Identity()          # type: ignore
            return net

        self.rgb_backbone  = make_backbone()
        self.diff_backbone = make_backbone()  # separate weights — learns motion features

        self.rgb_classifier  = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
        self.diff_classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward_stream(self, x, backbone, classifier):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)       # [B*T, C, H, W]
        f = backbone(x)                  # [B*T, 512]
        f = f.view(B, T, -1).mean(dim=1) # [B, 512] average pooling
        return classifier(f)             # [B, num_classes]

    def forward(self, rgb_diff_pair):
        rgb = rgb_diff_pair[0]
        diff = rgb_diff_pair[1]
        rgb_logits  = self.forward_stream(rgb,  self.rgb_backbone,  self.rgb_classifier)
        diff_logits = self.forward_stream(diff, self.diff_backbone, self.diff_classifier)

        # Weighted average of logits — simplest fusion, works well
        return (1 - self.fusion_weight) * rgb_logits + self.fusion_weight * diff_logits