import torch
import torch.nn as nn
import torchvision.models as models


class CNNTemporalConv(nn.Module):
    def __init__(
        self,
        num_classes,
        temporal_kernel_size=3,
        dropout=0.5,
    ):
        super().__init__()

        # ----------------------
        # 2D CNN Backbone
        # ----------------------
        cnn = models.resnet18(weights=None)
        feat_dim = cnn.fc.in_features
        cnn.fc = nn.Identity() # type: ignore

        self.backbone = cnn

        # ----------------------
        # Temporal Conv Head
        # ----------------------
        padding = temporal_kernel_size // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size=temporal_kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # ----------------------
        # Classification Head
        # ----------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        """
        x: Tensor of shape (B, T, 3, H, W)
        """
        B, T, C, H, W = x.shape

        # Merge batch and time
        x = x.view(B * T, C, H, W)

        # Frame-wise feature extraction
        feats = self.backbone(x)  # (B*T, feat_dim)

        # Restore temporal dimension
        feats = feats.view(B, T, -1)  # (B, T, feat_dim)

        # Prepare for Conv1D: (B, C, T)
        feats = feats.permute(0, 2, 1)

        # Temporal modeling
        feats = self.temporal_conv(feats)  # (B, C, T)

        # Temporal pooling
        feats = feats.mean(dim=-1)  # (B, C)

        # Classification
        logits = self.classifier(feats)

        return logits

# dummy = torch.rand(2, 4, 3, 224, 224)
# model = CNNTemporalConv(33)
# print(model(dummy).shape)