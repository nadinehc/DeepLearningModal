import torch.nn as nn
import torch

# Try a much smaller model
class TinyMidFusion(nn.Module):
    def __init__(self, num_classes=33, num_frames=4):
        super().__init__()
        # much smaller backbone
        self.per_frame = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1), nn.BatchNorm2d(32),  nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),  nn.GELU(),
            nn.Conv2d(64,128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(4),                   # → [B*T, 128, 4, 4]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(128 * num_frames, 128, 1), nn.BatchNorm2d(128), nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.per_frame(x)
        _, ch, h, w = x.shape
        x = x.view(B, T*ch, h, w)
        x = self.fusion(x)
        return self.classifier(x)

def build_tiny_optimizer(model, cfg, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        steps_per_epoch=steps,
        epochs=cfg.training.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    return optimizer, scheduler


    
if __name__ == "__main__":
    vid = torch.rand(2, 4, 3, 224, 224)
    model = TinyMidFusion()
    print(model(vid).shape)