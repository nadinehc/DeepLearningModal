from transformers import VideoMAEForVideoClassification, VideoMAEConfig
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR, CosineAnnealingLR
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random

class VideoMAE(nn.Module):
    def __init__(self, num_classes=33):
        super().__init__()
        config  = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-ssv2")
        config.num_frames = 4
        config.num_labels = num_classes
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-ssv2",
            config=config,
            ignore_mismatched_sizes=True,   # replaces the 174-class head
            cache_dir="/Data/nadine.hage-chehade"
        )

    def forward(self, pixel_values):
        return self.model(pixel_values).logits
    
def set_phase(model, phase: int):
    if phase == 1:
        for name, p in model.named_parameters():
            p.requires_grad = ('classifier' in name)   # head only
    elif phase == 2:
        for p in model.parameters():
            p.requires_grad = True                     # unfreeze all
    
def build_mae_optimizer(model, cfg, phase: int):
    lr = cfg.training.lr
    if phase == 1:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=0.05,
        )

        constant  = ConstantLR(optimizer, factor=1.0, total_iters=cfg.training.phase1_epochs)

        return optimizer, constant
    else:  # phase 2 — layer-wise LR
        optimizer = torch.optim.AdamW([
            {'params': model.model.classifier.parameters(), 'lr': lr/10},
            {'params': model.model.videomae.parameters(),   'lr': lr/100},  # 10x lower
        ], weight_decay=0.05)

        warmup  = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=3)
        cosine  = CosineAnnealingLR(optimizer, T_max= - cfg.training.epochs, eta_min=1e-7)

        scheduler_phase2 = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[3]
        )

        return optimizer, scheduler_phase2

VIDEOMAE_MEAN = [0.5, 0.5, 0.5]
VIDEOMAE_STD  = [0.5, 0.5, 0.5]

class VideoMAETrainTransform:
    """
    Applies consistent spatial transforms across all T frames.
    Input : list of T PIL images (or tensors)
    Output: tensor (T, C, H, W) ready for VideoMAE
    """
    def __init__(self, img_size=224, scale=(0.5, 1.0)):
        self.img_size = img_size
        self.scale    = scale
        self.normalize = T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD)

    def __call__(self, frames, label):
        # frames: list of T PIL images

        # 1. Consistent random resized crop across all frames
        i, j, h, w = T.RandomResizedCrop.get_params(
            frames[0],
            scale=self.scale,      # SSv2: tighter crop than Kinetics
            ratio=[0.75, 1.33],
        )
        frames = [F.resized_crop(f, i, j, h, w, [self.img_size, self.img_size]) for f in frames]

        # 2. Consistent random horizontal flip
        #    ⚠️ SSv2 warning: flip also requires flipping directional labels
        #    Only use if you handle label flipping in your dataset
        if random.random() < 0.5:
            frames = [F.hflip(f) for f in frames]

        # 3. Color jitter — applied per-frame independently (intentional)
        jitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        frames = [jitter(f) for f in frames]

        # 4. To tensor + normalize
        frames = [F.to_tensor(f) for f in frames]           # (C, H, W), range [0,1]
        frames = [self.normalize(f) for f in frames]

        return torch.stack(frames)  # (T, C, H, W)


class VideoMAEValTransform:
    """Deterministic — center crop only, no flipping."""
    def __init__(self, img_size=224):
        self.transform = T.Compose([
            T.Resize(256),                          # resize short side to 256
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
        ])

    def __call__(self, frames, label):
        frames = [self.transform(f) for f in frames]
        return torch.stack(frames)  # (T, C, H, W)d
    
def build_mae_transforms():
    return VideoMAETrainTransform(), VideoMAEValTransform()

if __name__ == "__main__":
    vid = torch.rand(4, 4, 3, 224, 224)  # (batch_size, num_frames, channels, height, width
    model = VideoMAE(num_classes=33)
    out = model(vid)
    print(out.shape)  # Should be (4, 33)

    opt, _ = build_mae_optimizer(model, cfg=type('cfg', (), {'training': type('training', (), {'lr': 1e-4, 'weight_decay': 0.05, 'phase1_epochs': 10, 'epochs': 20})()})(), phase=2)