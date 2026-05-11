import torch.nn as nn
import torch
from transformers import TimesformerForVideoClassification
from torchvision.transforms import v2

class Timesformer(nn.Module):
    def __init__(self, num_classes=33, model_name="facebook/timesformer-base-finetuned-k400"):
        super().__init__()
        
        # Load the base model with the pretrained weights
        self.model = TimesformerForVideoClassification.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )
        
        # Access the configuration to get the hidden size
        hidden_size = self.model.config.hidden_size
        
        # Replace the classifier head
        self.model.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        return self.model(pixel_values).logits

def build_timesformer_optimizer(model, cfg, total_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5*cfg.training.lr,               # The peak learning rate
        total_steps=total_steps,
        pct_start=0.1,             # Percentage of total steps for warmup (0.1 = 10%)
        anneal_strategy='cos',      # This makes it a Cosine Decay
        div_factor=25.0,           # Initial LR = max_lr / div_factor
        final_div_factor=10000.0   # Min LR = max_lr / final_div_factor
    )

    return optimizer, scheduler

class VideoOnlyTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, frames, label):
        if isinstance(frames, list):
            frames = torch.stack([v2.functional.to_image(f) for f in frames])
        out = self.base_transform(frames)
        if isinstance(out, (list, tuple)):
            return torch.stack(out)
        return out

def timesformer_transforms():
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256),
        v2.RandomCrop(224),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.Normalize(                            # ImageNet/Kinetics-400 defaults
            mean=[0.45, 0.45, 0.45], 
            std=[0.225, 0.225, 0.225]
        )
    ])

    # 3. For Validation, just use the built-in one directly
    val_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.Normalize(                            # ImageNet/Kinetics-400 defaults
            mean=[0.45, 0.45, 0.45], 
            std=[0.225, 0.225, 0.225]
        )
    ])

    return VideoOnlyTransform(train_transform), VideoOnlyTransform(val_transform)
    
if __name__ == "__main__":
    model = Timesformer()
    vid = torch.rand(4, 4, 3, 224, 224)
    print(model(vid).shape)