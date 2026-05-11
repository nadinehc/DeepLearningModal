import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from torchvision.transforms import v2
from torch.utils.data import DataLoader

class Swin(nn.Module):
    def __init__(self, num_classes=33) -> None:
        super().__init__()

        weights = Swin3D_T_Weights.DEFAULT
        self.model = swin3d_t(weights=weights)
        
        # Replace the head for your 33 classes
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
def build_swin_optimizer(model, cfg):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.training.lr},
            {"params": head_params, "lr": cfg.training.lr*10},
        ],
        weight_decay=0.05
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs
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

def swin_transforms():
    weights = Swin3D_T_Weights.KINETICS400_V1
    w_transforms = weights.transforms()

    # 2. Build your custom Training pipeline using the weight's constants
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=w_transforms.resize_size, interpolation=w_transforms.interpolation),
        v2.RandomCrop(size=(224, 224)), # Use Random instead of Center
        v2.Normalize(mean=w_transforms.mean, std=w_transforms.std),
        # Note: Permute is usually done in the Dataset or via a Lambda here
        v2.Lambda(lambda x: x.squeeze().permute(0, 1, 2, 3) if x.ndim == 5 else x.permute(1, 0, 2, 3))
    ])

    # 3. For Validation, just use the built-in one directly
    val_transform = w_transforms

    return VideoOnlyTransform(train_transform), VideoOnlyTransform(val_transform)

if __name__ == "__main__":
    model = Swin()
    vid = torch.rand(2, 3, 4, 224, 224)
    print(model(vid).shape)

    # train_dataset = VideoFrameDataset(
    #     root_dir="/Data/nadine.hage-chehade/processed_data/train",
    #     num_frames=4,
    #     transform=swin_transforms()[0],
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=False,
    # )

    # frames, target = next(iter(train_loader))

    # print(frames.shape)
    # print(target)