"""
Train a video classifier on folders of frames.

Run from the ``src/`` directory (so ``configs/`` resolves)::

    python train.py
    python train.py experiment=cnn_lstm

Pick an **experiment** under ``configs/experiment/`` (each one selects a model and can
add more overrides). You can still override any key, e.g. ``model.pretrained=false``.

Training uses ``dataset.train_dir`` and ``split_train_val`` for an internal train/val
split; the dedicated ``dataset.val_dir`` is for ``evaluate.py`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
import torch.nn as nn
import math
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset.video_dataset import VideoFrameDataset, collect_video_samples, VideoTransform
from models.cnn_baseline import CNNBaseline
from models.cnn_lstm import CNNLSTM
from models.cnn_attention import CNNAttention
from models.cnn_3d import CNN3D
from models.swin import VideoSwinTransformer
from models.cnn_1dtemp import CNNTemporalConv
from models.cnn_diff import TwoStreamResNet18

from models.TrackA.cnn_midfusion import MidFusionResNet18, build_mid_fusion_optimizer
from models.TrackA.tiny_midfusion import TinyMidFusion, build_tiny_optimizer
from models.TrackA.temporal_shift_cnn import TSMResNet

from models.transformer import SingleBlockVideoTransformer, build_vit_optimizer, DividedSpaceTimeVideoTransformer
from models.TrackB.timesformer import Timesformer, build_timesformer_optimizer, timesformer_transforms
from models.TrackB.x3d import X3D, build_x3d_optimizer
from models.TrackB.swin import Swin, build_swin_optimizer, swin_transforms
from models.TrackB.videoMAE import VideoMAE, set_phase, build_mae_optimizer, build_mae_transforms
from models.TrackB.TeacherStudent import TeacherStudentVideoMAE, distillation_loss, build_optimizer_videomae
from utils import build_transforms, set_seed, split_train_val

import wandb

def build_model(cfg: DictConfig) -> nn.Module:
    """Create the model described by cfg.model.name."""
    name = cfg.model.name
    num_classes = cfg.model.num_classes
    pretrained = cfg.model.pretrained

    # track A
    if name == "Atemporal_shift":
        return TSMResNet(num_classes=num_classes, n_segment=int(cfg.dataset.num_frames))
    if name == "Ax3d":
        return X3D(False)
    if name == "Acnn_midfusion_tiny":
        return TinyMidFusion()
    if name == "Acnn_midfusion":
        return MidFusionResNet18()
    if name == "cnn_baseline":
        return CNNBaseline(num_classes=num_classes, pretrained=False)
    if name == "cnn_lstm":
        hidden = cfg.model.get("lstm_hidden_size", 512)
        return CNNLSTM(
            num_classes=num_classes,
            pretrained=pretrained,
            lstm_hidden_size=int(hidden),
        )
    if name == "cnn_attention":
        return CNNAttention(
            num_classes=num_classes,
            pretrained=pretrained,
            num_frames=int(cfg.dataset.num_frames),
        )
    if name == "cnn_3d":
        return CNN3D(num_classes=num_classes) 
    if name == "cnn_1DT":
        return CNNTemporalConv(33)
    if name == "swin":
        return VideoSwinTransformer(
            num_classes=num_classes,
        )
    if name == "cnn_diff":
        return TwoStreamResNet18(fusion_weight=0.7)
    if name == "one_block_transformer":
        return DividedSpaceTimeVideoTransformer()

    # track B
    if name == "Btimesformer":
        return Timesformer()
    if name == "Bx3d":
        return X3D(True)
    if name == "Bswin":
        return Swin()
    if name == "BvideoMAE":
        return VideoMAE(num_classes=num_classes)
    if name == "Bstudent_teacher_mae":
        return TeacherStudentVideoMAE(num_classes=num_classes)
    raise ValueError(f"Unknown model.name: {name}")

def build_optimizer(model, cfg, steps):
    name = cfg.model.name

    if name == "Acnn_midfusion_tiny" or name == "Ax3d" or name == "Atemporal_shift":
        return build_tiny_optimizer(model, cfg, steps)

    if name == "Acnn_midfusion":
        return build_mid_fusion_optimizer(model, cfg)

    if name == "one_block_transformer":
        return build_vit_optimizer(cfg, model, steps)
    
    if name == "Bswin":
        return build_swin_optimizer(model, cfg)

    if name == "Btimesformer":
        return build_timesformer_optimizer(model, cfg, steps)
    
    if name == "BvideoMAE":
        set_phase(model, phase=1)  # start with head-only training
        return build_mae_optimizer(model, cfg, phase=1)

    if name == "Bstudent_teacher_mae":
        return build_optimizer_videomae(model, cfg)
    
    return build_x3d_optimizer(model, cfg.training.lr, int(cfg.training.epochs))

def train_eval_transforms(cfg):
    name = cfg.model.name

    if name == "Bswin":
        return swin_transforms()
    
    if name == "Btimesformer":
        return timesformer_transforms()
    
    if name == "BvideoMAE" or name == "Bstudent_teacher_mae":
        return build_mae_transforms()
    
    return VideoTransform(), VideoTransform(False)

def mixup_batch(x, y, alpha=0.3):
    """
    x     : [B, T, C, H, W] video clips
    y     : [B] integer labels
    alpha : controls how uniform the blend is
              alpha → 0   : λ close to 0 or 1 (barely mixed)
              alpha = 0.3 : mild mixing      ← good default
              alpha = 1.0 : fully uniform    (very aggressive)
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()

    # random permutation to get pairs
    idx   = torch.randperm(x.size(0), device=x.device)

    x_mix = lam * x + (1 - lam) * x[idx]   # blend inputs
    y_a   = y                                # original labels
    y_b   = y[idx]                           # shuffled labels

    return x_mix, y_a, y_b, lam

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    batch_size: int,
    nb_epoch: int = 0,
) -> Tuple[float, float, float]:
    """Returns (average loss, top-1 accuracy) on the training set for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for video_batch, labels in tqdm(data_loader, desc=f"Epoch {nb_epoch + 1}"):
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(video_batch)  # (B, num_classes)
            loss = loss_fn(logits, labels)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if type(logits) == tuple:
            logits = logits[0]

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        _, top5_indices = torch.topk(logits, 5, 1, True)
        correct += int((predictions == labels).sum().item())
        correct_top5 += (top5_indices == labels.view(-1, 1)).any(dim=1).sum().item()
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    acc_top5 = correct_top5 / max(total, 1)
    return average_loss, accuracy, acc_top5

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Returns (average loss, top-1 accuracy) on the validation loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for video_batch, labels in data_loader:
        # rgb_batch = video_batch[0].to(device)
        # diff_batch = video_batch[1].to(device)
        video_batch = video_batch.to(device)
        labels = labels.to(device)

        logits = model(video_batch)
        loss = loss_fn(logits, labels)

        if type(logits) == tuple:  # teacher-student case
            logits = logits[0]

        running_loss += float(loss.item()) * labels.size(0)
        predictions = logits.argmax(dim=1)
        _, top5_indices = torch.topk(logits, 5, 1, True)
        correct += int((predictions == labels).sum().item())
        correct_top5 += (top5_indices == labels.view(-1, 1)).any(dim=1).sum().item()
        total += labels.size(0)

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    acc_top5 = correct_top5 / max(total, 1)
    return average_loss, accuracy, acc_top5

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run = wandb.init(
        entity="nadinehagechehade-project",
    # Set the wandb project where this run will be logged.
    project="retry",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)   # type: ignore
    )
    print(OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.dataset.seed))

    device_str = cfg.training.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    train_dir = Path(cfg.dataset.train_dir).resolve()
    all_samples = collect_video_samples(train_dir)

    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        all_samples = all_samples[: int(max_samples)]

    train_samples, val_samples = split_train_val(
        all_samples,
        val_ratio=float(cfg.dataset.val_ratio),
        seed=int(cfg.dataset.seed),
    )

    train_frequences = np.bincount([s[1] for s in train_samples])

    train_transform, eval_transform = train_eval_transforms(cfg)

    train_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=train_transform,
        sample_list=train_samples,
    )
    val_dataset = VideoFrameDataset(
        root_dir=train_dir,
        num_frames=int(cfg.dataset.num_frames),
        transform=eval_transform,
        sample_list=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device)
    if cfg.model.name == "Bstudent_teacher_mae":
        loss_fn = lambda logits, labels: distillation_loss(
            logits[0],
            logits[1],
            labels,
            alpha=0.5,
            temperature=4.0
        )
    else:   
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer, scheduler = build_optimizer(model, cfg, len(train_loader))

    best_val_accuracy = 0.0
    checkpoint_path = Path(cfg.training.checkpoint_path).resolve()

    for epoch in range(int(cfg.training.epochs)):
        if "phase1_epochs" in cfg.training and epoch == int(cfg.training.phase1_epochs):
            print("Switching to phase 2: unfreezing all layers and adjusting optimizer.")
            if cfg.model.name == "BvideoMAE":
                set_phase(model, phase=2)
                optimizer, scheduler = build_mae_optimizer(model, cfg, phase=2)

        train_loss, train_acc, train_top5 = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device, int(cfg.training.batch_size), epoch
        )
        val_loss, val_acc, val_top5 = evaluate_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} top5 {train_top5:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} top5 {train_top5:.4f}"
        )

        run.log({"Train Loss": train_loss, 
                 "Val Loss": val_loss, 
                 "Train Accuracy": train_acc, 
                 "Val Accuracy": val_acc,
                 "Train Top 5": train_top5,
                 "Val Top5" : val_top5
                 })

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            payload: Dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.model.name,
                "num_classes": int(cfg.model.num_classes),
                "pretrained": bool(cfg.model.pretrained),
                "num_frames": int(cfg.dataset.num_frames),
                "val_accuracy": val_acc,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            if cfg.model.name == "cnn_lstm":
                payload["lstm_hidden_size"] = int(
                    cfg.model.get("lstm_hidden_size", 512)
                )

            torch.save(payload, checkpoint_path)
            print(
                f"  Saved new best model to {checkpoint_path} (val acc={val_acc:.4f})"
            )

    print(f"Done. Best validation accuracy: {best_val_accuracy:.4f}")
    wandb.finish()

if __name__ == "__main__":
    main()
