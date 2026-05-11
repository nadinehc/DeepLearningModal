"""
VideoFrameDataset: loads a fixed number of RGB frames per video folder.

Expected layout under root_dir::

    root_dir/
      000_SomeClassName/
        video_12345/
          frame_000.jpg
          frame_001.jpg
          ...
      001_AnotherClass/
        ...

Class index is parsed from the leading number in the class folder name (000, 001, ...).
Each __getitem__ returns:
    video_tensor: float tensor of shape (T, C, H, W)
    label: int64 scalar class index
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from torch.utils.data import DataLoader


def _list_frame_paths(video_dir: Path) -> List[Path]:
    """All image files in a video folder, sorted by name."""
    paths: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(sorted(video_dir.glob(extension)))
    return sorted(paths, key=lambda p: p.name)


def _parse_class_index(class_dir_name: str) -> Optional[int]:
    """
    Expect folder names like '017_Class_name'. Returns 17, or None if no prefix.
    """
    match = re.match(r"^(\d+)_", class_dir_name)
    if match is None:
        return None
    return int(match.group(1))


def collect_video_samples(root_dir: Path) -> List[Tuple[Path, int]]:
    """
    Walk root_dir: each class folder contains video subfolders with frames.

    Returns list of (video_folder_path, class_index).
    """
    root_dir = root_dir.resolve()
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    samples: List[Tuple[Path, int]] = []
    class_dirs = [p for p in sorted(root_dir.iterdir()) if p.is_dir()]

    # If folders lack numeric prefix, assign indices by sorted order (0..C-1).
    fallback_index = {p.name: i for i, p in enumerate(class_dirs)}

    for class_dir in class_dirs:
        parsed = _parse_class_index(class_dir.name)
        class_index = parsed if parsed is not None else fallback_index[class_dir.name]

        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            frame_paths = _list_frame_paths(video_dir)
            if len(frame_paths) == 0:
                continue
            samples.append((video_dir, class_index))

    if len(samples) == 0:
        raise RuntimeError(f"No video folders with frames under {root_dir}")

    return samples


def _pick_frame_indices(num_available: int, num_frames: int) -> List[int]:
    """
    Evenly spaced indices in [0, num_available - 1], inclusive.
    If fewer frames than requested, indices may repeat (last frame duplicated).
    """
    if num_available <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    if num_available == 1:
        return [0] * num_frames

    # linspace in index space
    positions = torch.linspace(0, num_available - 1, steps=num_frames)
    indices = [int(round(float(x))) for x in positions]
    return indices


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        num_frames: int,
        transform: Callable,
        sample_list: Optional[List[Tuple[Path, int]]] = None,
    ) -> None:
        """
        Args:
            root_dir: Split root (contains class folders).
            num_frames: T in the returned tensor (T, C, H, W).
            transform: Applied independently to each PIL image (typically Resize + ToTensor + Normalize).
            sample_list: Optional pre-built list of (video_dir, label). Use for train/val splits.
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform

        if sample_list is None:
            self.samples = collect_video_samples(self.root_dir)
        else:
            self.samples = list(sample_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir, label = self.samples[index]
        frame_paths = _list_frame_paths(video_dir)
        indices = _pick_frame_indices(len(frame_paths), self.num_frames)

        frames = []
        for frame_index in indices:
            path = frame_paths[frame_index]
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
                frames.append(rgb_image)
            # transform: PIL -> (C, H, W)
        tensor_chw = self.transform(frames, label)

        # Stack time dimension: (T, C, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return tensor_chw, label_tensor

class VideoTransform:
    """Applies consistent spatial transforms across all frames of a clip."""

    def __init__(self, train=True, size=224):
        self.train = train
        self.size = size
        self.normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __call__(self, frames, label):
        # frames: List[PIL.Image] or Tensor [T, C, H, W]

        if self.train:
            # --- Sample random params ONCE, apply to all frames ---
            i, j, h, w = T.RandomResizedCrop.get_params(
                frames[0], scale=[0.6, 1.0], ratio=[0.75, 1.33]
            )
            # 18 and 19 are for classes pulling from left to right or the opposite
            do_flip = label != 18 and label != 19 and random.random() > 0.5
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue        = random.uniform(-0.1, 0.1)

            processed = []
            for frame in frames:
                frame = TF.resized_crop(frame, i, j, h, w, [self.size, self.size])
                if do_flip:
                    frame = TF.hflip(frame)
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
                frame = TF.adjust_saturation(frame, saturation)
                frame = TF.adjust_hue(frame, hue)
                frame = TF.to_tensor(frame)
                frame = self.normalize(frame)
                processed.append(frame)

        else:  # val / test
            processed = []
            for frame in frames:
                frame = TF.resize(frame, [224, 224])
                frame = TF.center_crop(frame, [self.size])
                frame = TF.to_tensor(frame)
                frame = self.normalize(frame)
                processed.append(frame)

        return torch.stack(processed)  # List of [C, H, W] tensors

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class VideoTransformStronger:
    def __init__(self, train=True, size=224, erase_prob=0.3):
        self.train = train
        self.size = size
        self.erase_prob = erase_prob
        self.normalize  = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        self.erasing = T.RandomErasing(
            p=erase_prob, scale=(0.02, 0.2),
            ratio=(0.3, 3.0), value=0
        )

    def __call__(self, frames, label):

        if self.train:
            # ── spatial params sampled ONCE, shared across all frames ──
            i, j, h, w = T.RandomResizedCrop.get_params(
                frames[0], scale=[0.4, 1.0], ratio=[0.75, 1.33]  # more aggressive crop
            )

            brightness  = random.uniform(0.6, 1.4)
            contrast    = random.uniform(0.6, 1.4)
            saturation  = random.uniform(0.6, 1.4)
            hue         = random.uniform(-0.2, 0.2)

            # new: random grayscale (same decision for all frames)
            do_gray = random.random() < 0.2

            # new: random rotation (same angle for all frames)
            angle = random.uniform(-15, 15) if random.random() < 0.3 else 0

            # new: gaussian blur (same kernel for all frames)
            do_blur   = random.random() < 0.3
            blur_size = random.choice([3, 5])

            # new: per-frame independent color noise (mild)
            do_per_frame_jitter = random.random() < 0.5

            processed = []
            for frame in frames:
                # spatial (shared)
                frame = TF.resized_crop(frame, i, j, h, w, [self.size, self.size])
                if angle != 0:
                    frame = TF.rotate(frame, angle)

                # color (shared base)
                frame = TF.adjust_brightness(frame, brightness)
                frame = TF.adjust_contrast(frame, contrast)
                frame = TF.adjust_saturation(frame, saturation)
                frame = TF.adjust_hue(frame, hue)

                # per-frame independent mild jitter (simulates exposure variation)
                if do_per_frame_jitter:
                    frame = TF.adjust_brightness(frame, random.uniform(0.9, 1.1))
                    frame = TF.adjust_contrast(frame,   random.uniform(0.9, 1.1))

                if do_gray:
                    frame = TF.rgb_to_grayscale(frame, num_output_channels=3)
                if do_blur:
                    frame = TF.gaussian_blur(frame, kernel_size=[blur_size, blur_size])

                frame = TF.to_tensor(frame)
                frame = self.normalize(frame)

                # random erasing applied per-frame independently (after to_tensor)
                frame = self.erasing(frame)

                processed.append(frame)

            frames = torch.stack(processed)  # [T, C, H, W]

            # new: temporal dropout — replace a random frame with adjacent one
            if random.random() < 0.2:
                t = random.randint(0, len(processed) - 2)
                frames[t] = frames[t + 1].clone()

            return frames

        else:  # val / test
            processed = []
            for frame in frames:
                frame = TF.resize(frame, [self.size, self.size])
                frame = TF.to_tensor(frame)
                frame = self.normalize(frame)
                processed.append(frame)
            return torch.stack(processed)

class TwoStreamTransform(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __call__(self, frames, label):
        # ── Sample augmentation params ONCE for both streams ─────────────────
        # This is the key: both RGB and diff see the exact same crop/flip
        if self.train:
            i, j, h, w = T.RandomResizedCrop.get_params(
                frames[0], scale=[0.6, 1.0], ratio=[0.75, 1.33]
            )
            do_flip    = label != 18 and label != 19 and random.random() > 0.5
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue        = random.uniform(-0.1, 0.1)
        
        normalize_rgb  = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        normalize_diff = T.Normalize([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])  # diff is centered at 0

        rgb_frames  = []
        diff_frames = []

        for t, frame in enumerate(frames):
            # ── RGB stream ───────────────────────────────────────────────────
            if self.train:
                f = TF.resized_crop(frame, i, j, h, w, [224, 224])
                if do_flip: f = TF.hflip(f)
                f = TF.adjust_brightness(f, brightness)
                f = TF.adjust_contrast(f, contrast)
                f = TF.adjust_saturation(f, saturation)
                f = TF.adjust_hue(f, hue)
            else:
                f = TF.resize(frame, [256])
                f = TF.center_crop(f, [224])
            rgb_frames.append(normalize_rgb(TF.to_tensor(f)))

            # ── Difference stream ────────────────────────────────────────────
            # Subtract on raw PIL BEFORE any normalization
            if t == 0:
                prev = frame  # first frame: diff with itself → zero frame
            else:
                prev = frames[t - 1]

            # Convert to float32 tensors for subtraction
            curr_t = TF.to_tensor(frame).float()   # [C, H, W] in [0, 1]
            prev_t = TF.to_tensor(prev).float()    # [C, H, W] in [0, 1]
            diff   = curr_t - prev_t               # [-1, 1] naturally

            # Apply the SAME spatial transform (crop + flip) — NO color jitter on diff
            diff_pil = TF.to_pil_image((diff + 1) / 2)  # back to PIL for spatial ops
            if self.train:
                diff_pil = TF.resized_crop(diff_pil, i, j, h, w, [224, 224])
                if do_flip: diff_pil = TF.hflip(diff_pil)
            else:
                diff_pil = TF.resize(diff_pil, [256])
                diff_pil = TF.center_crop(diff_pil, [224])

            diff_t = TF.to_tensor(diff_pil) * 2 - 1   # back to [-1, 1]
            diff_frames.append(normalize_diff(diff_t))

        rgb  = torch.stack(rgb_frames,  dim=0)   # [T, 3, 224, 224]
        diff = torch.stack(diff_frames, dim=0)   # [T, 3, 224, 224]

        return rgb, diff

if __name__ == '__main__':
    video = torch.rand(4, 3, 224, 224)

    train_transform = VideoTransformStronger(train=False)
    # train_transform_two_streams = TwoStreamTransform()

    train_dataset = VideoFrameDataset(
        root_dir="/Data/nadine.hage-chehade/train",
        num_frames=4,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    frames, target = next(iter(train_loader))

    print(frames[0].shape, frames[1].shape)
    print(target)