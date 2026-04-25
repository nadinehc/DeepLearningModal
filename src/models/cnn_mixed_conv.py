import torch
import torch.nn as nn

class MixedConvolution(nn.Module):
    """
    A simple 3D CNN architecture that combines 2D and 3D convolutions.

    - 3 layers of 3D convolutions 
    - 2 layers of 2D convolutions (applied frame-wise)
    - Global average pooling
    - linear classifier
    """
    def __init__(self, num_classes=32):
        super().__init__()

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        returns logits: (B, num_classes)
        """
        

        