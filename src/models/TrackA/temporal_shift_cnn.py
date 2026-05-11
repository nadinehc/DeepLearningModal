import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock

class TemporalShift(nn.Module):
    def __init__(self, n_segment=4, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        # x: (B*T, C, H, W)
        nt, c, h, w = x.size()
        t = self.n_segment
        b = nt // t

        x = x.view(b, t, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)

        # shift left
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # shift right
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        # not shift
        out[:, :, 2*fold:] = x[:, :, 2*fold:]

        return out.view(nt, c, h, w)

class TSMBasicBlock(BasicBlock):
    def __init__(self, *args, n_segment=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.tsm = TemporalShift(n_segment=n_segment)

    def forward(self, x):
        identity = x

        out = self.tsm(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(self.tsm(x))

        out += identity
        out = self.relu(out)

        return out

class TSMResNet(nn.Module):
    def __init__(self, num_classes=33, n_segment=4):
        super().__init__()

        class ConfiguredTSMBlock(TSMBasicBlock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, n_segment=n_segment, **kwargs)

        # PASS THE CLASS, NOT A LAMBDA
        self.model = models.ResNet(
            block=ConfiguredTSMBlock, 
            layers=[2, 2, 2, 2]
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.temporal_fc = nn.Linear(num_classes, 1)
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        logits = self.model(x)

        logits = self.model(x)
    
        logits = logits.view(b, t, -1)

        # forward
        weights = self.temporal_fc(logits)      # [B, T, 1]
        weights = torch.softmax(weights, dim=1)
        out = (logits * weights).sum(dim=1)
        
        return out

if __name__ == "__main__":
    # Input: (Batch, Time, Channels, Height, Width)
    vid = torch.rand(2, 4, 3, 224, 224)
    model = TSMResNet(num_classes=33, n_segment=4)
    output = model(vid)
    print(f"Output shape: {output.shape}") # Should be [2, 33]