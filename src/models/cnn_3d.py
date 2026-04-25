import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3,
            stride=(1, stride, stride),
            padding=1,
            bias=True
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class CNN3D(nn.Module):
    def __init__(self, num_classes=33):
        super().__init__()

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv3d(
                3, 32,
                kernel_size=(3, 3, 3),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # ---- Stages ----
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)

        # ---- Head ----
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, num_classes)

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock3D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                nn.init.constant_(m.bias, 0)

        # Zero-init last BN in each residual block
        for m in self.modules():
            if isinstance(m, BasicBlock3D):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        # Rearrange for 3D Conv: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)        
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)