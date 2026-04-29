import torch.nn as nn
import torch.nn.functional as F

from models.ghostnet import GhostBottleneck


class GhostNetV2_Base(nn.Module):
    def __init__(self, num_classes=7, dropout=0.3, in_channels = 1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.stage1 = GhostBottleneck(16,  48,  24, stride=2)
        self.stage2 = GhostBottleneck(24,  72,  40, stride=2)
        self.stage3 = GhostBottleneck(40, 120,  80, stride=2)

        # Идентично TT-версии
        self.blocks = nn.Sequential(
            GhostBottleneck(80,  160,  80, stride=1),
            GhostBottleneck(80,  160,  80, stride=1),
            GhostBottleneck(80,  240, 112, stride=1),
            GhostBottleneck(112, 336, 112, stride=1),
            GhostBottleneck(112, 480, 160, stride=1),
        )

        self.head = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # Единственное отличие от TT-версии — обычный Linear вместо TTLinear
        self.fc         = nn.Linear(960, 128)
        self.bn_fc      = nn.BatchNorm1d(128)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)              # [B,  16, 48, 48]
        x = self.stage1(x)            # [B,  24, 24, 24]
        x = self.stage2(x)            # [B,  40, 12, 12]
        x = self.stage3(x)            # [B,  80,  6,  6]
        x = self.blocks(x)            # [B, 160,  6,  6]
        x = self.head(x).flatten(1)   # [B, 960]
        x = self.fc(x)                # [B, 128]
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)        # [B,   7]
        return x