"""
models.py
----------
Defines CNN architectures for the Face Mask Classification project.
Includes custom CNNs and pretrained baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# -------------------------
# Custom CNN Architectures
# -------------------------

class NetVer1(nn.Module):
    """3-layer CNN with pooling at each stage."""
    def __init__(self):
        super(NetVer1, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(28 * 28 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)


class NetVer2(nn.Module):
    """2-layer CNN with pooling, smaller fully connected layers."""
    def __init__(self):
        super(NetVer2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(56 * 56 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)


class NetVer3(nn.Module):
    """2-layer CNN without pooling (large FC layer)."""
    def __init__(self):
        super(NetVer3, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(16)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(224 * 224 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return F.log_softmax(x, dim=1)


# -------------------------
# Additional Architectures
# -------------------------

class SimpleCNN(nn.Module):
    """Minimal CNN baseline for quick tests."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(112 * 112 * 8, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class ResNet18(nn.Module):
    """ResNet18 pretrained on ImageNet, fine-tuned for 5 classes."""
    def __init__(self, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)


class MobileNetV2(nn.Module):
    """MobileNetV2 pretrained, fine-tuned for 5 classes."""
    def __init__(self, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 5)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)


class CustomDeepCNN(nn.Module):
    """Deeper CNN with more conv layers and pooling."""
    def __init__(self):
        super(CustomDeepCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(28 * 28 * 128, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)
