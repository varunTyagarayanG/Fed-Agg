import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Cifar(nn.Module):
    """
    Improved 2-layer CNN for CIFAR-10.
    Wider filters + BatchNorm + Global Average Pooling.
    Lightweight yet more accurate for FL.
    """
    def __init__(self, num_classes=10):
        super(CNN_Cifar, self).__init__()
        
        # Conv Block 1: 3x32x32 -> 64x16x16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv Block 2: 64x16x16 -> 128x8x8
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)   # -> 64x16x16

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)   # -> 128x8x8

        # Global Average Pooling
        x = self.gap(x)          # -> 128x1x1
        x = x.view(-1, 128)

        # Classifier
        x = self.fc1(x)
        return x


class CNN_Mnist(nn.Module):
    """
    Simple CNN for MNIST dataset (LeNet-like).
    Lightweight for federated learning.
    """
    def __init__(self, num_classes=10):
        super(CNN_Mnist, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                       
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),                                          
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fully_connected(x)
        return x
