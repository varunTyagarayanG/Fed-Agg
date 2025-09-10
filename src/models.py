import torch.nn as nn

class CNN_Cifar(nn.Module):
    """
    Improved CNN for CIFAR-10 classification.
    Inspired by VGG-like architectures but lightweight for federated learning.
    """
    def _init_(self, num_classes=10):
        super()._init_()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 64x16x16

            nn.Conv2d(64, 128, 3, padding=1), # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),# 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),               # 128x8x8

            nn.Conv2d(128, 256, 3, padding=1),# 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                # 256x4x4
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fully_connected(x)
        return x


class CNN_Mnist(nn.Module):
    """
    This CNN is inspired by LeNet-5. It differs from Lenet-5 in few things such as 
    using ReLU instead of Sigmoid and using MaxPooling instead of AveragePooling.
    This architecture is only suitable for MNIST dataset.
    """
    def _init_(self, num_classes=10):
        super()._init_()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = nn.Flatten()(x)
        x = self.fully_connected(x)
        return x