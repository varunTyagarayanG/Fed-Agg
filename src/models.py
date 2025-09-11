import torch
import torch.nn as nn

class CNN_Cifar(nn.Module):
    """
    LeNet-5 inspired CNN for CIFAR dataset. 
    Uses ReLU activations, BatchNorm, and MaxPooling.
    Suitable for 32x32 RGB images.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2), 
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2,2), 

            nn.Conv2d(6,16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fully_connected = nn.Sequential(
            nn.Flatten(),                                        
            nn.Linear(16*6*6, 120),
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

class CNN_Mnist(nn.Module):
    """
    LeNet-5 inspired CNN for MNIST dataset. 
    Uses ReLU activations, BatchNorm, and MaxPooling.
    Suitable for 28x28 single-channel images.
    """
    def __init__(self, num_classes=10):
        super().__init__()
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
