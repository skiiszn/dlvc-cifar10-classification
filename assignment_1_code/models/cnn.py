from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    """
    Custom CNN for CIFAR-10 classification.
    Architecture: 4 convolutional blocks + classifier.
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # block 1: 32 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # scans the image
        self.bn1 = nn.BatchNorm2d(32) # keep the values in a reasonable range
        
        # block 2: 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    
        # block 3: 16 -> 8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # block 4: 8 -> 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # pooling - takes the max value
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout = nn.Dropout(0.3)
        
        # classifier: after 4 pooling layers: 32/2/2/2/2 = 2, so 256 * 2 * 2 = 1024
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 4x4 -> 2x2
        
        x = x.view(x.size(0), -1)  # single vector
        
        # classifier
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x