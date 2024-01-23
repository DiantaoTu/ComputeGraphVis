'''
Author: Diantao Tu
Date: 2024-01-04 10:28:34
'''
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class ResNet50Autoencoder(nn.Module):
    def __init__(self):
        super(ResNet50Autoencoder, self).__init__()

        # Load pre-trained ResNet50 model
        self.resnet50 = resnet50(pretrained=False)

        # Remove the fully connected layers
        modules = list(self.resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)  # Assuming input channels are 3
        )

    def forward(self, x):
        x = self.resnet50(x)
        x = self.upsample(x)
        return x
    
