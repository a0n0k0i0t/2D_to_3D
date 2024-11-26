import torch
import torch.nn as nn


class DepthEstimationCNN(nn.Module):
    def __init__(self):
        super(DepthEstimationCNN, self).__init__()

        # Encoder: Feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Downsample 1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample 2
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsample 3
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Decoder: Upsample to reconstruct depth map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample 1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample 3
            nn.Sigmoid()  
        )

    def forward(self, x):
        x = self.encoder(x) 
        x = self.decoder(x)
        return x
