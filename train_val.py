import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import DepthEstimationCNN
from dataset import DepthDataset, transform
import open3d as o3d  # For point cloud visualization

# File paths
excel_file = r"./data/nyu2_train.csv"
data_dir = "./"
model_save_path = 'output/model.pth'

# Check and create necessary directories
if not os.path.exists('output'):
    os.makedirs('output')
    print("Created 'output' directory.")

# Initialize model, dataset, and dataloader
dataset = DepthDataset(excel_file, data_dir, transform=transform, nrows=1000)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthEstimationCNN().to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_val(model, loader, criterion, optimizer, num_epochs, model_save_path, train_ratio=0.8):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        train_loss = 0.0
        val_loss = 0.0
        train_samples = 0
        val_samples = 0

        for idx, (image, depth) in enumerate(loader):
            image, depth = image.to(device), depth.to(device)

            if idx % int(1 / (1 - train_ratio)) != 0:
                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, depth)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_samples += 1
            else:
                model.eval()
                with torch.no_grad():
                    output = model(image)
                    loss = criterion(output, depth)
                    val_loss += loss.item()
                    val_samples += 1

        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), model_save_path)
    print("Model training complete and saved.")

train_val(model,dataloader,criterion,optimizer,10,'./output/model_val.pth')