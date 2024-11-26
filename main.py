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


# Train the model
def train_model():
    model.train()
    for epoch in range(10):  # Number of epochs
        running_loss = 0.0
        for images, depths in dataloader:
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(dataloader)}")
    torch.save(model.state_dict(), model_save_path)
    print("Model training complete and saved.")


# Predict depth and visualize
def predict_depth(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
      
    model.eval()
    print(f"Predicting depth for image: {image_path}")
    image = dataset.load_image(image_path)
    print(f"Image shape: {image.shape}")
    with torch.no_grad():
        depth_map = model(image.unsqueeze(0).to(device)).cpu().squeeze().numpy()
        np.save('output/depth_map.npy', depth_map)  # Save depth map
        print(f"Depth map saved to output/depth_map.npy")

# Main execution
if __name__ == "__main__":
    if not os.path.exists(model_save_path):
        print("No pre-trained model found. Training the model...")
        train_model()
    else:
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        print("Loaded pre-trained model.")

    # Predict and visualize depth for a new image
    input_image_path = 'input_2.jpeg'  # Replace with your input image path
    if not os.path.exists(input_image_path):
        print(f"Input image '{input_image_path}' not found. Please provide a valid path.")
    else:
        predict_depth(input_image_path)
        