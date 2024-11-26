import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as T

class DepthDataset(Dataset):
    def __init__(self, excel_file, data_dir, transform=None, nrows=1000, target_size=(128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.data = pd.read_csv(excel_file, header=None, nrows=nrows)
        self.data.columns = ['image_path', 'depth_path']
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.data_dir, self.data.iloc[idx, 0])
        depth_path = os.path.join(self.data_dir, self.data.iloc[idx, 1])
        image = self.load_image(rgb_path)
        depth = self.load_depth(depth_path)
        return image, depth

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image

    def load_depth(self, depth_path):
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise ValueError(f"Failed to load depth map: {depth_path}")
        depth = cv2.resize(depth, self.target_size)
        depth = torch.from_numpy(depth).unsqueeze(0).float() / 255.0
        return depth

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
