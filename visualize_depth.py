
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import os


def depth_to_3d(depth_map):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()
    return x, y, z


def load_image(image_path, target_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.resize(image, target_size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def plot_3d_scatter(depth_map, image_path):
    original_image = load_image(image_path, (depth_map.shape[1], depth_map.shape[0]))
    x, y, z = depth_to_3d(depth_map)
    colors = original_image.reshape(-1, 3) / 255.0  # Normalize colors to [0, 1]

    print(f"Scatter plot: Sample points - x={x[:5]}, y={y[:5]}, z={z[:5]}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=colors, s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('3D Scatter Plot of Depth Map')
    plt.show()


def plot_3d_surface(depth_map):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, depth_map, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')
    ax.set_title('3D Surface Plot of Depth Map')
    plt.show()


if __name__ == "__main__":
    # Define paths
    depth_map_path = "output/depth_map.npy"
    image_path = "Bird.jpeg"

    # Load the depth map
    if not os.path.exists(depth_map_path):
        raise FileNotFoundError(f"Depth map file not found: {depth_map_path}")
    depth_map = np.load(depth_map_path)


    plt.imshow(depth_map, cmap='gray')
    plt.title('Depth Map')
    plt.colorbar()
    plt.show()

    # Visualization
    print("Plotting 3D surface from depth map...")
    plot_3d_surface(depth_map)

    print("Plotting 3D scatter plot with depth map and RGB colors...")
    plot_3d_scatter(depth_map, image_path)

    print("Saving and visualizing point cloud...")