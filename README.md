# 2D Image to 3D Point Cloud Depth Map
### (Contributions by Kush, Ankit, Ashutosh, Jasbir)

## PPT Link ([here](https://www.canva.com/design/DAGXdRC0h_E/JhMqzfEzHj6ISiktKK37cw/edit?utm_content=DAGXdRC0h_E&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton))
This project implements a deep learning-based depth estimation system using a custom convolutional neural network (CNN). The system trains on paired RGB images and depth maps to predict depth maps for new images. 

## Features
- **Depth Estimation Model**: Encoder-decoder architecture implemented in PyTorch.
- **Dataset Loader**: Handles paired RGB and depth maps, with preprocessing and augmentation.
- **Training and Validation**: Includes a training loop with logging of loss metrics.
- **Visualization**: Tools for visualizing depth maps in 2D and 3D.
- **Point Cloud Generation**: Generates and visualizes point clouds using Open3D.

---

## Directory Structure
```
project/
├── data/                     # Dataset directory
├── output/                   # Directory for saving models and predictions
├── main.py                   # Main script for training and prediction
├── model.py                  # Model implementation (DepthEstimationCNN)
├── dataset.py                # Dataset loading and preprocessing
├── train_val.py              # Training and validation script
├── visualize_depth.py        # Depth visualization and 3D rendering
├── plot.py                   # Script for plotting training/validation loss
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Kaggle account to download the dataset

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/a0n0k0i0t/2D_to_3D
   cd project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
1. Download the NYU Depth v2 dataset from Kaggle:
   [NYU Depth V2 Dataset](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) (Kaggle account required).
2. Extract the dataset and place the `nyu2_train.csv` file and associated image files in the home directory.
3. Ensure the dataset structure matches the paths in `nyu2_train.csv`.

---

## Training
To train the depth estimation model:
```bash
python main.py
```
The trained model will be saved in the `output/` directory as `model.pth`.

- **Training Results**:
   - Training result plot showing the training and validation loss curves saved as `./images/training_result.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/training_result.jpg" width="50%" height="50%" />

   Model trained for 10 epochs with a steadily decreasing loss.
The MSE loss computed between the predicted and ground truth values across all pixels in the depth map.

### Validating Dataset
To validate the dataset and model setup, run the `train_val.py` script:
```bash
python train_val.py
```
This will perform a training-validation split and log the loss metrics for each epoch.

---

### Predicting Depth Maps
To predict a depth map for a new image:
1. Place the input image in the project directory (e.g., `Bird.jpg`).
2. Update the `input_image_path` variable in `main.py` with the path to your image.
3. Make sure to have model weights. Trained model is in `output/model.pth`.
3. Run the script:
   ```bash
   python main.py
   ```
The predicted depth map will be saved as `output/depth_map.npy`.

---

## Visualization

### Depth Map Visualization
Use `visualize_depth.py` to visualize the depth map:
```bash
python visualize_depth.py
```

## Example Input and Output

### Example Input:
- **Input Image**: `Bird.jpeg` (RGB Image)

<img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/Bird.jpeg" width="50%" height="50%" />

### Example Output:
- **Depth Map**: 
   - Intermediate depth map saved as `./images/depth_map.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/depth_map.jpg" width="50%" height="50%" />

   - 3D depth map saved as `./images/3d_surface_plot_of_depthmap.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/3d_surface_plot_of_depthmap.jpg" width="50%" height="50%" />

- **Point Cloud Views**:
   - Front view of the point cloud saved as `./images/front_view.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/front_view.jpg" width="50%" height="50%" />

   - Top view of the point cloud saved as `./images/top_view.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/top_view.jpg" width="50%" height="50%" />

   - Side view of the point cloud saved as `./images/side_view.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/side_view.jpg" width="50%" height="50%" />

   - Birds-eye view of the point cloud saved as `./images/birds_eye_view.jpg`.

   <img src="https://github.com/a0n0k0i0t/2D_to_3D/blob/main/images/birds_eye_view.jpg" width="50%" height="50%" />

---

## Code Overview

### `model.py`
Defines the `DepthEstimationCNN`, a CNN with an encoder-decoder architecture for depth estimation.

### `dataset.py`
Implements the `DepthDataset` class for loading and preprocessing paired RGB and depth maps.

### `train_val.py`
Contains the training loop and validation logic for the model. This script can be run independently to validate the dataset and training process.

### `visualize_depth.py`
Handles 2D and 3D visualization of predicted depth maps and point cloud generation.

### `plot.py`
Plots training and validation loss curves to monitor model performance.

---

## Dependencies
The project depends on the following Python libraries:
- `torch`, `torchvision` for deep learning.
- `opencv-python` for image processing.
- `pandas` for handling the dataset.
- `matplotlib`, `numpy` for visualization and numerical operations.
- `open3d` for point cloud visualization.

---

## Future Work
- Use multiple images to get an overall 3D view of the object from all angles.
- Enhance model architecture for better accuracy.
- Experiment with larger datasets.

---
