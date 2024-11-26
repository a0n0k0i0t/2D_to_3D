import re
import matplotlib.pyplot as plt

# Raw log data
log_data = """
Epoch [1/10], Train Loss: 0.0151, Validation Loss: 0.0152
Epoch [2/10], Train Loss: 0.0128, Validation Loss: 0.0124
Epoch [3/10], Train Loss: 0.0120, Validation Loss: 0.0118
Epoch [4/10], Train Loss: 0.0105, Validation Loss: 0.0109
Epoch [5/10], Train Loss: 0.0095, Validation Loss: 0.0082
Epoch [6/10], Train Loss: 0.0073, Validation Loss: 0.0076
Epoch [7/10], Train Loss: 0.0064, Validation Loss: 0.0065
Epoch [8/10], Train Loss: 0.0055, Validation Loss: 0.0054
Epoch [9/10], Train Loss: 0.0046, Validation Loss: 0.0042
Epoch [10/10], Train Loss: 0.0039, Validation Loss: 0.0038
"""
epoch_loss_data = re.findall(r"Epoch \[(\d+)/\d+\], Train Loss: ([\d.]+), Validation Loss: ([\d.]+)", log_data)

epochs = [int(epoch) for epoch, _, _ in epoch_loss_data]
train_losses = [float(train_loss) for _, train_loss, _ in epoch_loss_data]
val_losses = [float(val_loss) for _, _, val_loss in epoch_loss_data]

# Plot the data
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.plot(epochs, val_losses, marker='+', label='Validation Loss')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(1, 11))
plt.grid(True)
plt.legend()
plt.show()
