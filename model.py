import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Convolutional Neural Network for 28x28 grayscale handwritten digit recognition."""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional block extracting low-level feature maps
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        # Second convolutional block extracting higher-level spatial features
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Max pooling layer to halve spatial dimensions (28x28 -> 14x14 -> 7x7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Regularization dropout layers to prevent overfitting
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Dense classification head mapping flattened feature maps to class probabilities
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes the forward pass on input image tensors of shape (N, 1, 28, 28)."""

        # Pass through Conv1, apply ReLU activation, and downsample with MaxPool (1x28x28 -> 32x14x14)
        x = self.pool(F.relu(self.conv1(x)))

        # Pass through Conv2, apply ReLU activation, and downsample with MaxPool (32x14x14 -> 64x7x7)
        x = self.pool(F.relu(self.conv2(x)))

        # Apply dropout regularization before flattening feature tensor
        x = self.dropout1(x)

        # Flatten spatial feature maps into 1D vector per sample (64*7*7 = 3136)
        x = torch.flatten(x, 1)

        # Pass through first dense layer with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)

        # Compute raw output logits for digits 0-9
        output = self.fc2(x)
        return output
