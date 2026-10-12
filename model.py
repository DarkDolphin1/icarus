import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Enhanced Convolutional Neural Network for robust 28x28 handwritten digit recognition.

    Optimized for out-of-distribution real-world handwriting through:
    - Stacked convolutional blocks with Batch Normalization (invariant to contrast & stroke intensity)
    - Hierarchical feature extraction across progressive channel capacities (1 -> 32 -> 64 -> 128)
    - Spatial 2D Dropout and Linear Dropout to prevent overfitting to clean templates
    - Batch-normalized dense classification head
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1: Low-level edge & stroke feature extraction (28x28 -> 14x14)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Block 2: Intermediate curve, loop, and junction detection (14x14 -> 7x7)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.25)

        # Block 3: High-level digit morphology & topological structure (7x7 -> 3x3)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.25)

        # Classification Head: Flattened feature maps (128 * 3 * 3 = 1152) -> 128 -> 10
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward pass on input image tensors of shape (N, 1, 28, 28)."""
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(self.pool1(F.relu(self.bn2(self.conv2(x)))))

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(self.pool2(F.relu(self.bn4(self.conv4(x)))))

        # Block 3
        x = self.dropout3(self.pool3(F.relu(self.bn5(self.conv5(x)))))

        # Classification Head
        x = torch.flatten(x, 1)
        x = self.dropout_fc(F.relu(self.bn_fc(self.fc1(x))))
        output = self.fc2(x)
        return output

