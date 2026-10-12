# Icarus: MNIST Digit Recognition

A simple and robust Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) from the MNIST dataset, optimized for real-world handwriting.

## Project Structure

```
/icarus/
├── README.md              # Project documentation
├── requirements.txt       # Dependencies (torch, torchvision, pillow, pytest)
├── model.py               # SimpleCNN architecture
├── dataset.py             # MNIST DataLoader utilities with transforms
├── train.py               # Training loop with validation & checkpoint saving
├── predict.py             # CLI inference script for single image prediction
├── tests/
│   └── test_model.py      # Unit tests for model & preprocessing
├── mnist_cnn.pth          # Trained model checkpoint (generated after training)
└── data/                  # MNIST dataset directory (auto-created on first run)
```

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project
cd icarus

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
# Or customize epochs and batch size:
python train.py --epochs 10 --batch-size 64
```

- Trains for 10 epochs (default) using the MNIST dataset with data augmentation
- Achieves ~99.6% validation accuracy
- Saves model checkpoint to `mnist_cnn.pth`

### 3. Run Prediction on an Image

```bash
# Predict digit from a local image file
python predict.py --image /path/to/digit.png

# Specify a custom model checkpoint:
python predict.py --image /path/to/digit.png --model mnist_cnn.pth
```

### 4. Run Unit Tests

```bash
pytest tests/test_model.py
```

## Model Architecture

The `SimpleCNN` model in `model.py` consists of:

- **Conv2d Block 1**: 2× (Conv 3x3, BatchNorm2d, ReLU), MaxPool2d, Dropout2d (0.25) → output: [B, 32, 14, 14]
- **Conv2d Block 2**: 2× (Conv 3x3, BatchNorm2d, ReLU), MaxPool2d, Dropout2d (0.25) → output: [B, 64, 7, 7]
- **Conv2d Block 3**: Conv 3x3, BatchNorm2d, ReLU, MaxPool2d, Dropout2d (0.25) → output: [B, 128, 3, 3]
- **Fully Connected**: 128 × 3 × 3 = 1152 inputs → 128 (BatchNorm1d, ReLU, Dropout 0.4) → 10 output logits (digits 0-9)

## Training Details

- **Optimizer**: AdamW with learning rate 0.002 and weight decay 1e-4
- **Scheduler**: CosineAnnealingLR
- **Data Augmentation**: RandomAffine (rotation ±15°, translation ±10%, scale 0.9-1.1x, shear 10°)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10 (configurable via `--epochs`)
- **Batch Size**: 64 (configurable via `--batch-size`)
- **Device**: Auto-detects CUDA GPU > Apple MPS > CPU

## Prediction Pipeline

The `predict.py` script accepts an image path and performs:

1. **Grayscale conversion**: Image converted to L (grayscale) mode
2. **Auto-inversion & noise filtering**: Background tone detected from borders; inverts light paper and removes low-level noise
3. **Bounding-box isolation**: Isolates digit strokes and removes empty page borders
4. **Centering & resize**: Digit aspect ratio preserved within 20×20 box and centered on 28×28 canvas by center of mass
5. **Normalization**: Normalized using MNIST statistics (mean=0.1307, std=0.3081)
6. **Inference**: Forward pass through SimpleCNN with Softmax to get class probabilities

Output format:
```
Predicted digit: 7
Confidence: 98.50%
```

## Citation / Inspiration

This project is inspired by the "Hello World" of machine learning - recognizing handwritten digits from the MNIST dataset. The SimpleCNN architecture achieves over 99% accuracy while remaining lightweight and easy to understand.