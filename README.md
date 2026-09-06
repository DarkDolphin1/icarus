# Icarus: MNIST Digit Recognition

A simple and basic Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) from the MNIST dataset.

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
python train.py --epochs 5 --batch-size 64
```

- Trains for 5 epochs (default) using the MNIST dataset
- Achieves ~99%+ validation accuracy
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

- **Conv2d Block 1**: 1 input channel → 32 filters, kernel 3x3, ReLU, MaxPool2d → output: [B, 32, 14, 14]
- **Conv2d Block 2**: 32 channels → 64 filters, kernel 3x3, ReLU, MaxPool2d → output: [B, 64, 7, 7]
- **Dropout**: Rate 0.25 after pooling, Rate 0.5 before final dense layer
- **Fully Connected**: 64 × 7 × 7 = 3136 inputs → 128 hidden → 10 output logits (digits 0-9)

## Training Details

- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 5 (sufficient to reach >98.5% accuracy quickly)
- **Batch Size**: 64 (configurable via `--batch-size`)
- **Device**: Auto-detects CUDA GPU > Apple MPS > CPU

## Prediction Pipeline

The `predict.py` script accepts an image path and performs:

1. **Grayscale conversion**: Image converted to L (grayscale) mode
2. **Auto-inversion**: If image mean brightness > 127, colors are inverted to match MNIST convention (black digit on white background)
3. **Resize**: Resized to 28×28 pixels using bilinear interpolation
4. **Normalization**: Normalized using MNIST statistics (mean=0.1307, std=0.3081)
5. **Inference**: Forward pass through SimpleCNN with Softmax to get class probabilities

Output format:
```
Predicted digit: 7
Confidence: 98.50%
```

## Citation / Inspiration

This project is inspired by the "Hello World" of machine learning - recognizing handwritten digits from the MNIST dataset. The SimpleCNN architecture achieves over 99% accuracy while remaining lightweight and easy to understand.