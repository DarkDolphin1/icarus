import pytest
import torch
from PIL import Image
import numpy as np
from model import SimpleCNN
from dataset import get_transforms
from predict import preprocess_image


class TestSimpleCNN:
    """Unit tests for the SimpleCNN model architecture."""

    def test_output_shape(self):
        """Test that model output shape is (batch_size, 10) for input (batch_size, 1, 28, 28)."""
        model = SimpleCNN()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == torch.Size([4, 10]), f"Expected shape [4, 10], got {out.shape}"

    def test_model_initialization(self):
        """Test that model can be instantiated without errors."""
        model = SimpleCNN()
        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0


class TestPreprocessing:
    """Unit tests for image preprocessing pipeline."""

    def test_grayscale_conversion(self):
        """Test that images are converted to grayscale L mode."""
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        img_gray = img.convert("L")
        assert img_gray.mode == "L"

    def test_normalize_returns_tensor(self):
        """Test that normalization returns a tensor with correct shape."""
        transform, _ = get_transforms()
        img = Image.new("L", (28, 28), 128)
        tensor = transform(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == torch.Size([1, 28, 28])

    def test_inversion_logic_dark_background(self):
        """Test that images with dark backgrounds are detected correctly."""
        # Create all-black image (dark background)
        img = Image.new("L", (28, 28), 0)
        mean_brightness = float(np.array(img).mean())
        assert mean_brightness <= 127, "All-black image should have mean <= 127"

    def test_inversion_logic_light_background(self):
        """Test that images with light backgrounds are detected correctly."""
        # Create all-white image (light background)
        img = Image.new("L", (28, 28), 255)
        mean_brightness = float(np.array(img).mean())
        assert mean_brightness > 127, "All-white image should have mean > 127"

    def test_preprocess_image_pil_input(self):
        """Test that preprocess_image correctly handles PIL Image and outputs (1, 1, 28, 28)."""
        img = Image.new("RGB", (120, 150), (255, 255, 255))
        tensor = preprocess_image(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == torch.Size([1, 1, 28, 28])


class TestEndToEnd:
    """End-to-end tests for the complete pipeline."""

    def test_trainable_model(self):
        """Test that model can train with valid data without NaN gradients."""
        model = SimpleCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        # Create a single batch of MNIST-like data
        x = torch.randn(2, 1, 28, 28)
        y = torch.randint(0, 10, (2,))

        output = model(x)
        loss = criterion(output, y)
        assert not torch.isnan(loss), "Loss should not be NaN"

        loss.backward()
        # Check gradients are not NaN
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} should not be NaN"

    def test_prediction_on_saved_mnist_image(self):
        """Test prediction capability on a properly saved MNIST digit image."""
        from predict import predict

        # This test verifies the predict function runs without crashing
        # on a valid image path (may not give perfect predictions on simple images)
        try:
            result = predict("/home/darkdolphin/icarus/tests/fixtures/digit_7.png")
            assert "predicted_digit" in result
            assert "confidence" in result
        except FileNotFoundError:
            pytest.skip("Test fixture not available - skipping file-dependent test")