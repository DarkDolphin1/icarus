import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from model import SimpleCNN
from dataset import get_transforms


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess a digit image for MNIST model inference.

    Steps:
    1. Open image and convert to grayscale (L mode)
    2. Auto-invert if background is light (mean brightness > 127)
    3. Resize to 28x28 pixels
    4. Convert to tensor and normalize with MNIST statistics
    """
    # Step 1: Load image in grayscale
    img = Image.open(image_path).convert("L")

    # Step 2: Auto-inversion detection
    # MNIST has black digits on white background (0=black, 255=white)
    # If input image has light background (mean > 127), invert to match
    pixels = list(img.getdata())
    mean_brightness = sum(pixels) / len(pixels)

    if mean_brightness > 127:
        # Invert pixel values: 255 - pixel_value for each pixel
        inverted_pixels = [255 - p for p in pixels]
        # Create new image from inverted pixel data
        # PIL Image.frombytes expects raw bytes; use frombuffer with proper shape
        img = Image.frombytes("L", img.size, bytes(inverted_pixels))

    # Step 3: Resize to 28x28 pixels
    img = img.resize((28, 28), Image.BILINEAR)

    # Step 4: Apply transforms (ToTensor + Normalize)
    _, val_transform = get_transforms()
    tensor = val_transform(img)

    # Add batch dimension: shape becomes [1, 1, 28, 28]
    tensor = tensor.unsqueeze(0)
    return tensor


def predict(image_path: str, model_path: str = "mnist_cnn.pth") -> dict:
    """Run inference on a single digit image and return prediction results.

    Args:
        image_path: Path to the input digit image file
        model_path: Path to the trained model checkpoint (.pth)

    Returns:
        dict with 'predicted_digit' (int) and 'confidence' (float in 0-100)
    """
    # Determine device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model and set to eval mode
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess the input image
    input_tensor = preprocess_image(image_path).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_digit = torch.max(probabilities, 1)

    predicted_digit_idx = predicted_digit.item()
    confidence_pct = confidence.item() * 100

    return {
        "predicted_digit": predicted_digit_idx,
        "confidence": round(confidence_pct, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict digit from image using SimpleCNN")
    parser.add_argument("--image", type=str, required=True, help="Path to input digit image")
    parser.add_argument("--model", type=str, default="mnist_cnn.pth", help="Path to trained model checkpoint")
    args = parser.parse_args()

    result = predict(args.image, args.model)
    print(f"Predicted digit: {result['predicted_digit']}")
    print(f"Confidence: {result['confidence']}%")


if __name__ == "__main__":
    main()