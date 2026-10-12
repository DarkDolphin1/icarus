import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from model import SimpleCNN
from dataset import get_transforms


def preprocess_image(image_input) -> torch.Tensor:
    """Load and preprocess a digit image for MNIST model inference.

    Optimized for real-world photos, scans, and drawings:
    1. Loads image and converts to grayscale (L mode)
    2. Detects background via border sampling & auto-inverts (paper vs dark canvas)
    3. Suppresses background noise and normalizes dynamic range
    4. Extracts bounding box of the handwritten digit
    5. Resizes to fit a 20x20 box preserving original aspect ratio
    6. Centers the digit in a 28x28 canvas using Center of Mass (MNIST specification)
    7. Applies standard MNIST normalization
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError("image_input must be a file path string or PIL Image")

    arr = np.array(img, dtype=np.float32)

    # Step 1: Detect background tone using image border pixels
    h, w = arr.shape
    if h > 2 and w > 2:
        border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        bg_brightness = float(np.median(border_pixels))
    else:
        bg_brightness = float(np.mean(arr))

    # Real-world paper images have light background and dark ink -> invert
    if bg_brightness > 127:
        arr = 255.0 - arr

    # Step 2: Suppress background noise and stretch contrast
    threshold = max(25.0, float(np.percentile(arr, 60)))
    arr = np.where(arr > threshold * 0.7, arr, 0.0)
    max_val = float(arr.max())
    if max_val > 0:
        arr = (arr / max_val) * 255.0

    # Step 3: Find bounding box of foreground digit
    row_sums = np.sum(arr, axis=1)
    col_sums = np.sum(arr, axis=0)
    rows = np.where(row_sums > 0)[0]
    cols = np.where(col_sums > 0)[0]

    if len(rows) == 0 or len(cols) == 0:
        canvas = np.zeros((28, 28), dtype=np.uint8)
    else:
        r_min, r_max = int(rows[0]), int(rows[-1])
        c_min, c_max = int(cols[0]), int(cols[-1])
        crop = arr[r_min:r_max + 1, c_min:c_max + 1]
        crop_h, crop_w = crop.shape

        crop_img = Image.fromarray(crop.astype(np.uint8))

        # Step 4: Aspect-ratio preserving resize into 20x20 box
        if crop_h > crop_w:
            new_h = 20
            new_w = max(1, int(round(crop_w * 20.0 / crop_h)))
        else:
            new_w = 20
            new_h = max(1, int(round(crop_h * 20.0 / crop_w)))

        crop_resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        resized_arr = np.array(crop_resized, dtype=np.float32)

        # Step 5: Center-of-mass centering in 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.float32)
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

        total_mass = float(np.sum(canvas))
        if total_mass > 1e-4:
            cy = float(np.sum(np.arange(28)[:, None] * canvas)) / total_mass
            cx = float(np.sum(np.arange(28)[None, :] * canvas)) / total_mass
            shift_y = int(round(14.0 - cy))
            shift_x = int(round(14.0 - cx))

            shifted = np.zeros_like(canvas)
            for y in range(28):
                for x in range(28):
                    src_y = y - shift_y
                    src_x = x - shift_x
                    if 0 <= src_y < 28 and 0 <= src_x < 28:
                        shifted[y, x] = canvas[src_y, src_x]
            canvas = shifted

        canvas = canvas.astype(np.uint8)

    # Step 6: Apply transforms (ToTensor + Normalize)
    _, val_transform = get_transforms(augment=False)
    tensor = val_transform(Image.fromarray(canvas))

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