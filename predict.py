import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from model import SimpleCNN
from dataset import get_transforms


def preprocess_image(image_input) -> torch.Tensor:
    """Load and preprocess a digit image for MNIST model inference.

    Handles both standard cropped samples and uncropped high-resolution
    phone/webcam photographs with room shadows and illumination gradients:
    1. Converts image to grayscale.
    2. For camera photos / full pages (>300px), removes lighting gradients via
       local background subtraction and isolates the primary digit stroke using
       connected-component analysis.
    3. For cropped digits, applies robust border-sampled inversion and contrast stretching.
    4. Resizes to fit within a 20x20 box preserving original aspect ratio.
    5. Centers the digit on a 28x28 canvas using Center of Mass (MNIST specification).
    6. Applies standard MNIST normalization.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError("image_input must be a file path string or PIL Image")

    w, h = img.size

    # Check if this is an uncropped camera photograph or high-res scan
    is_camera_photo = (w > 300 or h > 300)

    if is_camera_photo:
        # Downscale for fast, robust illumination analysis
        img_scaled = img.copy()
        img_scaled.thumbnail((800, 800), Image.Resampling.LANCZOS)
        arr = np.array(img_scaled, dtype=np.float32)

        # Local background estimation to remove room shadows / lighting gradients
        bg = gaussian_filter(arr, sigma=20)
        diff_dark = np.maximum(bg - arr, 0.0)   # dark ink on lighter paper
        diff_light = np.maximum(arr - bg, 0.0)  # light stroke on dark background
        diff = diff_dark if np.percentile(diff_dark, 99.5) >= np.percentile(diff_light, 99.5) else diff_light

        # Dynamic threshold for ink stroke
        p99 = np.percentile(diff, 99.8)
        thresh = max(10.0, p99 * 0.40)
        binary = diff > thresh

        # Ignore 3% border to discard photo frame / desk edge artifacts
        sh, sw = binary.shape
        by, bx = max(1, int(sh * 0.03)), max(1, int(sw * 0.03))
        binary[:by, :] = 0
        binary[-by:, :] = 0
        binary[:, :bx] = 0
        binary[:, -bx:] = 0

        # Connected component analysis to isolate main digit stroke
        lbl, num = label(binary)
        if num == 0:
            canvas = np.zeros((28, 28), dtype=np.uint8)
        else:
            sizes = [(i, np.count_nonzero(lbl == i)) for i in range(1, num + 1)]
            sizes.sort(key=lambda x: x[1], reverse=True)
            mask = (lbl == sizes[0][0])
            r = np.where(np.sum(mask, axis=1) > 0)[0]
            c = np.where(np.sum(mask, axis=0) > 0)[0]

            crop = (diff * mask)[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
            c_norm = (crop / max(1e-5, crop.max())) * 255.0

            dh, dw = c_norm.shape
            c_pil = Image.fromarray(c_norm.astype(np.uint8))
            if dh > dw:
                nh = 20
                nw = max(1, int(round(dw * 20.0 / dh)))
            else:
                nw = 20
                nh = max(1, int(round(dh * 20.0 / dw)))
            resized = np.array(c_pil.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32)

            canvas = np.zeros((28, 28), dtype=np.float32)
            canvas[(28 - nh) // 2:(28 - nh) // 2 + nh, (28 - nw) // 2:(28 - nw) // 2 + nw] = resized

            total_m = canvas.sum()
            if total_m > 0:
                cy = np.sum(np.arange(28)[:, None] * canvas) / total_m
                cx = np.sum(np.arange(28)[None, :] * canvas) / total_m
                sh_y, sh_x = int(round(14.0 - cy)), int(round(14.0 - cx))
                shifted = np.zeros_like(canvas)
                for y in range(28):
                    for x in range(28):
                        sy, sx = y - sh_y, x - sh_x
                        if 0 <= sy < 28 and 0 <= sx < 28:
                            shifted[y, x] = canvas[sy, sx]
                canvas = shifted.astype(np.uint8)
            else:
                canvas = canvas.astype(np.uint8)
    else:
        # Standard cropped sample or benchmark digit
        arr = np.array(img, dtype=np.float32)
        h_arr, w_arr = arr.shape
        if h_arr > 2 and w_arr > 2:
            border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
            bg_brightness = float(np.median(border))
        else:
            bg_brightness = float(np.mean(arr))

        if bg_brightness > 127:
            arr = 255.0 - arr

        threshold = max(25.0, float(np.percentile(arr, 60)))
        arr = np.where(arr > threshold * 0.7, arr, 0.0)
        max_val = float(arr.max())
        if max_val > 0:
            arr = (arr / max_val) * 255.0

        rows = np.where(np.sum(arr, axis=1) > 0)[0]
        cols = np.where(np.sum(arr, axis=0) > 0)[0]

        if len(rows) == 0 or len(cols) == 0:
            canvas = np.zeros((28, 28), dtype=np.uint8)
        else:
            r_min, r_max = int(rows[0]), int(rows[-1])
            c_min, c_max = int(cols[0]), int(cols[-1])
            crop = arr[r_min:r_max + 1, c_min:c_max + 1]
            crop_h, crop_w = crop.shape

            crop_img = Image.fromarray(crop.astype(np.uint8))
            if crop_h > crop_w:
                new_h = 20
                new_w = max(1, int(round(crop_w * 20.0 / crop_h)))
            else:
                new_w = 20
                new_h = max(1, int(round(crop_h * 20.0 / crop_w)))

            crop_resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_arr = np.array(crop_resized, dtype=np.float32)

            canvas = np.zeros((28, 28), dtype=np.float32)
            canvas[(28 - new_h) // 2:(28 - new_h) // 2 + new_h, (28 - new_w) // 2:(28 - new_w) // 2 + new_w] = resized_arr

            total_mass = float(np.sum(canvas))
            if total_mass > 1e-4:
                cy = float(np.sum(np.arange(28)[:, None] * canvas)) / total_mass
                cx = float(np.sum(np.arange(28)[None, :] * canvas)) / total_mass
                shift_y, shift_x = int(round(14.0 - cy)), int(round(14.0 - cx))
                shifted = np.zeros_like(canvas)
                for y in range(28):
                    for x in range(28):
                        src_y, src_x = y - shift_y, x - shift_x
                        if 0 <= src_y < 28 and 0 <= src_x < 28:
                            shifted[y, x] = canvas[src_y, src_x]
                canvas = shifted.astype(np.uint8)
            else:
                canvas = canvas.astype(np.uint8)

    _, val_transform = get_transforms(augment=False)
    tensor = val_transform(Image.fromarray(canvas))
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