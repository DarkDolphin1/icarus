import argparse
import glob
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, binary_dilation


def preprocess_to_canvas(image_input, return_stages: bool = False):
    """Load and preprocess a digit image into a 28x28 uint8 canvas.

    Handles both standard cropped samples and uncropped high-resolution
    phone/webcam photographs with room shadows and illumination gradients:
    1. Converts image to grayscale.
    2. For camera photos / full pages (>200px), removes lighting gradients via
       local background subtraction, detects ink polarity robustly, and groups
       adjacent digit strokes (e.g. European 1 with beak & base, 4, 5, etc.) using
       morphological dilation and connected-component analysis.
    3. For cropped digits, applies robust border-sampled inversion and contrast stretching.
    4. Resizes to fit within a 20x20 box preserving original aspect ratio.
    5. Centers the digit on a 28x28 canvas using Center of Mass (MNIST specification).

    Args:
        image_input: File path (str) or PIL Image.
        return_stages: If True, returns (canvas, stages_dict) with intermediate images.

    Returns:
        canvas (np.ndarray of shape (28, 28), dtype uint8) or
        tuple of (canvas, stages_dict) if return_stages is True.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise TypeError("image_input must be a file path string or PIL Image")

    w, h = img.size
    stages = {}

    is_camera_photo = (w > 200 or h > 200)

    if is_camera_photo:
        img_scaled = img.copy()
        img_scaled.thumbnail((800, 800), Image.Resampling.LANCZOS)
        arr = np.array(img_scaled, dtype=np.float32)
        stages["0_scaled_gray"] = img_scaled

        bg = gaussian_filter(arr, sigma=25)
        diff_dark = np.maximum(bg - arr, 0.0)
        diff_light = np.maximum(arr - bg, 0.0)
        diff = diff_dark if np.percentile(diff_dark, 99.9) >= np.percentile(diff_light, 99.9) else diff_light
        stages["1_diff"] = Image.fromarray(np.clip(diff / max(1e-5, diff.max()) * 255.0, 0, 255).astype(np.uint8))

        p99 = np.percentile(diff, 99.8)
        t_ink = max(4.0, p99 * 0.20)
        ink_binary = diff > t_ink

        sh, sw = ink_binary.shape
        by, bx = max(1, int(sh * 0.03)), max(1, int(sw * 0.03))
        ink_binary[:by, :] = False
        ink_binary[-by:, :] = False
        ink_binary[:, :bx] = False
        ink_binary[:, -bx:] = False

        struct = np.ones((11, 11), dtype=bool)
        grouped = binary_dilation(ink_binary, structure=struct)
        lbl, num = label(grouped)
        if num == 0:
            canvas = np.zeros((28, 28), dtype=np.uint8)
        else:
            sizes = [(i, np.count_nonzero(lbl == i)) for i in range(1, num + 1)]
            sizes.sort(key=lambda x: x[1], reverse=True)
            best_id = sizes[0][0]
            char_mask = ink_binary & (lbl == best_id)
            stages["2_mask"] = Image.fromarray((char_mask * 255).astype(np.uint8))

            r = np.where(np.sum(char_mask, axis=1) > 0)[0]
            c = np.where(np.sum(char_mask, axis=0) > 0)[0]

            crop = (diff * char_mask)[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
            c_norm = (crop / max(1e-5, crop.max())) * 255.0
            stages["3_crop"] = Image.fromarray(c_norm.astype(np.uint8))

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
            stages["3_crop"] = crop_img
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
                            shifted[y, x] = canvas[sy, sx]
                canvas = shifted.astype(np.uint8)
            else:
                canvas = canvas.astype(np.uint8)

    stages["4_canvas"] = Image.fromarray(canvas)

    if return_stages:
        return canvas, stages
    return canvas


def process_image(image_path: str, out_dir: str, save_stages: bool = False, scale: int = 10):
    """Preprocess a single image and save the output canvas to out_dir without running inference."""
    path = Path(image_path)
    if not path.is_file():
        print(f"[WARN] File not found: {image_path}")
        return None

    stem = path.stem
    os.makedirs(out_dir, exist_ok=True)

    if save_stages:
        canvas, stages = preprocess_to_canvas(str(path), return_stages=True)
        for stage_name, stage_img in stages.items():
            stage_out = os.path.join(out_dir, f"{stem}_stage_{stage_name}.png")
            stage_img.save(stage_out)
    else:
        canvas = preprocess_to_canvas(str(path), return_stages=False)

    # 1. Save raw 28x28 canvas (exact model input representation)
    canvas_img = Image.fromarray(canvas)
    raw_path = os.path.join(out_dir, f"{stem}_28x28.png")
    canvas_img.save(raw_path)

    # 2. Save enlarged preview (nearest-neighbor scaled for easy human inspection)
    preview_size = (28 * scale, 28 * scale)
    preview_img = canvas_img.resize(preview_size, Image.Resampling.NEAREST)
    preview_path = os.path.join(out_dir, f"{stem}_preview.png")
    preview_img.save(preview_path)

    print(f"Processed: {path.name}")
    print(f"  -> Model input (28x28) : {raw_path}")
    print(f"  -> Viewable preview    : {preview_path}")

    return {
        "source": str(path),
        "raw_28x28": raw_path,
        "preview": preview_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess digit images and export 28x28 canvases to out/ for manual verification."
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="One or more image paths or glob patterns to preprocess.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image file path to preprocess.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory of images to preprocess (processes all .jpg, .jpeg, .png).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out",
        help="Target output directory (default: out/).",
    )
    parser.add_argument(
        "--stages",
        action="store_true",
        help="Also export intermediate stages (scaled gray, diff, mask, bounding crop).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Upscale factor for preview images (default: 10, yields 280x280).",
    )

    args = parser.parse_args()

    targets = []
    if args.image:
        targets.append(args.image)
    if args.images:
        for item in args.images:
            expanded = glob.glob(item)
            if expanded:
                targets.extend(expanded)
            else:
                targets.append(item)
    if args.dir:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            targets.extend(glob.glob(os.path.join(args.dir, ext)))

    unique_targets = list(dict.fromkeys(targets))

    if not unique_targets:
        print("No input images specified.")
        print("Usage:")
        print("  python save_preprocessed.py <image_path>")
        print("  python save_preprocessed.py --image <image_path>")
        print("  python save_preprocessed.py --dir <directory_path>")
        print("  python save_preprocessed.py img1.jpg img2.jpg --stages")
        return

    print(f"Processing {len(unique_targets)} image(s) -> Output directory: {args.out_dir}/\n")
    processed_count = 0
    for target in unique_targets:
        res = process_image(target, args.out_dir, save_stages=args.stages, scale=args.scale)
        if res:
            processed_count += 1

    print(f"\nDone! Successfully preprocessed {processed_count}/{len(unique_targets)} images into {args.out_dir}/.")


if __name__ == "__main__":
    main()
