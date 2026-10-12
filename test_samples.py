import argparse
import csv
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image

from model import SimpleCNN
from predict import preprocess_image


def parse_digit_from_filename(filename: str):
    """Extract the digit label from a filename like sample_3_1.jpg or sample_5.jpg."""
    match = re.search(r"sample_(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def predict_image(image_path: str, model: SimpleCNN, device) -> dict:
    """Run inference on a single image, returning the predicted digit and confidence (%)."""
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence, predicted_digit = torch.max(probabilities, 1)
    return {
        "predicted_digit": predicted_digit.item(),
        "confidence": confidence.item() * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Test model on custom sample images")
    parser.add_argument("--samples-dir", type=str, required=True,
                        help="Directory containing sample images (e.g., tests/Samples/)")
    parser.add_argument("--model", type=str, default="mnist_cnn.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="sample_results.csv",
                        help="Output CSV file for per-sample results")
    args = parser.parse_args()

    # Auto-select device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and set to eval mode
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model}")

    # Collect sample images (sample_3_1.jpg, sample_5.jpg, ...)
    sample_files = sorted(
        f for f in os.listdir(args.samples_dir)
        if f.lower().startswith("sample_") and f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"Found {len(sample_files)} sample images\n")

    # Run predictions
    results = []
    for filename in sample_files:
        filepath = os.path.join(args.samples_dir, filename)
        true_digit = parse_digit_from_filename(filename)
        if true_digit is None:
            print(f"  ? {filename}: could not parse digit from filename (skipped)")
            continue

        prediction = predict_image(filepath, model, device)
        predicted = prediction["predicted_digit"]
        confidence = prediction["confidence"]
        is_correct = predicted == true_digit

        results.append({
            "filename": filename,
            "true_digit": true_digit,
            "predicted_digit": predicted,
            "confidence": round(confidence, 2),
            "correct": is_correct,
        })
        mark = "OK" if is_correct else "WRONG"
        print(f"  {mark:<5} {filename}: true={true_digit}, "
              f"predicted={predicted}, confidence={confidence:.2f}%")

    if not results:
        print("No valid sample images found in directory.")
        return

    # Write per-sample CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_digit", "predicted_digit", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPer-sample results saved to: {args.output}")

    # -------------------- SUMMARY TABLE --------------------
    correct_count = sum(1 for r in results if r["correct"])
    total = len(results)
    overall_accuracy = 100.0 * correct_count / total
    avg_confidence = sum(r["confidence"] for r in results) / total

    print("\n" + "=" * 72)
    print("SUMMARY REPORT")
    print("=" * 72)
    print(f"  Overall Accuracy     : {correct_count}/{total} ({overall_accuracy:.2f}%)")
    print(f"  Average Confidence   : {avg_confidence:.2f}%")

    # Per-digit breakdown
    per_digit = defaultdict(lambda: {"total": 0, "correct": 0, "confidences": []})
    for r in results:
        d = per_digit[r["true_digit"]]
        d["total"] += 1
        d["correct"] += r["correct"]
        d["confidences"].append(r["confidence"])

    print("\n  Per-Digit Breakdown:")
    print(f"  {'Digit':>6} {'Total':>5} {'Correct':>7} {'Accuracy':>9} {'Avg Conf':>10} {'Status'}")
    print("  " + "-" * 60)
    for digit in sorted(per_digit):
        d = per_digit[digit]
        acc = 100.0 * d["correct"] / d["total"]
        avg_conf = sum(d["confidences"]) / len(d["confidences"])
        status = "" if acc == 100.0 else " <-- WRONG"
        print(f"  {digit:>6} {d['total']:>5} {d['correct']:>7} {acc:>8.2f}% "
              f"{avg_conf:>9.2f}% {status}")
    print("=" * 72)

    # List misclassified samples
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print("\nMisclassified samples (marked WRONG):")
        for r in wrong:
            print(f"  [X] {r['filename']}: true={r['true_digit']}, "
                  f"predicted={r['predicted_digit']}, confidence={r['confidence']:.2f}%")
    else:
        print("\nAll samples classified correctly.")


if __name__ == "__main__":
    main()