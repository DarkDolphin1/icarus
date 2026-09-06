import argparse
import csv
from collections import defaultdict

import torch
import torch.nn.functional as F

from model import SimpleCNN
from dataset import get_val_loader


def main():
    parser = argparse.ArgumentParser(description="Evaluate SimpleCNN on the MNIST test set")
    parser.add_argument("--model", type=str, default="mnist_cnn.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default="mnist_eval_results.csv",
                        help="Optional CSV file for per-sample results")
    parser.add_argument("--show-errors", type=int, default=20,
                        help="How many misclassified samples to list on screen")
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

    # MNIST test set (10,000 samples)
    val_loader = get_val_loader(batch_size=args.batch_size, shuffle=False)

    # Run predictions over the whole test set
    all_predictions = []
    all_confidences = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            all_predictions.extend(predicted.cpu().tolist())
            all_confidences.extend((confidence * 100).cpu().tolist())
            all_targets.extend(target.cpu().tolist())

    total = len(all_targets)
    correct = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
    accuracy = 100.0 * correct / total
    avg_confidence = sum(all_confidences) / total
    correct_conf = [c for p, t, c in zip(all_predictions, all_targets, all_confidences) if p == t]
    wrong_conf = [c for p, t, c in zip(all_predictions, all_targets, all_confidences) if p != t]
    avg_correct_conf = sum(correct_conf) / len(correct_conf) if correct_conf else 0.0
    avg_wrong_conf = sum(wrong_conf) / len(wrong_conf) if wrong_conf else 0.0
    max_conf = max(all_confidences)
    min_conf = min(all_confidences)

    # -------------------- SUMMARY TABLE --------------------
    print("\n" + "=" * 72)
    print("SUMMARY REPORT - MNIST TEST SET")
    print("=" * 72)
    print(f"  Total samples        : {total}")
    print(f"  Overall Accuracy     : {correct}/{total} ({accuracy:.2f}%)")
    print(f"  Average Confidence   : {avg_confidence:.2f}%  (max {max_conf:.2f}%, min {min_conf:.2f}%)")
    print(f"  Avg Conf  CORRECT    : {avg_correct_conf:.2f}%")
    print(f"  Avg Conf  WRONG      : {avg_wrong_conf:.2f}%")

    # Per-digit breakdown
    per_digit = defaultdict(lambda: {"total": 0, "correct": 0, "confidences": []})
    for p, t, c in zip(all_predictions, all_targets, all_confidences):
        d = per_digit[t]
        d["total"] += 1
        d["correct"] += p == t
        d["confidences"].append(c)

    print("\n  Per-Digit Breakdown:")
    print(f"  {'Digit':>6} {'Total':>5} {'Correct':>7} {'Accuracy':>9} {'Avg Conf':>10} {'Err/FN':>7} {'Err/FP':>7}")
    print("  " + "-" * 62)
    for digit in range(10):
        d = per_digit[digit]
        acc = 100.0 * d["correct"] / d["total"]
        avg_conf = sum(d["confidences"]) / len(d["confidences"])
        fn = d["total"] - d["correct"]
        fp = sum(1 for p, t in zip(all_predictions, all_targets) if p == digit and p != t)
        mark = "" if acc >= 99 else " <--"
        print(f"  {digit:>6} {d['total']:>5} {d['correct']:>7} {acc:>8.2f}% "
              f"{avg_conf:>9.2f}% {fn:>7} {fp:>7}{mark}")
    print("=" * 72)

    # Confusion matrix
    print("\n  Confusion Matrix (rows=true digit, cols=predicted):")
    print("  " + "     ".join(f"{c:>4}" for c in range(10)))
    matrix = [[0] * 10 for _ in range(10)]
    for p, t in zip(all_predictions, all_targets):
        matrix[t][p] += 1
    for t in range(10):
        print(f"  {t}  " + "  ".join(f"{v:>4}" for v in matrix[t]))

    # Misclassified samples
    misclassified = [
        (i, t, p, c)
        for i, (p, t, c) in enumerate(zip(all_predictions, all_targets, all_confidences))
        if p != t
    ]
    print(f"\n  Misclassified: {len(misclassified)}/{total} samples (index: true -> predicted, conf)")
    for i, t, p, c in misclassified[:args.show_errors]:
        print(f"    index {i:5d} : true={t}, predicted={p}, confidence={c:.2f}%")
    if len(misclassified) > args.show_errors:
        print(f"    ... and {len(misclassified) - args.show_errors} more (see CSV output)")

    # Save full per-sample results to CSV
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "true_digit", "predicted_digit", "confidence", "correct"])
            for i, (p, t, c) in enumerate(zip(all_predictions, all_targets, all_confidences)):
                writer.writerow([i, t, p, round(c, 2), p == t])
        print(f"\n  Full per-sample results saved to: {args.output}")


if __name__ == "__main__":
    main()