import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_train_loader, get_val_loader
import argparse


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Execute one training epoch over the provided data loader."""
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """Execute validation pass and return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train SimpleCNN on MNIST")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Input batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="mnist_cnn.pth", help="Path to save model checkpoint")
    args = parser.parse_args()

    # Auto-select device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare data loaders
    train_loader = get_train_loader(batch_size=args.batch_size)
    val_loader = get_val_loader(batch_size=args.batch_size)

    # Training loop across specified epochs
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.2f}%"
        )

    # Save trained model checkpoint
    torch.save(model.state_dict(), args.checkpoint)
    print(f"Model checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()