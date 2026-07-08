"""
Train the CNN on CIFAR-10 and log everything to MLflow.

CIFAR-10 is downloaded automatically on first run (~170MB).

Usage:
    python src/train.py
    python src/train.py --epochs 20 --lr 0.001 --batch-size 64
"""

import argparse
import os

import mlflow
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from src.model import ImageClassifier

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders(batch_size: int, data_dir: str):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader, criterion) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * len(labels)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train(epochs: int, lr: float, batch_size: int) -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    mlflow.set_experiment("image-classifier")

    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "lr": lr, "batch_size": batch_size, "device": DEVICE})

        train_loader, test_loader = get_dataloaders(batch_size, data_dir)
        model = ImageClassifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            scheduler.step()

            val_loss, val_acc = evaluate(model, test_loader, criterion)
            mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(f"Epoch {epoch}/{epochs}  loss={val_loss:.4f}  acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)

        mlflow.log_metric("best_val_acc", best_acc)
        print(f"Training complete. Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch_size)
