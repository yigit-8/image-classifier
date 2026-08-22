"""
Train the CNN on CIFAR-10 and log everything to MLflow.

CIFAR-10 is downloaded automatically on first run (~170MB).

Runs are reproducible: the seed is fixed (default 42), logged to MLflow, and
stored inside the checkpoint, and cuDNN runs in deterministic mode unless
--no-deterministic is passed.

Usage:
    python src/train.py
    python src/train.py --epochs 20 --lr 0.001 --batch-size 64
    python src/train.py --seed 7 --no-deterministic
"""

import argparse
import json
import os
import random

import mlflow
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms

from src.model import IMAGE_SIZE, NORM_MEAN, NORM_STD, ImageClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pt")
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the training loop touches.

    Without this, shuffled DataLoaders and random augmentation make the same
    command produce a different number on every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN kernels cost some throughput but are what makes a
    # published accuracy number reproducible on the same hardware.
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_dataloaders(batch_size: int, data_dir: str):
    train_transform = transforms.Compose([
        transforms.RandomCrop(IMAGE_SIZE[0], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
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


def save_checkpoint(model: nn.Module, params: dict, seed: int, metrics: dict) -> None:
    """Save weights together with what produced them.

    A bare state_dict carries no record of its hyperparameters, seed or score,
    so a checkpoint on disk is unattributable. Bundling them mirrors how the
    airflow-ml-pipeline sibling project stores model + metrics + params.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": params,
            "seed": seed,
            "metrics": metrics,
            "classes": ImageClassifier.CLASSES,
        },
        MODEL_PATH,
    )
    with open(METRICS_PATH, "w") as f:
        json.dump({"params": params, "seed": seed, "metrics": metrics}, f, indent=2)


def train(
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int = DEFAULT_SEED,
    deterministic: bool = True,
) -> None:
    set_seed(seed, deterministic)

    data_dir = DATA_DIR
    mlflow.set_experiment("image-classifier")

    params = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "deterministic": deterministic,
        "device": DEVICE,
    }

    with mlflow.start_run():
        mlflow.log_params(params)

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
                save_checkpoint(
                    model,
                    params,
                    seed,
                    {"val_loss": val_loss, "val_acc": val_acc, "best_val_acc": best_acc,
                     "epoch": epoch},
                )

        mlflow.log_metric("best_val_acc", best_acc)
        print(f"Training complete. Best accuracy: {best_acc:.4f} (seed {seed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministic cuDNN kernels (default on). --no-deterministic trades "
             "reproducibility for speed.",
    )
    args = parser.parse_args()
    train(args.epochs, args.lr, args.batch_size, args.seed, args.deterministic)
