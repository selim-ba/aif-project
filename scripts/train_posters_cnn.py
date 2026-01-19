import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from app.posters.model import build_model


# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = MODELS_DIR / "genres.json"
# ---------------


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """
    Must match the preprocessing used in app/posters/inference.py.
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def load_datasets():
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise SystemExit(
            f"Dataset folders not found.\n"
            f"Expected:\n - {TRAIN_DIR}\n - {VAL_DIR}\n"
            f"Did you run scripts/create_repositories_train_val.py?"
        )

    transform = get_transforms()
    train_data = datasets.ImageFolder(str(TRAIN_DIR), transform=transform)
    val_data = datasets.ImageFolder(str(VAL_DIR), transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Found {len(train_data)} training images across {len(train_data.classes)} genres.")
    print(f"Found {len(val_data)} validation images.")

    return train_data, val_data, train_loader, val_loader


def evaluate(model, dataloader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_data, val_data, train_loader, val_loader = load_datasets()
    num_classes = len(train_data.classes)

    print(f"Genres (class order): {train_data.classes}")

    # Build model
    model = build_model(num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- train loss: {avg_loss:.4f}, val acc: {val_acc:.4f}"
        )

    # Save genres mapping
    with GENRES_PATH.open("w", encoding="utf-8") as f:
        json.dump({"classes": train_data.classes}, f, ensure_ascii=False, indent=2)
    print(f"Saved genre classes to {GENRES_PATH}")

    # Save CPU-compatible weights
    model.to("cpu")
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved model weights to {WEIGHTS_PATH} (CPU-compatible)")


if __name__ == "__main__":
    main()

