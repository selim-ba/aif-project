# scripts/test_inference.py

from pathlib import Path
import json
import random

from PIL import Image
import torch

from app.posters.model import load_trained_model
from app.posters.inference import preprocess_image, predict_genres


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "data/processed/dataset/val"

WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = MODELS_DIR / "genres.json"


def pick_random_image():
    """Pick a random image from the validation dataset."""
    genre_dirs = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    if not genre_dirs:
        raise SystemExit(f"No validation folders found in {DATASET_DIR}")

    genre_dir = random.choice(genre_dirs)
    images = [p for p in genre_dir.iterdir() if p.is_file()]
    if not images:
        raise SystemExit(f"No images found in {genre_dir}")

    img_path = random.choice(images)
    return img_path, genre_dir.name


def main():
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"Model weights not found at {WEIGHTS_PATH}")

    if not GENRES_PATH.exists():
        raise SystemExit(f"Genres JSON not found at {GENRES_PATH}")

    with GENRES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    classes = data["classes"]

    print(f"Classes: {classes}")

    img_path, true_genre = pick_random_image()
    print(f"Picked image: {img_path}")
    print(f"True folder (genre): {true_genre}")

    image = Image.open(img_path).convert("RGB")

    tensor = preprocess_image(image)
    model = load_trained_model(str(WEIGHTS_PATH), num_classes=len(classes), device="cpu")

    predictions = predict_genres(model, tensor, classes, top_k=3)

    print("\nTop-3 predictions:")
    for p in predictions:
        print(f"  {p['genre']}: {p['score']:.3f}")


if __name__ == "__main__":
    main()

