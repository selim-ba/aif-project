import shutil
import random
import sys
import os
from pathlib import Path

# --- 1. FIX DES CHEMINS (Indispensable pour la portabilité) ---
script_dir = Path(__file__).resolve().parent
if script_dir.name == 'scripts':
    project_root = script_dir.parent
else:
    project_root = script_dir
# -------------------------------------------------------------

# --- CONFIGURATION ADAPTÉE ---

# 1. SOURCE : Où sont vos images DÉJÀ triées par dossiers ?
# (Adaptez ce chemin si vos dossiers triés sont ailleurs)
# Exemple attendu : data/raw/sorted_movie_posters_paligema/Action/img1.jpg
SOURCE_DIR = project_root / "data/raw/sorted_movie_posters_paligema"

# 2. DESTINATION : Cible obligatoire pour que le script d'entrainement fonctionne
DEST_DIR = project_root / "data" / "dataset"

TRAIN_SPLIT = 0.8
RANDOM_SEED = 42
MOVE_INSTEAD_OF_COPY = False  # False = Copier (plus sûr), True = Déplacer
# -----------------------------

random.seed(RANDOM_SEED)

# Extensions reconnues
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}


def find_image_files(folder: Path):
    """Return a list of image Paths under `folder`, recursively."""
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


if not SOURCE_DIR.exists():
    raise SystemExit(f"Source directory not found: {SOURCE_DIR.resolve()}")

# Create destination train/val root folders
for split in ("train", "val"):
    (DEST_DIR / split).mkdir(parents=True, exist_ok=True)

# Each subfolder of SOURCE_DIR is treated as a genre
genres = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
if not genres:
    print(
        f"Aucun sous-dossier (genre) trouvé dans {SOURCE_DIR.resolve()}."
        " Vérifie la structure."
    )
else:
    print(f"Genres trouvés: {[g.name for g in genres]}")

for genre_dir in genres:
    print(f"\nTraitement du genre: {genre_dir.name}")
    images = find_image_files(genre_dir)
    print(f"  -> Images trouvées (récursif): {len(images)}")

    if not images:
        print(f"  !!! Aucun fichier image trouvé dans {genre_dir.resolve()}.")
        continue

    random.shuffle(images)
    split_idx = int(len(images) * TRAIN_SPLIT)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    train_target = DEST_DIR / "train" / genre_dir.name
    val_target = DEST_DIR / "val" / genre_dir.name
    train_target.mkdir(parents=True, exist_ok=True)
    val_target.mkdir(parents=True, exist_ok=True)

    def copy_list(img_list, target_dir):
        for i, src_path in enumerate(img_list):
            dst_name = f"{i:04d}_{src_path.name}"
            dst_path = target_dir / dst_name
            try:
                if MOVE_INSTEAD_OF_COPY:
                    shutil.move(str(src_path), str(dst_path))
                else:
                    shutil.copy2(str(src_path), str(dst_path))
            except Exception as e:
                print(f"    Erreur en copiant {src_path} -> {dst_path}: {e}")

    copy_list(train_imgs, train_target)
    copy_list(val_imgs, val_target)

    print(f"  [OK] {genre_dir.name}: {len(train_imgs)} train, {len(val_imgs)} val")

