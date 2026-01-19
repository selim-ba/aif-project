import sys
import os
from pathlib import Path

# Récupère le chemin du dossier parent (racine du projet)
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
from app.posters.model import load_trained_model
from scripts.train_posters_cnn import get_device, load_datasets, get_transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from app.validation.feature_extractor import FeatureExtractor
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
    
# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
VAL_OOD = DATASET_DIR / "val_OOD"

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = MODELS_DIR / "genres.json"

BATCH_SIZE = 32
# ---------------

def get_ood_score(knn_model, feature_test):
    distances, _ = knn_model.kneighbors(feature_test.reshape(1, -1))
    return np.max(distances)


def main():
    device = get_device()
    print("Using device:", device)

    train_data, val_data, train_loader, val_loader = load_datasets()
    transform = get_transforms()
    ood_data= datasets.ImageFolder(str(VAL_OOD), transform=transform)
    ood_loader = DataLoader(ood_data, batch_size=BATCH_SIZE, shuffle=True)
    
    num_classes = len(train_data.classes)
 
    # 1. Charger le modèle pré-entraîné
    model = load_trained_model(str(WEIGHTS_PATH), num_classes=num_classes, device="cpu")
    feature_extractor = FeatureExtractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()

    print("Extraction des features en cours...")
    features_list = []
    with torch.no_grad():
        for images,_ in train_loader:
                
                images = images.to(device)

                # Extraction du vecteur
                emb_batch = feature_extractor(images)

                features_list.append(emb_batch.cpu().numpy())

    if len(features_list) > 0:
        features_array = np.concatenate(features_list, axis=0)
        print(f"Extraction terminée ! Forme des features : {features_array.shape}")
    else:
        print("Erreur : Aucune feature extraite. Vérifiez votre DataLoader.")

    # 4. Entraîner le modèle OOD (OneClassSVM)
    clf = make_pipeline(
        StandardScaler(),
        # On utilise la distance Euclidienne (L2)
        NearestNeighbors(n_neighbors=5, algorithm='auto', metric='minkowski')
    )

    print(f"Entraînement Nearest Neighbors sur {features_array.shape[0]} images...")
    clf.fit(features_array)

    id_scores = []
    ood_scores = []

    with torch.no_grad():
        # 1. Scores pour les données normales (ID)
        for images, _ in val_loader:
            images = images.to(device)

            # Extraction du vecteur
            emb_batch = feature_extractor(images).cpu().numpy()
            scaled_features = clf[:-1].transform(emb_batch)
            dist, _ = clf[-1].kneighbors(scaled_features)
            id_scores.extend(np.max(dist, axis=1))

        # 2. Scores pour les données anormales (OOD)
        for images,_ in ood_loader:
            images = images.to(device)

            # Extraction du vecteur
            emb_batch = feature_extractor(images).cpu().numpy()
            scaled_features = clf[:-1].transform(emb_batch)
            dist, _ = clf[-1].kneighbors(scaled_features)
            ood_scores.extend(np.max(dist, axis=1))

    # 3. Préparation des labels (0 pour ID, 1 pour OOD)
    y_true = np.array([0] * len(id_scores) + [1] * len(ood_scores))
    all_scores = np.array(id_scores + ood_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, all_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # 5. Sélection du seuil qui maximise la précision (en évitant recall=0)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
    
    print(f"Meilleur Seuil (F1): {best_threshold:.4f}")
    print(f"Précision: {precisions[best_idx]:.4f}, Rappel: {recalls[best_idx]:.4f}")
    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=50, alpha=0.5, label='Films (In-Distribution)', color='blue')
    plt.hist(ood_scores, bins=50, alpha=0.5, label='Chats (Out-of-Distribution)', color='red')
    plt.axvline(best_threshold, color='green', linestyle='--', label=f'Seuil optimal: {best_threshold:.2f}')
    plt.title("Distribution des distances DKNN")
    plt.xlabel("Distance moyenne (Score OOD)")
    plt.ylabel("Nombre d'images")
    plt.legend()
    plt.savefig("models/ood_distribution.png")
    plt.show()
    
    ood_package = {
    'pipeline': clf,      # Contient le StandardScaler et le NearestNeighbors
    'threshold': best_threshold,
    'metadata': {
        'method': 'DKNN',
        'k': 5,
        'precision_at_threshold': precisions[best_idx]
    }
    }

    joblib.dump(ood_package, str(MODELS_DIR / "ood_detector.joblib"))
    print(f"Modèle OOD sauvegardé sous {MODELS_DIR / 'ood_detector.joblib'}")

if __name__ == "__main__":
    main()
