import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
from app.posters.model import load_trained_model
from scripts.train_posters_cnn import get_device, load_datasets
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from app.validation.feature_extractor import FeatureExtractor

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = MODELS_DIR / "genres.json"
# ---------------

def main():
    device = get_device()
    print("Using device:", device)

    train_data, val_data, train_loader, val_loader = load_datasets()
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
        OneClassSVM(kernel="rbf", nu=0.01, gamma='scale')
    )
    # nu=0.01 est l'équivalent de la contamination (environ)

    print(f"Entraînement OneClassSVM sur {features_array.shape[0]} images...")
    clf.fit(features_array)

    # 5. Sauvegarder
    joblib.dump(clf, str(MODELS_DIR / "ood_detector.joblib"))
    print(f"Modèle OOD sauvegardé sous {MODELS_DIR / 'ood_detector.joblib'}")

if __name__ == "__main__":
    main()
