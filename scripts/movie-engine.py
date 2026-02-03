import sys
import os
from pathlib import Path

# --- FIX DES CHEMINS (Logique Infaillible) ---
# 1. Chemin absolu du fichier actuel (movie-engine.py)
current_file = Path(__file__).resolve()

# 2. On identifie la racine 
if current_file.parent.name == 'scripts':
    project_root = current_file.parent.parent
else:
    project_root = current_file.parent

# 3. Ajout au Path pour les imports
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

print(f"📍 Script: {current_file}")
print(f"📍 RACINE PROJET DÉTECTÉE: {project_root}")
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from annoy import AnnoyIndex
import json
import pickle

# Import de la classe partagée
try:
    from app.nlp.nlp_model import BertClf
except ImportError as e:
    print("\n❌ ERREUR D'IMPORT: Impossible de trouver 'app.nlp.nlp_model'.")
    print(f"   Vérifiez que le fichier 'app/nlp/nlp_model.py' existe bien dans : {project_root}")
    sys.exit(1)

# --- CONFIGURATION DES DOSSIERS ---
# On force l'utilisation des dossiers à la racine
DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models"

# Création des dossiers racine si inexistants
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / 'movie_plots.csv'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔧 Device: {DEVICE}")
print(f"📂 Fichier CSV attendu ici: {CSV_FILE}")

# 1. Dataset Class
class MoviePlotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [str(t) for t in texts]
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Appel direct (plus robuste que encode_plus sur certaines versions)
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Training Loop
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)

        logits, _ = model(input_ids, attention_mask)
        _, preds = torch.max(logits, dim=1)
        loss = loss_fn(logits, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# 3. Embedding Generation
def get_embeddings(model, data_loader, device):
    model = model.eval()
    embeddings = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Generating Embeddings"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            
            _, cls_token = model(input_ids, attention_mask)
            embeddings.append(cls_token.cpu().numpy())
            
    return np.vstack(embeddings)

# --- MAIN ---
if __name__ == "__main__":
    if not CSV_FILE.exists():
        print(f"\n❌ ERREUR FATALE : Le fichier CSV est introuvable !")
        print(f"   Chemin cherché : {CSV_FILE}")
        print("   Action : Copiez 'movie_plots.csv' dans le dossier 'data' à la racine du projet.")
        exit(1)

    print("✅ CSV trouvé. Chargement des données...")
    df = pd.read_csv(CSV_FILE)
    
    # Label Mapping
    unique_labels = df['movie_category'].unique().tolist()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    df['label_id'] = df['movie_category'].map(label2id)
    print(f"Classes détectées ({len(label2id)}): {list(label2id.keys())[:5]}...")

    # Tokenizer
    print("Chargement Tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Split
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    train_dataset = MoviePlotDataset(df_train['movie_plot'].values, df_train['label_id'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Init Model
    print("Initialisation Modèle...")
    model = BertClf(MODEL_NAME, num_labels=len(unique_labels))
    model = model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)

    # Training
    print("\n=== DÉBUT ENTRAÎNEMENT ===")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, DEVICE, len(df_train))
        print(f"Train loss {train_loss:.4f} accuracy {train_acc:.4f}")

    # --- SAVING ARTIFACTS ---
    print("\n=== SAUVEGARDE DES RÉSULTATS ===")
    
    # 1. Weights
    save_path_weights = MODELS_DIR / "part3_nlp_weights.pth"
    torch.save(model.state_dict(), save_path_weights)
    print(f"✅ Poids sauvegardés : {save_path_weights}")

    # 2. Labels Mapping
    save_path_classes = MODELS_DIR / "part3_classes.json"
    with open(save_path_classes, "w") as f:
        json.dump(id2label, f)
    print(f"✅ Classes sauvegardées : {save_path_classes}")

    # 3. Annoy Index
    print("Construction de l'Index Annoy...")
    full_dataset = MoviePlotDataset(df['movie_plot'].values, df['label_id'].values, tokenizer, MAX_LEN)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings = get_embeddings(model, full_loader, DEVICE)
    
    annoy_dim = embeddings.shape[1]
    t = AnnoyIndex(annoy_dim, 'angular')
    for i, vector in enumerate(embeddings):
        t.add_item(i, vector)
        
    t.build(10)
    save_path_ann = MODELS_DIR / 'part3_plot.ann'
    t.save(str(save_path_ann))
    print(f"✅ Index Annoy sauvegardé : {save_path_ann}")
    print("\n🎉 TERMINÉ ! Vous pouvez relancer 'main.py'.")