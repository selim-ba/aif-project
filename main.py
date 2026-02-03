from pathlib import Path
import json
import joblib
import pickle
import torch
import numpy as np
import sys

# Ajout de la racine au path pour les imports locaux
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from flask import Flask, jsonify, request
from PIL import Image

# Imports Vision
from app.posters.model import load_trained_model
from app.posters.inference import preprocess_image, predict_genres
from app.validation.feature_extractor import FeatureExtractor

# Imports NLP (Part 3)
from transformers import DistilBertTokenizerFast
from app.nlp.nlp_model import BertClf

# Imports RAG (Part 4)
from annoy import AnnoyIndex
from app.rag.rag_model import RAG

app = Flask(__name__)

# --- CONFIGURATION DES CHEMINS ---
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ==============================================================================
# CONFIGURATION : TOUT EST DANS MODELS/
# ==============================================================================

# 1. VISION
VISION_WEIGHTS_PATH = MODELS_DIR / "vision_weights.pth"
VISION_CLASSES_PATH = MODELS_DIR / "genres.json"

# 2. NLP (Part 3)
NLP_WEIGHTS_PATH = MODELS_DIR / "part3_nlp_weights.pth"
NLP_CLASSES_PATH = MODELS_DIR / "part3_classes.json"

# 3. RAG (Part 4)
RAG_INDEX_PATH = MODELS_DIR / "part4_rag_index.ann"
RAG_MAP_PATH = MODELS_DIR / "part4_rag_map.json"
RAG_BROCHURE_PATH = MODELS_DIR / "part4_rag_brochure.pkl"


# --- CHARGEMENT DES CLASSES VISION ---
CLASSES = []
if VISION_CLASSES_PATH.exists():
    with VISION_CLASSES_PATH.open("r", encoding="utf-8") as f:
        CLASSES = json.load(f)["classes"]
else:
    # Fallback sur l'ancien emplacement
    OLD_GENRES_PATH = PROJECT_ROOT / "app/posters/genres.json"
    if OLD_GENRES_PATH.exists():
        with OLD_GENRES_PATH.open("r", encoding="utf-8") as f:
            CLASSES = json.load(f)["classes"]

# ==============================================================================
# 1. CHARGEMENT VISION (ResNet)
# ==============================================================================
MODEL = None
FEATURE_EXTRACTOR_MODEL = None
OOD_DETECTOR = None

try:
    if VISION_WEIGHTS_PATH.exists():
        print(f"👁️ Chargement Vision ({VISION_WEIGHTS_PATH.name})...")
        MODEL = load_trained_model(str(VISION_WEIGHTS_PATH), num_classes=len(CLASSES), device="cpu")
        
        FEATURE_EXTRACTOR_MODEL = FeatureExtractor(MODEL)
        FEATURE_EXTRACTOR_MODEL.to("cpu")
        FEATURE_EXTRACTOR_MODEL.eval()
        
        try:
            OOD_DETECTOR = joblib.load(MODELS_DIR / "ood_detector.joblib")
        except:
            OOD_DETECTOR = None
            
        print("✅ Vision chargée.")
    else:
        print(f"⚠️ Vision ignorée : '{VISION_WEIGHTS_PATH}' introuvable dans models/.")
except Exception as e:
    print(f"⚠️ Erreur chargement Vision : {e}")


# ==============================================================================
# 2. CHARGEMENT NLP (Part 3 - DistilBERT)
# ==============================================================================
NLP_MODEL = None
NLP_TOKENIZER = None
NLP_CLASSES = {}

try:
    if NLP_WEIGHTS_PATH.exists() and NLP_CLASSES_PATH.exists():
        print(f"🧠 Chargement NLP Part 3 ({NLP_WEIGHTS_PATH.name})...")
        
        with open(NLP_CLASSES_PATH, "r", encoding="utf-8") as f:
            NLP_CLASSES = json.load(f)
            NLP_CLASSES = {int(k): v for k, v in NLP_CLASSES.items()}

        NLP_MODEL = BertClf('distilbert-base-uncased', num_labels=len(NLP_CLASSES))
        NLP_MODEL.load_state_dict(torch.load(NLP_WEIGHTS_PATH, map_location=torch.device('cpu')))
        NLP_MODEL.to("cpu")
        NLP_MODEL.eval()
        
        NLP_TOKENIZER = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        print("✅ NLP Part 3 chargé.")
    else:
        print(f"⚠️ NLP ignoré : Fichiers 'part3_...' introuvables dans models/.")
except Exception as e:
    print(f"⚠️ Erreur NLP: {e}")


# ==============================================================================
# 3. CHARGEMENT RAG (Part 4 - Qwen + CLIP + Annoy)
# ==============================================================================
RAG_MODEL = None

def init_rag():
    global RAG_MODEL
    missing = []
    if not RAG_INDEX_PATH.exists(): missing.append(RAG_INDEX_PATH.name)
    if not RAG_BROCHURE_PATH.exists(): missing.append(RAG_BROCHURE_PATH.name)
    
    if missing:
        print(f"❌ RAG Erreur : Fichiers manquants dans models/ : {missing}")
        return False

    try:
        print("🤖 Initialisation RAG Part 4...")
        
        annoy_index = AnnoyIndex(512, 'angular')
        annoy_index.load(str(RAG_INDEX_PATH))

        with open(RAG_MAP_PATH, "r", encoding="utf-8") as f:
            id_map = {int(k): v for k, v in json.load(f).items()}

        with open(RAG_BROCHURE_PATH, "rb") as f:
            movies = pickle.load(f)

        CONFIG = {
            "FOUND_MODEL_PATH": "Qwen/Qwen3-0.6B",
            "CLIP_MODEL_ID": "openai/clip-vit-base-patch32",
            "SYSTEM_PROMPT": "You are a helpful movie assistant..."
        }

        RAG_MODEL = RAG(CONFIG, annoy_index, id_map, movies)
        print("✅ RAG Part 4 prêt !")
        return True

    except Exception as e:
        print(f"❌ Erreur RAG: {e}")
        return False

init_rag()


# ================= ROUTES API =================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "rag_part4": RAG_MODEL is not None,
        "nlp_part3": NLP_MODEL is not None,
        "vision": MODEL is not None
    }), 200

# --- CHAT ---
@app.route("/api/chat", methods=["POST"])
def chat():
    if RAG_MODEL is None: return jsonify({"error": "RAG not initialized"}), 503
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query: return jsonify({"error": "Empty query"}), 400
    try:
        print(f"User: {query}")
        return jsonify({"response": RAG_MODEL.ask(query), "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/reset_chat", methods=["POST"])
def reset_chat():
    if RAG_MODEL: RAG_MODEL.reset_chat()
    return jsonify({"success": True}), 200

# --- NLP ---
@app.route("/api/predict_plot_genre", methods=["POST"])
def predict_plot_genre():
    if NLP_MODEL is None: return jsonify({"error": "NLP Model not loaded"}), 503
    data = request.get_json()
    plot = data.get("plot", "").strip()
    if not plot: return jsonify({"error": "Empty plot"}), 400
    try:
        encoding = NLP_TOKENIZER(plot, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits, _ = NLP_MODEL(encoding['input_ids'].to("cpu"), encoding['attention_mask'].to("cpu"))
            probs = torch.nn.functional.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, 3)
        results = [{"genre": NLP_CLASSES.get(idx.item(), "Unknown"), "score": float(score)} for score, idx in zip(top_probs[0], top_indices[0])]
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- VISION ---
@app.route("/api/predict_poster_genre", methods=["POST"])
def predict_poster_genre():
    if MODEL is None: return jsonify({"error": "Vision Model not loaded"}), 503
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    try:
        image = Image.open(request.files['file'].stream).convert("RGB")
        input_tensor = preprocess_image(image).to("cpu")
        return jsonify({"predictions": predict_genres(MODEL, input_tensor, CLASSES)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/check_is_poster", methods=["POST"])
def check_is_poster():
    if OOD_DETECTOR is None: return jsonify({"error": "OOD Model not loaded"}), 503
    if 'file' not in request.files: return jsonify({"error": "No file"}), 400
    try:
        image = Image.open(request.files['file'].stream).convert("RGB")
        input_tensor = preprocess_image(image).to("cpu")
        with torch.no_grad():
            features = FEATURE_EXTRACTOR_MODEL(input_tensor).cpu().numpy()
        pipeline = OOD_DETECTOR['pipeline']
        dist = float(pipeline[-1].kneighbors(pipeline[:-1].transform(features))[0].max())
        is_poster = bool(dist <= OOD_DETECTOR['threshold'])
        return jsonify({"is_poster": is_poster, "anomaly_score": dist})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Serveur Flask (Mode PROD - Debug OFF)")
    app.run(host="0.0.0.0", port=8000, debug=False)