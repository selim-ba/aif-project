from pathlib import Path
import json
import joblib

from flask import Flask, jsonify, request
from PIL import Image

from app.posters.model import load_trained_model
from app.posters.inference import preprocess_image, predict_genres
from app.validation.feature_extractor import FeatureExtractor
from app.validation.inference_ood import get_features

# ===== PART 4: RAG IMPORTS =====
from annoy import AnnoyIndex
from app.rag.rag_model import RAG
# ===============================

app = Flask(__name__)

# Loading paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = PROJECT_ROOT / "app/posters/genres.json"

# Load classes
with GENRES_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)
CLASSES = data["classes"]

# Load Classifier Model
MODEL = load_trained_model(str(WEIGHTS_PATH), num_classes=len(CLASSES), device="cpu")

# Load OOD model
try:
    OOD_DETECTOR = joblib.load(MODELS_DIR / "ood_detector.joblib")
    print("OOD detector loaded.")
except:
    print("Warning: 'ood_detector.joblib' not found.")
    OOD_DETECTOR = None

FEATURE_EXTRACTOR_MODEL = FeatureExtractor(MODEL)
FEATURE_EXTRACTOR_MODEL.to("cpu")
FEATURE_EXTRACTOR_MODEL.eval()


# ===== PART 4: RAG INITIALIZATION =====
RAG_MODEL = None

def init_rag():
    """Initialize the RAG model. Called once at startup."""
    global RAG_MODEL

    # Paths for RAG resources
    INDEX_PATH = DATA_DIR / "movies_clip.ann"
    ID_MAP_PATH = DATA_DIR / "id_map.json"
    MOVIES_PATH = DATA_DIR / "movies.json"

    # Check if files exist
    if not INDEX_PATH.exists():
        print(f"Warning: RAG index not found at {INDEX_PATH}")
        return False
    if not ID_MAP_PATH.exists():
        print(f"Warning: id_map.json not found at {ID_MAP_PATH}")
        return False
    if not MOVIES_PATH.exists():
        print(f"Warning: movies.json not found at {MOVIES_PATH}")
        return False

    try:
        # 1. Load Annoy index
        CLIP_DIM = 512
        annoy_index = AnnoyIndex(CLIP_DIM, 'angular')
        annoy_index.load(str(INDEX_PATH))
        print(f"Loaded Annoy index from {INDEX_PATH}")

        # 2. Load id_map AND CONVERT KEYS TO INT
        # AJOUT CORRECTIF : On convertit les clés str en int pour correspondre à Annoy
        with open(ID_MAP_PATH, "r", encoding="utf-8") as f:
            id_map_raw = json.load(f)
        id_map = {int(k): v for k, v in id_map_raw.items()}
        print(f"Loaded id_map with {len(id_map)} entries (keys converted to int)")

        # 3. Load movies metadata AND CONVERT KEYS TO INT
        # CORRECTIF (Déjà présent mais critique) :
        with open(MOVIES_PATH, "r", encoding="utf-8") as f:
            movies_raw = json.load(f)
        movies = {int(k): v for k, v in movies_raw.items()}
        print(f"Loaded {len(movies)} movies metadata (keys converted to int)")

        # RAG configuration
        CONFIG = {
            "FOUND_MODEL_PATH": "Qwen/Qwen3-0.6B",
            "CLIP_MODEL_ID": "openai/clip-vit-base-patch32",
            "SYSTEM_PROMPT": (
                "You are a friendly movie recommendation assistant. "
                "Based on the retrieved context, recommend relevant movies. "
                "Do not invent movies. If the context is empty, say you don't know."
            ),
        }

        RAG_MODEL = RAG(CONFIG, annoy_index, id_map, movies)
        print("RAG model initialized successfully!")
        return True

    except Exception as e:
        print(f"Error initializing RAG: {e}")
        import traceback
        traceback.print_exc()
        return False


# Initialize RAG at startup
print("Initializing RAG model...")
rag_initialized = init_rag()
if not rag_initialized:
    print("RAG model not available. Chat endpoints will return errors.")
# ======================================


# Sanity check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "rag_available": RAG_MODEL is not None
    }), 200


# Main prediction route
@app.route("/api/predict_poster_genre", methods=["POST"])
def predict_poster_genre():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to read image: {e}"}), 400

    tensor = preprocess_image(image)
    predictions = predict_genres(MODEL, tensor, CLASSES, top_k=3)
    return jsonify({"predictions": predictions}), 200


# OOD route
@app.route("/api/check_is_poster", methods=["POST"])
def check_is_poster():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if OOD_DETECTOR is None:
        return jsonify({"error": "OOD model not loaded"}), 500

    try:
        features = get_features(file, FEATURE_EXTRACTOR_MODEL).reshape(1, -1)
        prediction = OOD_DETECTOR.predict(features)[0]
        score = OOD_DETECTOR.decision_function(features)[0]
        is_poster = True if prediction == 1 else False

        return jsonify({
            "is_poster": is_poster,
            "anomaly_score": float(score),
            "message": "It is a poster!" if is_poster else "This doesn't look like a poster."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== PART 4: RAG CHAT ROUTES =====

@app.route("/api/chat", methods=["POST"])
def chat():
    if RAG_MODEL is None:
        return jsonify({"error": "RAG model not available.", "success": False}), 503

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query'.", "success": False}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Query cannot be empty.", "success": False}), 400

    try:
        response = RAG_MODEL.ask(query)
        return jsonify({"response": response, "success": True}), 200
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}", "success": False}), 500


@app.route("/api/reset_chat", methods=["POST"])
def reset_chat():
    if RAG_MODEL is None:
        return jsonify({"error": "RAG model not available.", "success": False}), 503
    try:
        RAG_MODEL.reset_chat()
        return jsonify({"message": "Conversation reset.", "success": True}), 200
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)