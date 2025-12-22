from pathlib import Path
import json
import joblib

from flask import Flask, jsonify, request
from PIL import Image

from app.posters.model import load_trained_model # resnet model
from app.posters.inference import preprocess_image, predict_genres #the inference pipeline
from scripts.train_ood import FeatureExtractor #feature extractor class
from app.validation.inference_ood import get_features #feature extraction function

app = Flask(__name__) #creation of the Flask application instance

# Loading paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = MODELS_DIR / "genres.json"

# Load classes : {"classes": ["action", "animation", "comedy", ...]}
with GENRES_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)
CLASSES = data["classes"] #so that : CLASSES = ['action', 'animation', 'comedy', ...]

# Loads model once when the server starts (so that it's not reloaded for every request)
MODEL = load_trained_model(str(WEIGHTS_PATH), num_classes=len(CLASSES), device="cpu")

#load OOD model
try:
    OOD_DETECTOR = joblib.load(MODELS_DIR / "ood_detector.joblib")
    print("Détecteur OOD chargé.")
except:
    print("Attention: 'ood_detector.joblib' introuvable.")  
    OOD_DETECTOR = None

FEATURE_EXTRACTOR_MODEL = FeatureExtractor(MODEL)
FEATURE_EXTRACTOR_MODEL.to("cpu")
FEATURE_EXTRACTOR_MODEL.eval()

# Sanity check, it should return {"status": "ok"} with HTTP code 200
@app.route("/health", methods=["GET"])
def health():
    """ health check."""
    return jsonify({"status": "ok"}), 200

# Main prediction route
@app.route("/api/predict_poster_genre", methods=["POST"])
def predict_poster_genre():
    """
    Accepts an image file and returns top-k predicted genres.

    Expected request:
      - Content-Type: multipart/form-data
      - Field name: "file"

    Response:
      {
        "predictions": [
          {"genre": "comedy", "score": 0.73},
          ...
        ]
      }
    """
    if "file" not in request.files:
        """
          Client must send :
            multipart/form-data (http request format, tells the server that the request contains multiple parts, and some of them are binary files)
            file =<image>
        """
        return jsonify({"error": "No file uploaded. Expected field 'file'."}), 400


    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400 #prevens empty upload

    # guards against, non-image files, corrupt images or unsupported formats
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Unable to read image: {e}"}), 400

    # here we run the ML inference
    tensor = preprocess_image(image) #applies training transforms
    predictions = predict_genres(MODEL, tensor, CLASSES, top_k=3) #applies torch.softmax and returns top 3 best predictions
    # labels are mapped with CLASSES
    return jsonify({"predictions": predictions}), 200 #json format response

#OOD route
@app.route("/api/check_is_poster", methods=["POST"])
def check_is_poster():
    """
    Route pour vérifier si l'image est un poster.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    if OOD_DETECTOR is None:
        return jsonify({"error": "OOD model not loaded"}), 500

    try:
        features = get_features(file,FEATURE_EXTRACTOR_MODEL).reshape(1, -1) #reshaping to fit the model input
        # 1 = Inlier (Poster), -1 = Outlier (Pas un poster)
        prediction = OOD_DETECTOR.predict(features)[0]
        # On peut aussi récupérer le score d'anomalie (plus c'est bas, plus c'est anormal)
        score = OOD_DETECTOR.decision_function(features)[0]

        is_poster = True if prediction == 1 else False
        
        return jsonify({
            "is_poster": is_poster,
            "anomaly_score": float(score),
            "message": "It is a poster!" if is_poster else "This doesn't look like a poster."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=8000, debug=True) 

