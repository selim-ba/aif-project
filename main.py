from pathlib import Path
import json
import joblib

from flask import Flask, jsonify, request
from PIL import Image
from pydantic import ValidationError
import torch
import torch.nn as nn
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from annoy import AnnoyIndex

from app.plot.model import predict_genre_logic
from app.plot.schemas import PlotRequest
from app.posters.model import load_trained_model # resnet model
from app.posters.inference import preprocess_image, predict_genres #the inference pipeline
from app.validation.feature_extractor import FeatureExtractor #feature extraction model
from app.validation.inference_ood import get_features #feature extraction function
from app.plot import config

app = Flask(__name__) #creation of the Flask application instance 

# Loading paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "movie_genre_cpu.pt"
GENRES_PATH = PROJECT_ROOT / "app/posters/genres.json"

# Load classes : {"classes": ["action", "animation", "comedy", ...]}
with GENRES_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)
CLASSES = data["classes"] #so that : CLASSES = ['action', 'animation', 'comedy', ...]

# Loads classification model once when the server starts (so that it's not reloaded for every request)
try :
    print("Chargement du modèle de classification des posters...")
    MODEL = load_trained_model(str(WEIGHTS_PATH), num_classes=len(CLASSES), device="cpu")
    print("Modèle de classification des posters chargé avec succès.")
except:
    print(f"Erreur lors du chargement du modèle de classification des posters depuis {WEIGHTS_PATH}")
    MODEL = None

#Load OOD model and feature extractor
try:
    print("Chargement du détecteur OOD...")
    OOD_DETECTOR = joblib.load(MODELS_DIR / "ood_detector.joblib")
    print("Détecteur OOD chargé.")
except:
    print("Attention: 'ood_detector.joblib' introuvable.")  
    OOD_DETECTOR = None

FEATURE_EXTRACTOR_MODEL = FeatureExtractor(MODEL)
FEATURE_EXTRACTOR_MODEL.to("cpu")
FEATURE_EXTRACTOR_MODEL.eval()

# Load classification model of plots
# Model Definition
class BertClf(nn.Module):
    def __init__(self, distilbert_model):
        super(BertClf, self).__init__()
        self.distilbert = distilbert_model
        
    def forward(self, input_ids, mask):
        out = self.distilbert(input_ids, attention_mask=mask)
        return out.logits, out.hidden_states, out.attentions

# Load resources

print('Chargement du modèle NLP') # test print to ensure everything is going right

df = pd.read_pickle(config.PATH_BROCHURE)
id2label = {i: cat for i, cat in enumerate(df['movie_category'].unique())}
label2id = {cat: i for i, cat in enumerate(df['movie_category'].unique())}

TOKENIZER = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)

base_model = DistilBertForSequenceClassification.from_pretrained(
    config.MODEL_NAME, 
    num_labels=len(label2id),
    output_attentions=False, 
    output_hidden_states=False
)
print("Modèle de base DistilBERT chargé.")
MODEL_NLP = BertClf(base_model)
MODEL_NLP.load_state_dict(torch.load(config.PATH_WEIGHTS, map_location="cpu"))
MODEL_NLP.to("cpu")
MODEL_NLP.eval()

annoy_index = AnnoyIndex(768, 'angular')
annoy_index.load(config.PATH_INDEX)

print("Modèle NLP chargé avec succès.")

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
        pipeline = OOD_DETECTOR['pipeline']
        threshold = OOD_DETECTOR['threshold']

        features = get_features(file, FEATURE_EXTRACTOR_MODEL)
        scaled_features = pipeline[:-1].transform(features)
        distances, _ = pipeline[-1].kneighbors(scaled_features)
        
        max_distance = float(distances.max())

        # 5. Décision basée sur le seuil
        is_poster = True if max_distance <= threshold else False
        
        return jsonify({
            "is_poster": is_poster,
            "distance_score": max_distance,
            "threshold": threshold
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#NLP route
@app.route("/api/plot_predict_genre", methods=['POST'])
def predict_genre():
    """
    Flask route to predict the genre of a movie based on its plot.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data"}), 400
    try:
        # Will raise error if plot is missing
        req_data = PlotRequest(**data)
    except ValidationError as e:
        return jsonify(e.errors()), 422

    try:
        result = predict_genre_logic(req_data.plot,MODEL_NLP, TOKENIZER,id2label)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=8000, debug=True) 

