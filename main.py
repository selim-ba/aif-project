from pathlib import Path
import json

from flask import Flask, jsonify, request
from PIL import Image

from app.posters.model import load_trained_model # resnet model
from app.posters.inference import preprocess_image, predict_genres #the inference pipeline

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


if __name__ == "__main__":
    # For local dev only
    app.run(host="0.0.0.0", port=8000, debug=True)

