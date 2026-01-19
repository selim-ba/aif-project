# config.py: to hold variable values in one place only.

import torch

MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PATH_BROCHURE = "models/part3_movie_brochure.pkl"
PATH_WEIGHTS = "models/part3_model_weights.pth"
PATH_INDEX = "models/part3_movie_index.ann"