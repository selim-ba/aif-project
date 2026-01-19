from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from annoy import AnnoyIndex
import pandas as pd
import numpy as np

app = FastAPI(title="Movie Genre & Recommendation API")

# Configuration & Loading
MODEL_NAME = 'distilbert-base-uncased'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 256

# Load Dataframe for Metadata
df = pd.read_pickle("../models/part3_movie_brochure.pkl")
id2label = {i: cat for i, cat in enumerate(df['movie_category'].unique())}
label2id = {cat: i for i, cat in enumerate(df['movie_category'].unique())}

# Define Model Class
class BertClf(nn.Module):
    def __init__(self, distilbert_model):
        super(BertClf, self).__init__()
        self.distilbert = distilbert_model
    def forward(self, input_ids, mask):
        out = self.distilbert(input_ids, attention_mask=mask)
        return out.logits, out.hidden_states, out.attentions

# Load Tokenizer & Model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
base_model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(label2id),
    output_attentions=True, 
    output_hidden_states=True
)
model = BertClf(base_model)
model.load_state_dict(torch.load("../models/part3_model_weights.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load Annoy Index
annoy_index = AnnoyIndex(768, 'angular')
annoy_index.load('../models/part3_movie_index.ann')

# Request Models
class PlotRequest(BaseModel):
    plot: str

# Helper Function
def process_text(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )
    return inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE)

# API Routes

@app.post("/predict_genre")
async def predict_genre(request: PlotRequest):
    """
    Predicts the genre of a movie based on its plot description.
    """
    input_ids, mask = process_text(request.plot)
    
    with torch.no_grad():
        logits, _, _ = model(input_ids, mask)
        prediction = torch.argmax(logits, dim=1).item()
        
    return {
        "predicted_genre": id2label[prediction],
        "label_id": prediction
    }

# Run with: uvicorn app:app --reload
