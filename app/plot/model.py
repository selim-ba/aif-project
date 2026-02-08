import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from annoy import AnnoyIndex
from app.plot import config

# Helper functions

def process_text(text: str,tokenizer):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=config.MAX_LEN
    )
    return inputs['input_ids'].to("cpu"), inputs['attention_mask'].to("cpu")

def predict_genre_logic(plot: str,model: nn.Module , tokenizer ,id2label: dict):
    input_ids, mask = process_text(plot,tokenizer=tokenizer)
    
    with torch.no_grad():
        logits, _, _ = model(input_ids, mask)
        prediction = torch.argmax(logits, dim=1).item()
        
    return {
        "predicted_genre": id2label[prediction],
        "label_id": prediction
    }