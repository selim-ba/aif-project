import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from annoy import AnnoyIndex
import pandas as pd
from app.plot import config

# Model Definition
class BertClf(nn.Module):
    def __init__(self, distilbert_model):
        super(BertClf, self).__init__()
        self.distilbert = distilbert_model
        
    def forward(self, input_ids, mask):
        out = self.distilbert(input_ids, attention_mask=mask)
        return out.logits, out.hidden_states, out.attentions

# Load resources

print('Chargement du modèle...') # test print to ensure everything is going right

df = pd.read_pickle(config.PATH_BROCHURE)
id2label = {i: cat for i, cat in enumerate(df['movie_category'].unique())}
label2id = {cat: i for i, cat in enumerate(df['movie_category'].unique())}

tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)

base_model = DistilBertForSequenceClassification.from_pretrained(
    config.MODEL_NAME, 
    num_labels=len(label2id),
    output_attentions=True, 
    output_hidden_states=True
)

model = BertClf(base_model)
model.load_state_dict(torch.load(config.PATH_WEIGHTS, map_location=config.DEVICE))
model.to(config.DEVICE)
model.eval()

annoy_index = AnnoyIndex(768, 'angular')
annoy_index.load(config.PATH_INDEX)

print("Modèle chargé avec succès.")


# Helper functions

def process_text(text: str):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=config.MAX_LEN
    )
    return inputs['input_ids'].to(config.DEVICE), inputs['attention_mask'].to(config.DEVICE)

def predict_genre_logic(plot: str):
    input_ids, mask = process_text(plot)
    
    with torch.no_grad():
        logits, _, _ = model(input_ids, mask)
        prediction = torch.argmax(logits, dim=1).item()
        
    return {
        "predicted_genre": id2label[prediction],
        "label_id": prediction
    }