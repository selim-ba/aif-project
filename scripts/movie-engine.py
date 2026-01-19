import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from annoy import AnnoyIndex
import os

# 1. Data Configuration
CSV_FILE = 'movie_plots.csv'
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 256   # Max sequence length for plots
BATCH_SIZE = 16
EPOCHS = 2      # To increase if we want better results (only on one of our PCs with a good GPU, but time consuming)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load and Preprocess Data
def load_data(filepath):
    df = pd.read_csv(filepath)
    
    # Create Label Mappings

    unique_labels = df['movie_category'].unique().tolist()
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    df['label_id'] = df['movie_category'].map(label2id)
    return df, label2id, id2label

# 3. Dataset Class (like we did in practical sessions)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class MoviePlotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.to_list()
        self.labels = labels.to_list()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encodings = tokenizer(
            self.texts, 
            truncation=True, 
            padding=True, 
            max_length=self.max_len
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# 4. Model Definition
class BertClf(nn.Module):
    def __init__(self, distilbert_model):
        super(BertClf, self).__init__()
        self.distilbert = distilbert_model
        
        # Freeze parameters except the classifier
        for name, param in distilbert_model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(self, input_ids, mask):
        out = self.distilbert(input_ids, attention_mask=mask)
        return out.logits, out.hidden_states, out.attentions

# 5. Training Function
def train_bert(model, optimizer, dataloader, epochs, criterion):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        t = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(t):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            preds, _, _ = model(input_ids, mask=attention_mask)
            loss = criterion(preds, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            t.set_description(f"Epoch {epoch+1} Loss: {running_loss / (i+1):.4f}")

# Main Execution Block for Training
if __name__ == "__main__":
    print("Loading Data...")
    df, label2id, id2label = load_data(CSV_FILE)
    print(f"Found {len(df)} movies and {len(label2id)} genres.")

    # Split Data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['movie_plot'], df['label_id'], test_size=0.1, random_state=42
    )

    # Create Datasets
    train_dataset = MoviePlotDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_dataset = MoviePlotDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    base_model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        output_attentions=True,
        output_hidden_states=True
    )
    model = BertClf(base_model).to(DEVICE)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4) # We can use a higher LR since we froze the body already
    criterion = nn.CrossEntropyLoss()

    # Train
    print("Starting Training...")
    train_bert(model, optimizer, train_loader, EPOCHS, criterion)
    
    # Save Model State
    torch.save(model.state_dict(), "../models/part3_model_weights.pth")
    print("Model saved to ../models/part3_model_weights.pth")

# 6. Embedding Extraction Function
def get_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Embeddings"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            # Forward pass
            _, hidden_states, _ = model(input_ids, mask=attention_mask)
            
            # hidden_states is a tuple, -1 is the last layer
            last_layer_cls = hidden_states[-1][:, 0, :] 
            embeddings.append(last_layer_cls.cpu().numpy())
            
    return np.vstack(embeddings)

if __name__ == "__main__":
    # Create a full dataset loader for indexing
    full_dataset = MoviePlotDataset(df['movie_plot'], df['label_id'], tokenizer, MAX_LEN)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Get Embeddings
    print("Computing Embeddings for Annoy Index...")
    plot_embeddings = get_embeddings(model, full_loader)

    # Build Annoy Index
    embedding_dim = 768  # Dimension of DistilBert
    annoy_index = AnnoyIndex(embedding_dim, 'angular') # Seems like Angular is a good distance for text

    for i, vector in enumerate(plot_embeddings):
        annoy_index.add_item(i, vector)

    annoy_index.build(10)
    annoy_index.save('../models/part3_movie_index.ann')
    print("Annoy index saved to ../models/part3_movie_index.ann")
    
    # Save dataframe mapping for API retrieval
    df.reset_index(drop=True).to_pickle("../models/part3_movie_brochure.pkl")
