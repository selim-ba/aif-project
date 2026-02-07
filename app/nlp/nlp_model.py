# Fichier: app/nlp/nlp_model.py

import torch.nn as nn
from transformers import DistilBertModel

class BertClf(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        super(BertClf, self).__init__()
        # On charge le modèle de base ici pour faciliter l'init
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        
        # Optimisation : on gèle les poids de DistilBERT
        for name, param in self.distilbert.named_parameters():
            param.requires_grad = False

    def forward(self, input_ids, mask):
        # DistilBert retourne (last_hidden_state, ...)
        output = self.distilbert(input_ids, attention_mask=mask)
        cls_token = output.last_hidden_state[:, 0, :] # Le token [CLS] du début
        logits = self.classifier(cls_token)
        return logits, cls_token # On retourne aussi l'embedding pour Annoy