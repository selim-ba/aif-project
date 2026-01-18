from pathlib import Path
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1]) 
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) # Aplatir le vecteur
