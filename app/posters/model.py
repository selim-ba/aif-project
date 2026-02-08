# app/posters/model.py

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    """
    Classifeur simple, basé sur ResNet50 pré-entraîné sur ImageNet.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained_model(weights_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    """
    Pour charger un modèle entraîné avec des poids adaptés au CPU.
    """
    model = build_model(num_classes)
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

