import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    """
    Build a ResNet50-based classifier for `num_classes` genres.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_trained_model(weights_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    """
    Load a trained model from a state_dict saved with CPU compatibility.
    """
    model = build_model(num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

