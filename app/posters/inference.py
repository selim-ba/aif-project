# app/posters/inference.py

from typing import List, Dict
import torch
from PIL import Image
from torchvision import transforms

_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Meme processing que pendant l'entraînement, retourne un tenseur [1, C, H, W].
    """
    tensor = _transform(img)
    return tensor.unsqueeze(0)


def predict_genres(model, tensor: torch.Tensor, labels: List[str], top_k: int = 3) -> List[Dict]:
    """
    Inference sur un tenseur unique et retour les top-k genres (k fixé à 3 par défaut).
    """
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(labels)))
    return [
        {"genre": labels[int(idx)], "score": float(p)}
        for p, idx in zip(top_probs, top_indices)
    ]

