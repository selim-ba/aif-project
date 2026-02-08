# app/validation/inference_ood.py

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
    Meme preprocessing que pendant l'entraînement, retourne un tenseur [1, C, H, W].
    """
    tensor = _transform(img)
    return tensor.unsqueeze(0)

def get_features(image_file,feature_extractor_model) -> List[float]:
    """
    Prend un fichier image (binaire), le prépare et retourne son vecteur de features.
    """
    image = Image.open(image_file).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to("cpu")

    # (Inférence, méthode forward)
    with torch.no_grad(): # Désactive le calcul des gradients (économise mémoire et calcul)
        features_vector = feature_extractor_model(image_tensor)
        
    return features_vector.cpu().numpy()