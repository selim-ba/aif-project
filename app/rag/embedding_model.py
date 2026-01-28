from typing import List
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

try:
    from PIL import Image
except ImportError:
    Image = None


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EmbeddingModel:
    """CLIP embedder for text (plots/queries) and images (posters) in the SAME space."""
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.device = pick_device()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.dim = self.model.config.projection_dim  # e.g., 512

    @torch.inference_mode()
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, p=2, dim=-1)

    @torch.inference_mode()
    def get_image_embeddings(self, images: List["Image.Image"]) -> torch.Tensor:
        if Image is None:
            raise RuntimeError("PIL not installed. pip install pillow")
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, p=2, dim=-1)

# usage
# EmbedModel = EmbeddingModel("openai/clip-vit-base-patch32")
#q = EmbedModel.get_text_embeddings(["movies like Blade Runner, rainy neon city"])
