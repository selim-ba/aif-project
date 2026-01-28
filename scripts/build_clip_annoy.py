#!/usr/bin/env python3
"""
Build a SINGLE Annoy index containing BOTH:
- plot embeddings (CLIP text encoder)
- poster embeddings (CLIP image encoder)

CSV columns expected (your format):
- movie_poster_path  (e.g. "action/100108.jpg")
- movie_plot
- movie_category

Index item IDs:
- annoy_id = 2*row_index     -> plot vector
- annoy_id = 2*row_index + 1 -> poster vector

Outputs:
- .ann annoy index
- id_map.json mapping annoy_id -> metadata (row_index, modality, poster_path, category)
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CLIPEmbedder:
    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.dim = self.model.config.projection_dim  # usually 512

    @torch.inference_mode()
    def embed_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, p=2, dim=-1)

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, p=2, dim=-1)


def load_image(abs_path: str) -> Optional[Image.Image]:
    try:
        return Image.open(abs_path).convert("RGB")
    except Exception:
        return None


def batched_ranges(n: int, batch_size: int):
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        yield s, e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to your CSV file")
    parser.add_argument("--posters-root", required=True, help="Root folder that contains genre subfolders")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--metric", default="angular", choices=["angular", "euclidean", "manhattan", "hamming", "dot"])
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-index", required=True, help="Output annoy index path (e.g. movies_clip.ann)")
    parser.add_argument("--out-map", required=True, help="Output id_map json path (e.g. id_map.json)")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug: cap number of rows (0 = all)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    required = ["movie_poster_path", "movie_plot", "movie_category"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing column '{c}'. Found columns: {list(df.columns)}")

    posters_root = args.posters_root
    device = pick_device()
    clip = CLIPEmbedder(args.clip_model, device)

    annoy = AnnoyIndex(clip.dim, args.metric)
    id_map = {}  # str(annoy_id) -> metadata

    n = len(df)
    bs = max(1, args.batch_size)

    for start, end in batched_ranges(n, bs):
        batch = df.iloc[start:end]

        plots = batch["movie_plot"].fillna("").astype(str).tolist()
        poster_rel = batch["movie_poster_path"].fillna("").astype(str).tolist()
        cats = batch["movie_category"].fillna("").astype(str).tolist()

        # ---- Plot embeddings (batched) ----
        plot_embs = clip.embed_text(plots)  # (B, dim)

        # ---- Poster embeddings (load what exists, batched) ----
        images = []
        img_row_idxs = []  # index within batch
        abs_paths = []

        for j, rel in enumerate(poster_rel):
            rel = rel.strip()
            if not rel:
                continue
            abs_path = os.path.join(posters_root, rel)
            img = load_image(abs_path)
            if img is not None:
                images.append(img)
                img_row_idxs.append(j)
                abs_paths.append(abs_path)

        poster_embs = clip.embed_images(images) if images else None
        row_to_k = {j: k for k, j in enumerate(img_row_idxs)}  # faster than list.index

        # ---- Add to Annoy ----
        for j in range(end - start):
            row_idx = start + j
            plot_annoy_id = 2 * row_idx
            poster_annoy_id = 2 * row_idx + 1

            # plot vector always added
            v_plot = plot_embs[j].detach().cpu().numpy().astype(np.float32)
            annoy.add_item(plot_annoy_id, v_plot)
            id_map[str(plot_annoy_id)] = {
                "row_index": row_idx,
                "modality": "plot",
                "movie_poster_path": poster_rel[j],
                "movie_category": cats[j],
            }

            # poster vector if image loaded
            if poster_embs is not None and j in row_to_k:
                k = row_to_k[j]
                v_img = poster_embs[k].detach().cpu().numpy().astype(np.float32)
                annoy.add_item(poster_annoy_id, v_img)
                id_map[str(poster_annoy_id)] = {
                    "row_index": row_idx,
                    "modality": "poster",
                    "movie_poster_path": poster_rel[j],
                    "movie_category": cats[j],
                }

        print(f"Processed rows {start}..{end-1}")

    print("Building Annoy trees...")
    annoy.build(args.n_trees)

    os.makedirs(os.path.dirname(args.out_index) or ".", exist_ok=True)
    annoy.save(args.out_index)

    with open(args.out_map, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    print(f"Saved index: {args.out_index}")
    print(f"Saved map:   {args.out_map}")
    print(f"dim={clip.dim}, metric={args.metric}, trees={args.n_trees}")


if __name__ == "__main__":
    main()
