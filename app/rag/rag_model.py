import numpy as np
from typing import Dict, List, Tuple, Optional

from app.rag.foundation_model import FoundationModel
from app.rag.embedding_model import EmbeddingModel

class RAG:
    def __init__(
        self,
        CONFIG: dict,
        annoy_index,                     # AnnoyIndex already built/loaded
        id_map: Dict[int, Tuple[int,str]],# annoy_id -> (movie_id, "plot"/"poster")
        movies: Dict[int, dict],          # movie_id -> metadata (title, plot, poster_url/path, etc.)
    ):
        # Chat model (your updated chat-capable FoundationModel)
        self.foundation_model = FoundationModel(
            FOUND_MODEL_PATH=CONFIG["FOUND_MODEL_PATH"],
            SYSTEM_PROMPT=CONFIG.get("SYSTEM_PROMPT", None) or
                "You are a movie recommendation assistant. Ask clarifying questions when needed "
                "and recommend a small set of movies with brief reasons."
        )

        # CLIP embedder (your final EmbeddingModel)
        self.embedding_model = EmbeddingModel(
            model_id=CONFIG.get("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        )

        # Single Annoy index + mapping
        self.annoy = annoy_index
        self.id_map = id_map
        self.movies = movies

    def reset_chat(self):
        self.foundation_model.reset()

    def _retrieve(self, query: str, top_k: int = 30) -> List[Tuple[int, str, float]]:
        # 1. Encodage de la requête (Conversion Tensor -> Numpy explicite)
        q = self.embedding_model.get_text_embeddings([query])[0]
        # Sécurité pour Annoy : détacher du GPU et convertir en numpy float32
        if hasattr(q, "detach"):
             q = q.detach().cpu().numpy()
        q = q.astype(np.float32)
        
        # 2. Recherche Annoy
        ids, dists = self.annoy.get_nns_by_vector(q, top_k, include_distances=True)
        
        hits = []
        for annoy_id, dist in zip(ids, dists):
            # A. Récupération des métadonnées (Gérer Clé Int vs Str)
            # id_map keys sont souvent des strings dans le JSON ("0", "1"...)
            meta = self.id_map.get(annoy_id) or self.id_map.get(str(annoy_id))
            
            if not meta:
                continue # ID Annoy orphelin (ne devrait pas arriver)

            movie_id = meta["row_index"] # C'est un int (ex: 105)
            modality = meta["modality"]

            # B. Récupération du film (LE CORRECTIF EST ICI)
            # On cherche d'abord avec l'ID tel quel, puis en string, puis en int
            movie_data = self.movies.get(movie_id)
            if not movie_data:
                movie_data = self.movies.get(str(movie_id))
            if not movie_data:
                # Si toujours rien, c'est que l'ID n'est pas dans la base movies
                continue 

            # C. On a trouvé le film, on l'ajoute
            # Note: on garde movie_id tel qu'il est dans meta pour la cohérence
            hits.append((movie_id, modality, float(dist)))
            
        return hits

    def _dedupe_movies(
        self,
        hits: List[Tuple[int, str, float]],
        max_movies: int = 8
    ) -> List[Tuple[int, float, List[str]]]:
        """
        Dedupe plot/poster hits into per-movie score:
        returns list of (movie_id, best_dist, modalities_seen)
        """
        best_dist = {}
        mods = {}
        for movie_id, modality, dist in hits:
            if (movie_id not in best_dist) or (dist < best_dist[movie_id]):
                best_dist[movie_id] = dist
            mods.setdefault(movie_id, set()).add(modality)

        ranked = sorted(best_dist.items(), key=lambda x: x[1])[:max_movies]
        return [(mid, d, sorted(list(mods[mid]))) for mid, d in ranked]

    def _build_context(self, ranked_movies) -> List[str]:
        """Turn retrieved movies into context strings for the LLM."""
        ctx = []
        for movie_id, dist, modalities in ranked_movies:
            m = self.movies[movie_id]
            ctx.append(
                f"Title: {m.get('title','')}\n"
                f"Plot: {m.get('plot','')}\n"
                f"Poster: {m.get('poster_url','') or m.get('poster_path','')}\n"
                f"Matched via: {', '.join(modalities)}\n"
                f"(annoy_dist={dist:.4f})"
            )
        return ctx

    def ask(self, query: str, top_k: int = 30, max_movies: int = 8) -> str:
        hits = self._retrieve(query, top_k=top_k)
        ranked = self._dedupe_movies(hits, max_movies=max_movies)
        context = self._build_context(ranked)

        # Multi-turn refinement happens automatically via FoundationModel.history
        return self.foundation_model.chat(query, retrieved_context=context)
