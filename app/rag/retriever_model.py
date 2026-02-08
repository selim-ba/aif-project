# part 4 - app/rag/retriever_model.py

import numpy as np

class Retriever:
    def __init__(self, embed_model, annoy_index, id_map):
        """
        annoy_index: AnnoyIndex
        id_map: dict[int] -> (movie_id, modality)  # modality in {"plot","poster"}
        """
        self.embed_model = embed_model
        self.annoy = annoy_index
        self.id_map = id_map

    def search_best(self, query: str, top_k: int = 20):
        q = self.embed_model.get_text_embeddings([query])[0]
        q = q.detach().cpu().numpy().astype(np.float32)

        ids, dists = self.annoy.get_nns_by_vector(q, top_k, include_distances=True)

        # return raw hits
        return [(*self.id_map[i], i, d) for i, d in zip(ids, dists)]
        # -> (movie_id, modality, annoy_id, dist)
