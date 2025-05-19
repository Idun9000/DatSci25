import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts, cache_path=None, force_recompute=False):
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            return pd.read_csv(cache_path)

        embeddings = self.model.encode(texts, show_progress_bar=True, device=self.device)
        df = pd.DataFrame(embeddings)

        if cache_path:
            df.to_csv(cache_path, index=False)

        return df