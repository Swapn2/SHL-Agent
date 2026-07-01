"""
Stage 1: Retrieval layer.

Embeds the cleaned catalog once (name + description + categories + job_levels)
using a local sentence-transformers model, then does cosine-similarity search
at query time. No external vector DB needed at this scale (370 items).
"""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

# Project root = parent of the app/ package, so paths work regardless of
# the working directory uvicorn is launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_embedding_text(entry: dict) -> str:
    parts = [
        entry["name"],
        entry["description"],
        "Categories: " + ", ".join(entry.get("categories", [])),
    ]
    if entry.get("job_levels"):
        parts.append("Job levels: " + ", ".join(entry["job_levels"]))
    return " | ".join(p for p in parts if p)


class CatalogIndex:
    """Loads catalog + embeddings once, then serves fast in-memory search."""

    def __init__(self, catalog_path=None, embeddings_path=None):
        catalog_path = catalog_path or os.path.join(_ROOT, "shl_catalog_clean.json")
        embeddings_path = embeddings_path or os.path.join(_ROOT, "catalog_embeddings.npy")
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
        self.model = SentenceTransformer(MODEL_NAME)

        try:
            self.embeddings = np.load(embeddings_path)
            assert self.embeddings.shape[0] == len(self.catalog)
        except (FileNotFoundError, AssertionError):
            self.embeddings = self._build_and_save(embeddings_path)

    def _build_and_save(self, embeddings_path):
        texts = [build_embedding_text(e) for e in self.catalog]
        emb = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        np.save(embeddings_path, emb)
        return emb

    def search(self, query: str, top_k: int = 10,
               test_type_filter=None, job_level_filter=None):
        """
        query: natural language search string
        test_type_filter: optional list of letters e.g. ['K', 'P']
        job_level_filter: optional list of job-level strings
        Returns list of dicts: name, url, test_type, score
        """
        q_emb = self.model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True)[0]
        scores = self.embeddings @ q_emb

        idxs = list(range(len(self.catalog)))
        if test_type_filter:
            idxs = [i for i in idxs if self.catalog[i]["test_type"] in test_type_filter]
        if job_level_filter:
            idxs = [i for i in idxs
                    if any(jl in self.catalog[i].get("job_levels", [])
                           for jl in job_level_filter)]

        ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "name": self.catalog[i]["name"],
                "url": self.catalog[i]["url"],
                "test_type": self.catalog[i]["test_type"],
                "score": float(scores[i]),
            }
            for i in ranked
        ]

    def is_valid_url(self, url: str) -> bool:
        """Used as a grounding guardrail: never return a URL not in our catalog."""
        return any(e["url"] == url for e in self.catalog)
