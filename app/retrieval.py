"""
Stage 1: Retrieval layer (TF-IDF version).

Originally built with sentence-transformers, but switched to TF-IDF
(scikit-learn) because sentence-transformers pulls in `tokenizers`, which
needs Rust compilation and has no pre-built wheel on some free-tier hosts
(Render's build image blocks writing to the cargo cache, so the build
fails outright). TF-IDF has zero compiled/Rust dependencies, installs
instantly anywhere, and needs no model download at runtime - at this
catalog size (370 items) the retrieval quality difference is minor and
the deployment reliability win is worth it.
"""
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    """Loads catalog once, fits a TF-IDF matrix in-memory (fast at this size,
    no need to persist/cache to disk - fitting takes well under a second
    for 370 short documents)."""

    def __init__(self, catalog_path=None):
        catalog_path = catalog_path or os.path.join(_ROOT, "shl_catalog.json")
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        texts = [build_embedding_text(e) for e in self.catalog]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10,
               test_type_filter=None, job_level_filter=None):
        """
        query: natural language search string
        test_type_filter: optional list of letters e.g. ['K', 'P']
        job_level_filter: optional list of job-level strings
        Returns list of dicts: name, url, test_type, score
        """
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]

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
