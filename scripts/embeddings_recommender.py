"""Embeddings-based recommender using sentence-transformers.

This module builds embeddings for job categories (from JOB_KEYWORDS/JOB_TITLES)
and exposes `recommend_jobs_semantic(text, n=5)` which returns top-N by cosine similarity.

Requirements: `sentence-transformers`, `numpy`
"""
from typing import List, Dict, Tuple
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import re

from recommend import JOB_KEYWORDS, JOB_TITLES


def _normalize(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[\/_\-\+]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class EmbeddingsRecommender:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Try to instantiate the SentenceTransformer normally. Some environments
        # (with torch meta tensors / lazy init) may raise a NotImplementedError
        # like: "Cannot copy out of meta tensor; no data!" when moving a module
        # from 'meta' to a device. In that case, fall back to creating the
        # model directly on CPU to avoid meta-device moves.
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            msg = str(e)
            if "meta" in msg or "Cannot copy out of meta tensor" in msg or isinstance(e, NotImplementedError):
                # fallback
                print(
                    "Warning: model instantiation hit a meta-tensor error — falling back to CPU instantiation."
                )
                self.model = SentenceTransformer(self.model_name, device="cpu")
            else:
                # re-raise unexpected errors
                raise
        self.job_keys = list(JOB_KEYWORDS.keys())
        self.job_texts = [self._build_job_text(k) for k in self.job_keys]
        self.job_embeddings = self.model.encode(self.job_texts, convert_to_numpy=True)

    def _build_job_text(self, key: str) -> str:
        # Combine title and keywords into a short description for embedding
        title = JOB_TITLES.get(key, key)
        kws = JOB_KEYWORDS.get(key, [])
        return f"{title}. Keywords: {'; '.join(kws)}"

    def recommend(self, text: str, n: int = 5) -> List[Dict]:
        t = _normalize(text)
        emb = self.model.encode([t], convert_to_numpy=True)[0]

        # cosine similarities
        sims = (self.job_embeddings @ emb) / (
            (np.linalg.norm(self.job_embeddings, axis=1) * np.linalg.norm(emb)) + 1e-12
        )
        idxs = np.argsort(-sims)[:n]

        results = []
        for i in idxs:
            key = self.job_keys[int(i)]
            results.append({
                "title": JOB_TITLES.get(key, key),
                "score": float(sims[int(i)]),
                "explanation": self.job_texts[int(i)],
            })
        return results


def recommend_jobs_semantic(text: str, n: int = 5, model_name: str = "all-MiniLM-L6-v2") -> List[Dict]:
    """Convenience wrapper that loads a model and returns recommendations.

    Note: constructing the recommender repeatedly is expensive; in a server keep one instance.
    """
    rec = EmbeddingsRecommender(model_name=model_name)
    return rec.recommend(text, n=n)


if __name__ == "__main__":
    # quick smoke test (only works when sentence-transformers is installed)
    r = EmbeddingsRecommender()
    sample = "Experienced teacher with curriculum development, classroom management, lesson planning"
    print(r.recommend(sample, n=5))
