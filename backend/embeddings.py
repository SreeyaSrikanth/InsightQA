"""
Thin wrapper around sentence-transformers so the rest of the code
doesn't care which model we use.
"""

from sentence_transformers import SentenceTransformer
from functools import lru_cache
from typing import List
import numpy as np


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> list[np.ndarray]:
    model = _get_model()
    return list(model.encode(texts, convert_to_numpy=True))


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
