"""
VoyageAI Embedding API (Free Tier Compatible)
Replaces local SentenceTransformer model.

Outputs: numpy arrays (same format as before)
"""

import os
import requests
import numpy as np
from typing import List

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")

# Small, fast model (ideal for Railway)
MODEL = "voyage-lite-01"

EMBED_URL = "https://api.voyageai.com/v1/embeddings"


def _check_key():
    if not VOYAGE_API_KEY:
        raise RuntimeError("Missing VOYAGE_API_KEY environment variable.")


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Takes a list of strings.
    Returns a list of numpy embeddings.
    """
    _check_key()

    response = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"},
        json={
            "model": MODEL,
            "input": texts
        },
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"VoyageAI embedding error {response.status_code}: {response.text}"
        )

    data = response.json()["data"]

    # Convert to numpy arrays
    return [np.array(item["embedding"], dtype=float) for item in data]


def embed_text(text: str) -> np.ndarray:
    """
    Single-text embedding (returns a single numpy vector).
    """
    return embed_texts([text])[0]
