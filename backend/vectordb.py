"""
Chroma-based vector store interface.
"""

import os
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from .embeddings import embed_texts, embed_text

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
_COLLECTION_NAME = "insightqa"


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(_COLLECTION_NAME)


def add_chunks(
    chunks: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    if not chunks:
        return
    collection = _get_collection()
    embs = embed_texts(chunks)
    collection.add(
        embeddings=embs,
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )


def query(
    query_text: str,
    top_k: int = 5,
    kb_id: str | None = None,
    doc_roles: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Query the vector DB with optional filters:
      - kb_id: only chunks from a specific knowledge base
      - doc_roles: e.g. ["support"] or ["main","support"]
    """
    collection = _get_collection()
    q_emb = embed_text(query_text)

    where = None

    clauses: List[Dict[str, Any]] = []

    if kb_id:
        clauses.append({"kb_id": kb_id})

    if doc_roles:
        if len(doc_roles) == 1:
            clauses.append({"doc_role": doc_roles[0]})
        else:
            # main OR support, etc.
            clauses.append({"$or": [{"doc_role": r} for r in doc_roles]})

    if len(clauses) == 1:
        where = clauses[0]
    elif len(clauses) > 1:
        # Top-level must have exactly one operator -> use $and
        where = {"$and": clauses}
    else:
        where = None

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    for doc, meta, dist, _id in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
        res["ids"][0],
    ):
        out.append(
            {
                "id": _id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
        )
    return {"results": out}
