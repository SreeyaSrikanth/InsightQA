"""
Thin wrapper around the local Ollama HTTP API.

Default:
- BASE URL: http://localhost:11434 (override with OLLAMA_BASE_URL)
- MODEL:    llama3.1 (override with OLLAMA_MODEL)
"""

from __future__ import annotations

import os
from typing import List, Dict, Any
import requests


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


class LLMError(RuntimeError):
    pass


def chat(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Call Ollama's /api/chat endpoint with non-streaming response.
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    Returns the assistant's content as a string.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
    except requests.RequestException as e:
        raise LLMError(f"Failed to reach Ollama at {OLLAMA_BASE_URL}: {e}")

    if resp.status_code != 200:
        raise LLMError(f"Ollama error {resp.status_code}: {resp.text}")

    data = resp.json()

    # Newer Ollama /api/chat returns a single JSON object with "message"
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    # Some versions may stream or structure differently; handle basic fallback
    if "content" in data:
        return data["content"]

    raise LLMError(f"Unexpected Ollama response shape: {data}")
