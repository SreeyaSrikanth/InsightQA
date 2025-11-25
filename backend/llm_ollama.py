"""
Thin wrapper around the Groq API (replacing local Ollama).
"""

from __future__ import annotations

import os
from typing import List, Dict
from groq import Groq


# Read API key from Railway environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Best Groq model for speed + reliability
DEFAULT_MODEL = "llama-3.1-8b-instant"


class LLMError(RuntimeError):
    """Custom error class for LLM-related problems."""
    pass


def chat(model: str, messages: List[Dict[str, str]]) -> str:
    """
    Wrapper for Groq's streaming chat API.
    Returns the final combined text, matching original Ollama behavior.
    """

    if not GROQ_API_KEY:
        raise LLMError("Missing GROQ_API_KEY environment variable")

    client = Groq(api_key=GROQ_API_KEY)

    try:
        # Note: stream=True to match your original interface
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_completion_tokens=1024,
            top_p=1.0,
            stream=True,
        )

        output_text = ""

        for chunk in completion:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                output_text += delta.content

        return output_text

    except Exception as e:
        raise LLMError(f"Groq API error: {str(e)}")
