"""
RAG-based test case generator using Ollama.

- Retrieves top-k chunks from the vector DB
- Builds a strict JSON-only prompt
- Calls Ollama
- Parses JSON into Python objects
"""

from __future__ import annotations

from typing import List, Dict, Any
import json
import re

from .vectordb import query as vectordb_query
from .llm_ollama import chat, DEFAULT_MODEL, LLMError


# ------------------ Prompt builder ------------------ #
def build_prompt(user_query: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_strs = []
    for i, item in enumerate(context_chunks, start=1):
        meta = item["metadata"]
        src = meta.get("source_document", "unknown")
        context_strs.append(
            f"[CONTEXT {i} | source_document={src}]\n{item['document']}\n"
        )
    context_block = "\n\n".join(context_strs)

    prompt = f"""
You are a senior QA engineer.

You are given:
- A user request for test cases.
- Context snippets from product documentation and UI specs.

USER REQUEST:
{user_query}

CONTEXT SNIPPETS:
{context_block}

TASK:
1. Use ONLY the information in the CONTEXT. Do NOT invent features or behavior that is not described.
2. Generate a JSON ARRAY of test cases. Do NOT include any explanation outside the JSON.
3. Each element in the array MUST be an object with EXACTLY these keys:

   - "Test_ID": string like "TC-001", "TC-002", ...
   - "Feature": short string feature name
   - "Test_Scenario": one-sentence description
   - "Steps": array of step strings
   - "Expected_Result": one-sentence expected outcome
   - "Type": "Positive" or "Negative"
   - "Grounded_In": array of source_document names you used

4. Ensure JSON is valid and directly parseable.
5. Start your answer with '[' and end with ']'. No backticks, no markdown, no extra text.
"""
    return prompt.strip()


# ------------------ JSON parsing helper ------------------ #
def _extract_json_array(text: str) -> Any:
    """
    Extract a JSON array from the model's response.
    We expect the whole response to be the array, but we are robust to minor wrappers.
    """
    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # If it already starts with [ and ends with ], use directly
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return json.loads(text)

    # Fallback: grab the first [...] block
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON array found in LLM output.")
    return json.loads(m.group(0))


# ------------------ Public RAG function ------------------ #
def generate_testcases_rag(
    user_query: str,
    top_k: int = 5,
    kb_id: str | None = None,
    doc_roles: list[str] | None = None,
    model: str | None = None,
) -> Dict[str, Any]:
    """
    RAG test generation restricted to a specific KB and optionally certain doc roles.
    doc_roles examples:
        ["support"]           # only support docs
        ["main", "support"]   # both
        None                  # no role filter, all docs
    """
    # 1) Retrieve relevant chunks from vector DB
    retrieval = vectordb_query(
        user_query,
        top_k=top_k,
        kb_id=kb_id,
        doc_roles=doc_roles,
    )
    chunks = retrieval["results"]

    # 2) Build strict JSON prompt
    prompt = build_prompt(user_query, chunks)

    # 3) Call Ollama
    used_model = model or DEFAULT_MODEL
    try:
        content = chat(
            used_model,
            [
                {
                    "role": "system",
                    "content": "You generate high-quality software test cases strictly in JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except LLMError as e:
        raise

    # 4) Parse JSON array
    try:
        testcases = _extract_json_array(content)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON testcases from LLM output: {e}")

    if not isinstance(testcases, list):
        raise ValueError("LLM output is not a JSON array of test cases.")

    return {
        "model": used_model,
        "kb_id": kb_id,
        "testcases": testcases,
        "retrieved_chunks": chunks,
        "prompt_used": prompt,
    }