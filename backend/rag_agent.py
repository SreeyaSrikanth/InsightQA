"""
RAG-based test case generator adapted for Groq.

- Retrieves top-k chunks from the vector DB
- Builds a strict JSON-only prompt
- Calls Groq LLM
- Tries to parse JSON
- If parsing fails → tries auto-repair
- If still fails → returns raw output so Streamlit can show it
"""

from __future__ import annotations

from typing import List, Dict, Any
import json
import re
import ast

from vectordb import query as vectordb_query
from llm_ollama import chat, DEFAULT_MODEL, LLMError


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

Your job:
Generate ONLY a valid JSON array of software test cases.

STRICT RULES:
- Output MUST be valid JSON.
- Output MUST start with '[' and end with ']'.
- No prose, no explanation, no markdown, no headings.
- Each element must include:
  "Test_ID", "Feature", "Test_Scenario",
  "Steps", "Expected_Result", "Type", "Grounded_In"

USER REQUEST:
{user_query}

RETRIEVED CONTEXT:
{context_block}

Now output ONLY the JSON array. If the JSON would be invalid, FIX it first.
"""
    return prompt.strip()


# ------------------ JSON extraction helper ------------------ #
def _clean_and_parse_json(text: str):
    """
    Attempt to extract JSON even if malformed.
    """
    raw = text.strip()

    # Remove code fences
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    # Extract everything between first '[' and last ']'
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found.")

    json_candidate = raw[start:end+1]

    # Remove trailing commas
    json_candidate = re.sub(r",\s*]", "]", json_candidate)
    json_candidate = re.sub(r",\s*}", "}", json_candidate)

    # Try direct json.loads
    try:
        return json.loads(json_candidate)
    except:
        pass

    # Try Python literal eval fallback
    py_friendly = (
        json_candidate.replace("null", "None")
                      .replace("true", "True")
                      .replace("false", "False")
    )

    try:
        return ast.literal_eval(py_friendly)
    except Exception as e:
        raise ValueError(f"JSON remained invalid: {e}\nCandidate:\n{json_candidate}")


# ------------------ Auto-repair using LLM ------------------ #
def _attempt_json_repair(bad_output: str, model: str):
    """
    Ask LLM to fix broken JSON and return corrected version.
    """
    repair_prompt = (
        "Fix the following JSON. "
        "Output ONLY the corrected JSON array. No explanation.\n\n"
        + bad_output
    )

    repaired = chat(
        model,
        [
            {
                "role": "system",
                "content": "You ONLY fix JSON. Respond ONLY with valid JSON array."
            },
            {"role": "user", "content": repair_prompt},
        ],
    )

    return repaired


# ------------------ Main RAG function ------------------ #
def generate_testcases_rag(
    user_query: str,
    top_k: int = 5,
    kb_id: str | None = None,
    doc_roles: list[str] | None = None,
    model: str | None = None,
) -> Dict[str, Any]:

    used_model = model or DEFAULT_MODEL

    # 1) Retrieve relevant chunks
    retrieval = vectordb_query(
        user_query,
        top_k=top_k,
        kb_id=kb_id,
        doc_roles=doc_roles,
    )
    chunks = retrieval["results"]

    # 2) Build LLM prompt
    prompt = build_prompt(user_query, chunks)

    # 3) Call LLM
    try:
        raw_output = chat(
            used_model,
            [
                {
                    "role": "system",
                    "content": (
                        "You produce ONLY valid JSON arrays. "
                        "No markdown, no explanation, no reasoning."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    except LLMError as e:
        return {
            "error": f"LLM Error: {e}",
            "raw_llm_output": None,
            "json_valid": False,
            "retrieved_chunks": chunks,
            "prompt_used": prompt,
        }

    # 4) Try parsing JSON directly
    try:
        parsed = _clean_and_parse_json(raw_output)
        return {
            "model": used_model,
            "kb_id": kb_id,
            "testcases": parsed,
            "raw_llm_output": raw_output,
            "json_valid": True,
            "retrieved_chunks": chunks,
            "prompt_used": prompt,
        }
    except Exception as e1:
        pass  # Try auto-repair

    # 5) Attempt JSON repair
    try:
        repaired_text = _attempt_json_repair(raw_output, used_model)
        parsed_repaired = _clean_and_parse_json(repaired_text)

        return {
            "model": used_model,
            "kb_id": kb_id,
            "testcases": parsed_repaired,
            "raw_llm_output": raw_output,
            "repaired_output": repaired_text,
            "json_valid": True,
            "retrieved_chunks": chunks,
            "prompt_used": prompt,
        }

    except Exception as e2:
        # Total failure → return raw output for debugging
        return {
            "model": used_model,
            "kb_id": kb_id,
            "error": f"JSON failed twice: {e1} | Repair failed: {e2}",
            "raw_llm_output": raw_output,
            "json_valid": False,
            "retrieved_chunks": chunks,
            "prompt_used": prompt,
        }
