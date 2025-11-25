from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, Any, List
import json

from llm_ollama import chat, DEFAULT_MODEL, LLMError


# -----------------------------
# Extract HTML elements
# -----------------------------
def extract_ui_elements(html_path: str) -> List[Dict[str, str]]:
    if not Path(html_path).exists():
        return []

    html = Path(html_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    elements = []
    for el in soup.find_all(True):
        elements.append({
            "tag": el.name,
            "id": el.get("id") or "",
            "name": el.get("name") or "",
            "class": " ".join(el.get("class")) if el.get("class") else "",
            "placeholder": el.get("placeholder") or "",
            "text": el.get_text(strip=True)
        })

    return elements


# -----------------------------
# Build the main prompt
# -----------------------------
def build_prompt(testcase: Dict[str, Any], html_path: str) -> str:

    tc_id = testcase.get("Test_ID", "TC-UNKNOWN")
    scenario = testcase.get("Test_Scenario", "")
    steps = testcase.get("Steps", [])

    ui = extract_ui_elements(html_path)
    html_abs = Path(html_path).resolve()

    prompt = f"""
You are an expert QA automation engineer.

Your job:
Generate a COMPLETE Selenium Python test script for the given test case.

STRICT RULES:
- Use Selenium + webdriver_manager + Chrome.
- ALWAYS prefer:
    By.ID → By.NAME → By.CLASS_NAME → XPath (only last resort).
- The script MUST follow the test steps EXACTLY.
- The script MUST open:
    file://{html_abs}
- Use time.sleep(1) between steps.
- RETURN ONLY PYTHON CODE. No explanations. No markdown.

TEST CASE:
ID: {tc_id}
Scenario: {scenario}

Steps:
{json.dumps(steps, indent=2)}

HTML UI ELEMENTS:
{json.dumps(ui, indent=2)}

Now output ONLY runnable Python code.
"""

    return prompt.strip()


# -----------------------------
# Clean the code
# -----------------------------
def clean_code(raw: str) -> str:
    cleaned = raw.replace("```python", "").replace("```", "").strip()
    return cleaned


# -----------------------------
# Auto-repair using LLM
# -----------------------------
def attempt_code_repair(bad_code: str, model: str) -> str:

    repair_prompt = f"""
Fix the following Selenium Python code so that it becomes valid and runnable.
Return ONLY the corrected code. No markdown. No comments unless inside code.

Broken code:
{bad_code}
"""

    repaired = chat(
        model,
        [
            {"role": "system", "content": "You ONLY fix Python code. Output code only."},
            {"role": "user", "content": repair_prompt},
        ],
    )

    return repaired


# -----------------------------
# Main Public Function
# -----------------------------
def generate_selenium_script(testcase: Dict[str, Any], html_path: str) -> str:

    prompt = build_prompt(testcase, html_path)
    model = DEFAULT_MODEL

    # 1) FIRST ATTEMPT
    try:
        raw = chat(
            model,
            [
                {"role": "system",
                 "content": (
                     "You output ONLY executable Selenium Python code. "
                     "No markdown. No explanation."
                 )
                },
                {"role": "user", "content": prompt},
            ],
        )
    except LLMError as e:
        return f"# LLM Error: {e}"

    cleaned = clean_code(raw)

    # Try naive validation: script must contain `def run_test()`
    if "def run_test" in cleaned:
        return cleaned

    # 2) AUTO-REPAIR IF BROKEN
    try:
        repaired = attempt_code_repair(cleaned, model)
        repaired_cleaned = clean_code(repaired)

        return repaired_cleaned

    except Exception:
        return "# Failed to generate valid code.\n" + cleaned
