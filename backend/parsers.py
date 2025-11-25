"""
Parsers for different document types: txt/md/pdf/html/json.

All functions return plain UTF-8 text for vector embeddings.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import json
from typing import Union


def _read_bytes_or_path(path_or_bytes: Union[str, bytes]) -> bytes:
    """Return bytes whether input is a file path or raw bytes."""
    if isinstance(path_or_bytes, bytes):
        return path_or_bytes
    return Path(path_or_bytes).read_bytes()


def parse_txt(path_or_bytes: Union[str, bytes]) -> str:
    data = _read_bytes_or_path(path_or_bytes)
    return data.decode("utf-8", errors="ignore")


def parse_md(path_or_bytes: Union[str, bytes]) -> str:
    return parse_txt(path_or_bytes)


def parse_pdf(path_or_bytes: Union[str, bytes]) -> str:
    """Parse PDF from path or in-memory bytes."""
    if isinstance(path_or_bytes, bytes):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(path_or_bytes)

    pages = []
    for page in doc:
        pages.append(page.get_text())

    return "\n".join(pages)


def parse_html(path_or_bytes: Union[str, bytes]) -> str:
    """
    Extract clean text from HTML.
    DO NOT use prettify() — it destroys structure & creates huge embeddings.
    """
    data = _read_bytes_or_path(path_or_bytes)
    soup = BeautifulSoup(data, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def parse_json(path_or_bytes: Union[str, bytes]) -> str:
    data = _read_bytes_or_path(path_or_bytes)
    try:
        obj = json.loads(data.decode("utf-8", errors="ignore"))
        return json.dumps(obj, indent=2)
    except Exception:
        return ""


def parse_any(path: str, raw_bytes: bytes | None = None) -> str:
    """
    Detect file type from suffix and dispatch.
    If raw_bytes is provided (uploaded file), use that.
    """
    suffix = Path(path).suffix.lower()
    data = raw_bytes if raw_bytes is not None else path

    if suffix == ".txt":
        return parse_txt(data)
    if suffix in [".md", ".markdown"]:
        return parse_md(data)
    if suffix == ".pdf":
        return parse_pdf(data)
    if suffix in [".html", ".htm"]:
        return parse_html(data)
    if suffix == ".json":
        return parse_json(data)

    # fallback
    return parse_txt(data)
