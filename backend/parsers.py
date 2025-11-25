"""
Parsers for different document types: md/txt/pdf/html/json.

All functions return plain UTF-8 text.
"""

from pathlib import Path
from bs4 import BeautifulSoup
import fitz
import json
from typing import Union


def _read_bytes_or_path(path_or_bytes: Union[str, bytes]) -> bytes:
    if isinstance(path_or_bytes, bytes):
        return path_or_bytes
    return Path(path_or_bytes).read_bytes()


def parse_txt(path_or_bytes: Union[str, bytes]) -> str:
    data = _read_bytes_or_path(path_or_bytes)
    return data.decode("utf-8", errors="ignore")


def parse_md(path_or_bytes: Union[str, bytes]) -> str:
    return parse_txt(path_or_bytes)


def parse_pdf(path_or_bytes: Union[str, bytes]) -> str:
    """If given a path, open it directly. If bytes, use pymupdf memory open."""
    if isinstance(path_or_bytes, bytes):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(path_or_bytes)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)


def parse_html(path_or_bytes: Union[str, bytes]) -> str:
    data = _read_bytes_or_path(path_or_bytes)
    soup = BeautifulSoup(data, "html.parser")
    return soup.prettify()


def parse_json(path_or_bytes: Union[str, bytes]) -> str:
    data = _read_bytes_or_path(path_or_bytes)
    obj = json.loads(data.decode("utf-8", errors="ignore"))
    return json.dumps(obj, indent=2)


def parse_any(path: str, raw_bytes: bytes | None = None) -> str:
    """
    Detect file type from suffix and dispatch.
    If raw_bytes is provided (e.g. from UploadFile), we use that.
    """
    suffix = Path(path).suffix.lower()
    data = raw_bytes if raw_bytes is not None else path

    if suffix in [".txt"]:
        return parse_txt(data)
    if suffix in [".md", ".markdown"]:
        return parse_md(data)
    if suffix in [".pdf"]:
        return parse_pdf(data)
    if suffix in [".html", ".htm"]:
        return parse_html(data)
    if suffix in [".json"]:
        return parse_json(data)
    return parse_txt(data)
