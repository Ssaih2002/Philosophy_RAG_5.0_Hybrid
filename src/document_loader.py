import json
import os
from typing import Any, Dict, List

from tqdm import tqdm

from .pdf_loader import load_pdf


def _slice_text(text: str, chars_per_page: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chars_per_page <= 0:
        return [text]
    return [text[i : i + chars_per_page] for i in range(0, len(text), chars_per_page)]


def load_docx(file_path: str, chars_per_page: int = 2500) -> List[Dict[str, Any]]:
    """
    Load a .docx file into "pages" compatible with semantic_chunker:
    [{"text": ..., "page": <int>, "source": <filename>}]

    We don't have real PDF-like page numbers for Word documents, so we create
    pseudo-pages by slicing the full text into fixed-size character windows.
    """
    try:
        from docx import Document  # python-docx
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency for .docx support. Install with: pip install python-docx"
        ) from e

    doc = Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
    parts = _slice_text(full_text, chars_per_page=chars_per_page)
    source = os.path.basename(file_path)
    return [{"text": t, "page": i + 1, "source": source} for i, t in enumerate(parts)]


def _json_to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    # default: stable-ish json dump
    return json.dumps(obj, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a .json file into pages.

    - If JSON is a list: each element becomes one page (page = index+1).
    - If JSON is an object: whole object becomes one page (page = 1).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source = os.path.basename(file_path)
    if isinstance(data, list):
        pages = []
        for i, item in enumerate(data):
            text = _json_to_text(item).strip()
            if text:
                pages.append({"text": text, "page": i + 1, "source": source})
        return pages

    text = _json_to_text(data).strip()
    return [{"text": text, "page": 1, "source": source}] if text else []


def load_all_documents(input_dir: str = "data", show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Recursively load supported documents under input_dir:
    - .pdf (via PyMuPDF)
    - .docx (via python-docx)
    - .json
    """
    paths: List[str] = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            path = os.path.join(root, filename)
            lower = filename.lower()
            if lower.endswith(".pdf") or lower.endswith(".docx") or lower.endswith(".json"):
                paths.append(path)

    all_pages: List[Dict[str, Any]] = []
    path_iter = paths
    if show_progress and paths:
        path_iter = tqdm(paths, desc="加载文档", unit="file")

    for path in path_iter:
        lower = path.lower()
        if lower.endswith(".pdf"):
            all_pages.extend(load_pdf(path))
        elif lower.endswith(".docx"):
            all_pages.extend(load_docx(path))
        elif lower.endswith(".json"):
            all_pages.extend(load_json(path))
    return all_pages

