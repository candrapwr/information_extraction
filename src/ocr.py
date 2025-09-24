import re

import pytesseract
from PIL import Image
from passporteye import read_mrz


def _resolve_lang(lang_spec: str) -> str:
    """Return a pytesseract language string that is available locally."""
    available = set(pytesseract.get_languages(config=""))
    requested = [item for item in re.split(r"[+\s]", lang_spec) if item]
    resolved = [item for item in requested if item in available]
    if not resolved:
        fallback = "eng" if "eng" in available else next(iter(available), "eng")
        resolved = [fallback]
    return "+".join(resolved)


def extract_text(image_path, lang="eng+ind"):
    """Extract text from image using pytesseract."""
    resolved_lang = _resolve_lang(lang)
    text = pytesseract.image_to_string(Image.open(image_path), lang=resolved_lang)
    return text

def extract_mrz(image_path):
    """Extract MRZ data from passport using passporteye."""
    mrz = read_mrz(image_path)
    return mrz.to_dict() if mrz else None
