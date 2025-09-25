import base64
import json
import mimetypes
import os
import re
import warnings
import shutil
from functools import lru_cache
from typing import Dict, Iterable, Optional

import pytesseract
import requests
from PIL import Image
from passporteye import read_mrz

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*pin_memory.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*plugin infrastructure in `skimage.io`.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*square is deprecated.*",
)

try:
    import easyocr  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    easyocr = None


def _resolve_lang(lang_spec: str) -> str:
    """Return a pytesseract language string that is available locally."""
    available = set(pytesseract.get_languages(config=""))
    requested = [item for item in re.split(r"[+\s]", lang_spec) if item]
    resolved = [item for item in requested if item in available]
    if not resolved:
        fallback = "eng" if "eng" in available else next(iter(available), "eng")
        resolved = [fallback]
    return "+".join(resolved)


def _extract_text_pytesseract(image_path: str, lang: str, config: Optional[Dict]) -> str:
    """Extract text via pytesseract with configuration-aware settings."""
    tess_config = (config or {}).get("tesseract", {})
    tess_path = tess_config.get("path")
    if tess_path:
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
        else:
            fallback = shutil.which("tesseract")
            if fallback:
                pytesseract.pytesseract.tesseract_cmd = fallback
            else:
                raise RuntimeError(
                    f"Tesseract executable not found at '{tess_path}'. Update `config/config.yaml` or install Tesseract."
                )
    else:
        fallback = shutil.which("tesseract")
        if fallback:
            pytesseract.pytesseract.tesseract_cmd = fallback
        else:
            raise RuntimeError(
                "Tesseract executable not found. Install Tesseract or set `tesseract.path` in config/config.yaml."
            )
    resolved_lang = _resolve_lang(lang)
    return pytesseract.image_to_string(Image.open(image_path), lang=resolved_lang)


@lru_cache(maxsize=4)
def _get_easyocr_reader(lang_tuple: Iterable[str], use_gpu: bool):
    if easyocr is None:
        raise ImportError("easyocr is not installed. Install it via `pip install easyocr`." )
    return easyocr.Reader(list(lang_tuple), gpu=use_gpu)


def _extract_text_easyocr(image_path: str, lang: str, config: Optional[Dict]) -> str:
    """Extract text using EasyOCR and return a newline-joined string."""
    easy_config = (config or {}).get("easyocr", {})
    lang_list = easy_config.get("lang")
    if not lang_list:
        lang_list = [code.lower() for code in re.split(r"[+\s]", lang) if code]
    use_gpu = bool(easy_config.get("gpu", False))
    reader = _get_easyocr_reader(tuple(lang_list), use_gpu)
    lines = reader.readtext(image_path, detail=0)
    return "\n".join(line.strip() for line in lines if isinstance(line, str) and line.strip())


def extract_text(image_path: str, lang: str = "eng+ind", provider: str = "pytesseract", config: Optional[Dict] = None) -> str:
    """Extract text from image using the selected OCR provider."""
    provider_normalized = (provider or "pytesseract").lower()
    if provider_normalized in {"pytesseract", "tesseract"}:
        return _extract_text_pytesseract(image_path, lang, config)
    if provider_normalized == "easyocr":
        return _extract_text_easyocr(image_path, lang, config)
    raise ValueError(f"Unsupported OCR provider '{provider}'.")


def extract_mrz(image_path: str):
    """Extract MRZ data from passport using passporteye."""
    mrz = read_mrz(image_path)
    return mrz.to_dict() if mrz else None


def _default_prompt(doc_type: str) -> str:
    base_instruction = (
        "You are an AI that only outputs raw JSON. Never include explanations or markdown. "
        "Return a valid JSON object only."
    )
    if doc_type.lower() == "passport":
        return (
            f"{base_instruction}\n\n"
            "Extract the following fields from the passport image:\n"
            "- passport_number\n- name\n- nationality\n- date_of_birth\n- gender\n"
            "- expiration_date\n- country_code\n"
            "If a field is missing or unreadable, set it to null."
        )
    return (
        f"{base_instruction}\n\n"
        "Extract the following fields from the Indonesian KTP image:\n"
        "- province\n- city\n- nik\n- name\n- birth_place\n- birth_date\n"
        "- gender\n- blood_type\n- address\n- rt_rw\n- kelurahan_desa\n"
        "- kecamatan\n- religion\n- marital_status\n- occupation\n- nationality\n"
        "- valid_until\nIf any field is missing or unreadable, set its value to null."
    )


def extract_llm_data(image_path: str, doc_type: str, config: Optional[Dict] = None):
    """Call an LLM API (e.g., Gemini) to extract structured data from an image.

    Returns a tuple `(data, usage)` where `data` is the parsed JSON payload and
    `usage` contains token usage metadata (if provided by the API).
    """
    llm_config = (config or {}).get("llm", {})
    if not llm_config:
        raise ValueError("LLM configuration is missing. Populate the 'llm' section in config.yaml.")

    api_key_env = llm_config.get("api_key_env")
    api_key = None
    if api_key_env:
        api_key = os.getenv(api_key_env)
    if not api_key:
        api_key = llm_config.get("api_key")
    if not api_key:
        target = api_key_env or "llm.api_key"
        raise ValueError(
            f"LLM API key is not configured. Set environment variable '{target}' or provide `llm.api_key`."
        )

    model = llm_config.get("model", "gemini-2.5-flash")
    endpoint = llm_config.get("endpoint", "https://generativelanguage.googleapis.com/v1beta")
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/models/{model}:generateContent"

    prompts = llm_config.get("prompts", {})
    prompt = prompts.get(doc_type) or prompts.get("default") or _default_prompt(doc_type)

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    timeout = llm_config.get("timeout", 30)

    response = requests.post(url, headers=headers, params=params, data=json.dumps(payload), timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"LLM request failed with status {response.status_code}: {response.text}"
        )

    body = response.json()
    usage = body.get("usageMetadata")
    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError("LLM response did not contain any candidates.")

    parts = candidates[0].get("content", {}).get("parts") or []
    candidate_usage = candidates[0].get("usageMetadata")
    if candidate_usage:
        usage = usage or candidate_usage
    text_candidates = [part.get("text") for part in parts if isinstance(part, dict) and part.get("text")]
    if not text_candidates:
        raise RuntimeError("LLM response missing text content.")

    raw_text = text_candidates[0].strip().replace("\n", "").replace("```json", "").replace("```", "")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - depends on external API
        raise RuntimeError(f"LLM response was not valid JSON: {exc}") from exc
    return data, usage
