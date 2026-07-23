import base64
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple, Union

_MODEL = None
_MODEL_KEY = None
_WARMED_UP_KEY = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()

_WARMUP_IMAGE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/l3gH"
    "NAAAAABJRU5ErkJggg=="
)


def _project_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(base_dir, path))


def _load_llama_cpp():
    try:
        from llama_cpp import Llama
        from llama_cpp import llama_chat_format
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python belum terinstall. Jalankan `pip install llama-cpp-python` "
            "di virtualenv project."
        ) from exc
    return Llama, llama_chat_format


def _load_huggingface_hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return None
    return hf_hub_download


def _load_pillow():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError(
            "Pillow belum terinstall. Jalankan `pip install -r requirements.txt` "
            "atau set image_preprocess.enabled: false."
        ) from exc
    return Image, ImageOps


def _extract_json_object(text):
    raw_text = (text or "").strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw_text) if raw_text else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        preview = raw_text[:500].replace("\n", "\\n")
        raise RuntimeError(f"Model response was not valid JSON. Response preview: {preview}")


# Valid values for structured KTP fields. Anything outside these sets is treated
# as a misread (e.g. a neighbouring label/value leaking in). This is the most
# reliable fix for the recurring "Gol. Darah" confusion: the field is usually
# blank on the physical card, so models leak the address/other label into it.
# Tokens that look valid but are actually blank markers are kept separate so a
# post-check can still drop them.
_VALID_BLOOD = {"A", "B", "O", "AB"}
_VALID_GENDER = {"LAKI-LAKI", "PEREMPUAN", "L", "P"}
# Sentinels that mean "blank/unknown" on the physical card, not a real value.
_BLANK_VALUES = {"", "-", "—", "--", "N/A", "Not found", "Not Found", "null"}


def _extract_valid(value, valid_set):
    """Return the value only if it (or a token inside it) is a known member of
    valid_set. Handles cases like 'Gol. Darah: B' -> 'B'."""
    text = str(value).strip()
    upper = text.upper()
    if upper in valid_set:
        return text
    # Strip label noise like "Gol. Darah :-" or "Jenis Kelamin: L" by scanning
    # tokens for a valid one.
    cleaned = re.sub(r"[^A-Za-z0-9/+\-]", " ", upper)
    for token in cleaned.split():
        token = token.strip()
        if token in valid_set:
            return token
    return None


def _clean_value(key, value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.upper() in {v.upper() for v in _BLANK_VALUES}:
        return None
    if key == "gol_darah":
        return _extract_valid(text, _VALID_BLOOD)
    if key == "jenis_kelamin":
        return _extract_valid(text, _VALID_GENDER)
    return text or None


def _apply_schema(data, schema):
    if not isinstance(data, dict):
        data = {}
    normalized = {}
    for key, schema_value in schema.items():
        value = data.get(key)
        if isinstance(schema_value, dict):
            normalized[key] = _apply_schema(value, schema_value)
        else:
            normalized[key] = _clean_value(key, value)
    return normalized


def _resolve_chat_handler(local_config, mmproj_path):
    _, chat_format = _load_llama_cpp()
    handler_name = (local_config.get("chat_handler") or "qwen2.5-vl").lower()
    handlers = {
        "mtmd": "MTMDChatHandler",
        "llava-1-5": "Llava15ChatHandler",
        "llava": "Llava15ChatHandler",
        "qwen2.5-vl": "Qwen25VLChatHandler",
        "qwen25-vl": "Qwen25VLChatHandler",
        "gemma4": "Gemma4ChatHandler",
        "minicpm-v-2.6": "MiniCPMv26ChatHandler",
        "nanollava": "NanoLlavaChatHandler",
    }
    class_name = handlers.get(handler_name, local_config.get("chat_handler_class"))
    if not class_name:
        raise ValueError(f"Unsupported local_model.chat_handler '{handler_name}'.")

    handler_cls = getattr(chat_format, class_name, None)
    if handler_cls is None:
        raise RuntimeError(
            f"llama-cpp-python yang terinstall belum memiliki {class_name}. "
            "Upgrade llama-cpp-python ke versi terbaru."
        )

    handler_kwargs = {"clip_model_path": mmproj_path}
    if "verbose" in local_config:
        handler_kwargs["verbose"] = bool(local_config.get("verbose"))
    if "mmproj_use_gpu" in local_config:
        handler_kwargs["use_gpu"] = bool(local_config.get("mmproj_use_gpu"))

    try:
        return handler_cls(**handler_kwargs)
    except TypeError:
        handler_kwargs.pop("use_gpu", None)
        handler_kwargs.pop("verbose", None)
        return handler_cls(**handler_kwargs)


def _download_model_file(repo_id, filename, target_path, revision=None):
    hf_hub_download = _load_huggingface_hub()
    target_dir = os.path.dirname(target_path)
    os.makedirs(target_dir, exist_ok=True)
    temp_target_path = f"{target_path}.part"
    if os.path.exists(temp_target_path):
        os.remove(temp_target_path)

    print(f"Downloading {filename} from Hugging Face repo {repo_id}...", file=sys.stderr, flush=True)
    if hf_hub_download:
        with tempfile.TemporaryDirectory(prefix="hf-download-", dir=target_dir) as cache_dir:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
            )
            shutil.copyfile(downloaded_path, temp_target_path)
    else:
        revision = revision or "main"
        repo_quoted = urllib.parse.quote(repo_id, safe="/")
        filename_quoted = urllib.parse.quote(filename)
        url = f"https://huggingface.co/{repo_quoted}/resolve/{revision}/{filename_quoted}"
        urllib.request.urlretrieve(url, temp_target_path)
    os.replace(temp_target_path, target_path)
    shutil.rmtree(os.path.join(target_dir, ".cache"), ignore_errors=True)
    print(f"Saved model file to {target_path}", file=sys.stderr, flush=True)


def ensure_model_files(config):
    local_config = (config or {}).get("local_model", {})
    source_config = local_config.get("model_source", {})
    if not source_config.get("auto_download", True):
        return

    repo_id = source_config.get("repo_id")
    revision = source_config.get("revision")
    files = source_config.get("files") or {}
    if not repo_id or not files:
        return

    model_path = _project_path(local_config.get("model_path", "./model/Nanonets-OCR-s-Q4_0.gguf"))
    mmproj_path = _project_path(local_config.get("mmproj_path", "./model/Nanonets-OCR-s-mmproj-F16.gguf"))
    targets = {
        "model": model_path,
        "mmproj": mmproj_path,
    }

    for key, target_path in targets.items():
        if os.path.exists(target_path):
            continue
        filename = files.get(key)
        if not filename:
            raise RuntimeError(f"Missing Hugging Face filename for local_model.model_source.files.{key}.")
        _download_model_file(repo_id, filename, target_path, revision=revision)


def _model_key(local_config):
    return (
        _project_path(local_config.get("model_path", "./model/Nanonets-OCR-s-Q4_0.gguf")),
        _project_path(local_config.get("mmproj_path", "./model/Nanonets-OCR-s-mmproj-F16.gguf")),
        local_config.get("chat_handler", "qwen2.5-vl"),
        local_config.get("ctx_size", 8192),
        local_config.get("n_gpu_layers", -1),
        local_config.get("n_threads"),
        local_config.get("n_threads_batch"),
        local_config.get("n_batch"),
        local_config.get("n_ubatch"),
        local_config.get("flash_attn"),
        local_config.get("op_offload"),
        local_config.get("use_mmap"),
        local_config.get("use_mlock"),
    )


def get_local_model(config):
    """Load the GGUF OCR model once and reuse it for later requests."""
    global _MODEL, _MODEL_KEY
    local_config = (config or {}).get("local_model", {})
    if not local_config.get("enabled", False):
        raise RuntimeError("local_model.enabled is false.")

    key = _model_key(local_config)
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_KEY == key:
            return _MODEL

        ensure_model_files(config)

        model_path, mmproj_path = key[0], key[1]
        if not os.path.exists(model_path):
            raise RuntimeError(f"Local model file not found: {model_path}")
        if not os.path.exists(mmproj_path):
            raise RuntimeError(f"Local mmproj file not found: {mmproj_path}")

        Llama, _ = _load_llama_cpp()
        chat_handler = _resolve_chat_handler(local_config, mmproj_path)

        llama_kwargs = {
            "model_path": model_path,
            "chat_handler": chat_handler,
            "n_ctx": int(local_config.get("ctx_size", 8192)),
            "n_gpu_layers": int(local_config.get("n_gpu_layers", -1)),
            "verbose": bool(local_config.get("verbose", False)),
        }
        optional_args = {
            "n_threads": local_config.get("n_threads"),
            "n_threads_batch": local_config.get("n_threads_batch"),
            "n_batch": local_config.get("n_batch"),
            "n_ubatch": local_config.get("n_ubatch"),
            "flash_attn": local_config.get("flash_attn"),
            "op_offload": local_config.get("op_offload"),
            "seed": local_config.get("seed"),
            "use_mmap": local_config.get("use_mmap"),
            "use_mlock": local_config.get("use_mlock"),
        }
        for name, value in optional_args.items():
            if value is not None:
                llama_kwargs[name] = value

        _MODEL = Llama(**llama_kwargs)
        _MODEL_KEY = key
        return _MODEL


def warmup_local_model(config, model=None):
    global _WARMED_UP_KEY
    local_config = (config or {}).get("local_model", {})
    if not local_config.get("warmup_on_start", True):
        return None

    model = model or get_local_model(config)
    if _WARMED_UP_KEY == _MODEL_KEY:
        return model

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _WARMUP_IMAGE_DATA_URI},
                },
                {
                    "type": "text",
                    "text": "Return {}",
                },
            ],
        }
    ]
    with _INFERENCE_LOCK:
        model.create_chat_completion(
            messages=messages,
            temperature=0,
            max_tokens=8,
            response_format={"type": "json_object"},
        )
    _WARMED_UP_KEY = _MODEL_KEY
    return model


def shutdown_local_model():
    global _MODEL, _MODEL_KEY, _WARMED_UP_KEY
    model = _MODEL
    _MODEL = None
    _MODEL_KEY = None
    _WARMED_UP_KEY = None
    if model is None:
        return
    close = getattr(model, "close", None)
    if callable(close):
        close()


def preload_local_model(config, force=False):
    local_config = (config or {}).get("local_model", {})
    should_preload = force or local_config.get("preload_on_start", True)
    if local_config.get("enabled", False) and should_preload:
        if (config or {}).get("image_preprocess", {}).get("enabled", False):
            _load_pillow()
        print("Loading local GGUF model...", file=sys.stderr, flush=True)
        model = get_local_model(config)
        print("Warming up local GGUF model...", file=sys.stderr, flush=True)
        warmup_local_model(config, model)
        print("Local GGUF model ready.", file=sys.stderr, flush=True)
        return model
    return None


def _image_to_data_uri(image_path, preprocess_config=None):
    preprocess_config = preprocess_config or {}
    metadata = {
        "image_preprocess_enabled": bool(preprocess_config.get("enabled", False)),
        "image_original_bytes": os.path.getsize(image_path),
    }

    if preprocess_config.get("enabled", False):
        Image, ImageOps = _load_pillow()

        max_width = int(preprocess_config.get("max_width", 768))
        max_height = int(preprocess_config.get("max_height", 768))
        jpeg_quality = int(preprocess_config.get("jpeg_quality", 85))
        jpeg_optimize = bool(preprocess_config.get("jpeg_optimize", False))
        reencode_if_unchanged = bool(preprocess_config.get("reencode_if_unchanged", False))

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            metadata["image_original_width"] = image.width
            metadata["image_original_height"] = image.height
            original_size = image.size

            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba_image = image.convert("RGBA")
                background = Image.new("RGB", rgba_image.size, "white")
                background.paste(rgba_image, mask=rgba_image.getchannel("A"))
                image = background
            elif image.mode != "RGB":
                image = image.convert("RGB")

            if max_width > 0 and max_height > 0:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image.thumbnail((max_width, max_height), resampling)

            if image.size == original_size and not reencode_if_unchanged:
                mime_type, _ = mimetypes.guess_type(image_path)
                if not mime_type:
                    mime_type = "image/jpeg"
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                metadata["image_processed_width"] = image.width
                metadata["image_processed_height"] = image.height
                metadata["image_processed_bytes"] = len(image_bytes)
                metadata["image_compression_ratio"] = 1.0
                metadata["image_preprocess_skipped"] = "within_limit"
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                return f"data:{mime_type};base64,{image_b64}", metadata

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=jpeg_optimize)
            image_bytes = buffer.getvalue()
            metadata["image_processed_width"] = image.width
            metadata["image_processed_height"] = image.height
        metadata["image_processed_bytes"] = len(image_bytes)
        metadata["image_compression_ratio"] = round(
            len(image_bytes) / max(metadata["image_original_bytes"], 1),
            3,
        )
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}", metadata

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    metadata["image_processed_bytes"] = len(image_bytes)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}", metadata


def available_templates(config):
    templates = (config or {}).get("templates", {})
    if not isinstance(templates, dict) or not templates:
        return {}
    return templates


def default_template_name(config):
    config = config or {}
    local_config = config.get("local_model", {})
    templates = available_templates(config)
    configured = local_config.get("default_template")
    if configured in templates:
        return configured
    return next(iter(templates), None)


def _schema_for_template(config, template_name):
    templates = available_templates(config)
    selected = template_name or default_template_name(config)
    if selected not in templates:
        raise RuntimeError(f"Template '{selected}' tidak ditemukan di config/config.yaml.")
    return selected, templates[selected]


def _prompt(config, template_name=None):
    local_config = (config or {}).get("local_model", {})
    instruction = local_config.get("prompt") or "Return JSON matching schema exactly."
    selected, schema = _schema_for_template(config, template_name)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    return f"{instruction}\nTemplate: {selected}\nSchema:\n{schema_text}"


def _response_content(response):
    if not isinstance(response, dict):
        raise RuntimeError("Model response was not a dictionary.")
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Model response did not contain choices.")
    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text")
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    if not content:
        raise RuntimeError("Model response did not contain text content.")
    return content


def extract_document(
    image_path: str,
    config: Optional[Dict] = None,
    template_name: Optional[str] = None,
    return_timings: bool = False,
) -> Union[Tuple[Dict, Optional[Dict]], Tuple[Dict, Optional[Dict], Dict]]:
    timings = {}
    started_at = time.perf_counter()
    local_config = (config or {}).get("local_model", {})
    selected_template, schema = _schema_for_template(config, template_name)
    timings["schema_seconds"] = round(time.perf_counter() - started_at, 3)

    model_started_at = time.perf_counter()
    llm = get_local_model(config)
    timings["get_model_seconds"] = round(time.perf_counter() - model_started_at, 3)

    # KTP-specific preprocessing: crop + verify NIK + uniform resize.
    # Runs only for the "ktp" template; raises on detection failure.
    active_image_path = image_path
    ktp_cfg = (config or {}).get("ktp_preprocess", {})
    if selected_template == "ktp" and ktp_cfg.get("enabled", False):
        from src.ktp_preprocess import preprocess_ktp
        ktp_started_at = time.perf_counter()
        ktp_path, ktp_timings = preprocess_ktp(image_path, config)
        timings.update(ktp_timings)
        timings["ktp_pipeline_seconds"] = round(time.perf_counter() - ktp_started_at, 3)
        active_image_path = ktp_path

    image_started_at = time.perf_counter()
    image_data_uri, image_metadata = _image_to_data_uri(active_image_path, config.get("image_preprocess"))
    timings["image_preprocess_seconds"] = round(time.perf_counter() - image_started_at, 3)
    timings.update(image_metadata)

    prompt_started_at = time.perf_counter()
    prompt_text = _prompt(config, template_name)
    timings["prompt_build_seconds"] = round(time.perf_counter() - prompt_started_at, 3)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri},
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]

    kwargs = {
        "messages": messages,
        "temperature": float(local_config.get("temperature", 0)),
        "max_tokens": int(local_config.get("max_tokens", 512)),
    }
    if local_config.get("json_mode", True):
        kwargs["response_format"] = {"type": "json_object"}

    with _INFERENCE_LOCK:
        inference_started_at = time.perf_counter()
        response = llm.create_chat_completion(**kwargs)
        timings["inference_seconds"] = round(time.perf_counter() - inference_started_at, 3)

    parse_started_at = time.perf_counter()
    data = _extract_json_object(_response_content(response))
    normalized = _apply_schema(data, schema)
    timings["parse_seconds"] = round(time.perf_counter() - parse_started_at, 3)
    timings["total_extract_seconds"] = round(time.perf_counter() - started_at, 3)
    if return_timings:
        return normalized, response.get("usage"), timings
    return normalized, response.get("usage")
