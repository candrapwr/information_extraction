import base64
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import urllib.parse
import urllib.request
from typing import Dict, Optional, Tuple

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


def _labels_for_field(field_config):
    if not isinstance(field_config, dict):
        return []
    labels = field_config.get("labels", field_config.get("label"))
    if not labels:
        return []
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return [str(labels)]


def _clean_value(key, value, field_config=None):
    if value in ("", "Not found"):
        return None
    if isinstance(value, str):
        normalized_value = re.sub(r"[^A-Z0-9]", "", value.upper())
        for label in _labels_for_field(field_config):
            normalized_label = re.sub(r"[^A-Z0-9]", "", label.upper())
            if normalized_value == normalized_label:
                return None
        if key == "birth_place" and "," in value:
            return value.split(",", 1)[0].strip() or None
    return value


def _apply_schema(data, schema, fields=None):
    if not isinstance(data, dict):
        data = {}
    fields = fields or {}
    normalized = {}
    for key, schema_value in schema.items():
        value = data.get(key)
        if isinstance(schema_value, dict):
            normalized[key] = _apply_schema(value, schema_value, fields.get(key))
        else:
            normalized[key] = _clean_value(key, value, fields.get(key))
    return normalized


def _template_fields(template):
    if isinstance(template, dict) and isinstance(template.get("fields"), dict):
        return template["fields"]
    if isinstance(template, dict):
        return template
    return {}


def _field_schema(fields):
    schema = {}
    for key, field_config in fields.items():
        if isinstance(field_config, dict) and any(meta in field_config for meta in ("label", "labels", "hint")):
            schema[key] = field_config.get("default")
        else:
            schema[key] = field_config
    return schema


def _field_hints(fields):
    hints = []
    for key, field_config in fields.items():
        if not isinstance(field_config, dict):
            continue
        labels = field_config.get("labels", field_config.get("label"))
        hint = field_config.get("hint")
        if isinstance(labels, list):
            labels = ", ".join(str(label) for label in labels)
        parts = []
        if labels:
            parts.append(f"label: {labels}")
        if hint:
            parts.append(str(hint))
        if parts:
            hints.append(f"- {key}: {'; '.join(parts)}")
    return "\n".join(hints)


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

    print(f"Downloading {filename} from Hugging Face repo {repo_id}...", file=sys.stderr, flush=True)
    if hf_hub_download:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
    else:
        revision = revision or "main"
        repo_quoted = urllib.parse.quote(repo_id, safe="/")
        filename_quoted = urllib.parse.quote(filename)
        url = f"https://huggingface.co/{repo_quoted}/resolve/{revision}/{filename_quoted}"
        downloaded_path = os.path.join(target_dir, filename)
        urllib.request.urlretrieve(url, downloaded_path)
    if os.path.abspath(downloaded_path) != os.path.abspath(target_path):
        shutil.move(downloaded_path, target_path)
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

        model_path, mmproj_path, _, _, _ = key
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
            "n_batch": local_config.get("n_batch"),
            "flash_attn": local_config.get("flash_attn"),
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
        print("Loading local GGUF model...", file=sys.stderr, flush=True)
        model = get_local_model(config)
        print("Warming up local GGUF model...", file=sys.stderr, flush=True)
        warmup_local_model(config, model)
        print("Local GGUF model ready.", file=sys.stderr, flush=True)
        return model
    return None


def _image_to_data_uri(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{image_b64}"


def available_templates(config):
    templates = (config or {}).get("templates", {})
    if not isinstance(templates, dict) or not templates:
        return {}
    return templates


def default_template_name(config):
    local_config = (config or {}).get("local_model", {})
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
    fields = _template_fields(templates[selected])
    schema = _field_schema(fields)
    return selected, schema


def _prompt(config, template_name=None):
    local_config = (config or {}).get("local_model", {})
    instruction = local_config.get("prompt") or "Return JSON matching schema exactly."
    templates = available_templates(config)
    selected, schema = _schema_for_template(config, template_name)
    fields = _template_fields(templates[selected])
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    hints = _field_hints(fields)
    if hints:
        return f"{instruction}\nTemplate: {selected}\nSchema:\n{schema_text}\nField mapping:\n{hints}"
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


def extract_document(image_path: str, config: Optional[Dict] = None, template_name: Optional[str] = None) -> Tuple[Dict, Optional[Dict]]:
    local_config = (config or {}).get("local_model", {})
    selected, schema = _schema_for_template(config, template_name)
    fields = _template_fields(available_templates(config)[selected])
    llm = get_local_model(config)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_uri(image_path)},
                },
                {
                    "type": "text",
                    "text": _prompt(config, template_name),
                },
            ],
        }
    ]

    kwargs = {
        "messages": messages,
        "temperature": float(local_config.get("temperature", 0)),
        "max_tokens": int(local_config.get("max_tokens", 2048)),
    }
    if local_config.get("json_mode", True):
        kwargs["response_format"] = {"type": "json_object"}

    with _INFERENCE_LOCK:
        response = llm.create_chat_completion(**kwargs)

    data = _extract_json_object(_response_content(response))
    return _apply_schema(data, schema, fields), response.get("usage")
