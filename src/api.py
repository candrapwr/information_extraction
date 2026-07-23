import os
import signal
import sys
import tempfile
import time
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

# Allow running the script directly via `python src/api.py`
src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_root)

from src.config import load_config
from src.local_model import (
    available_templates,
    default_template_name,
    extract_document,
    preload_local_model,
    shutdown_local_model,
)

CONFIG = load_config()
preload_local_model(CONFIG, force=True)

app = Flask(__name__)
BASE_DIR = src_root


def _handle_shutdown(signum, _frame):
    print(f"\nStopping app, received signal {signum}...", file=sys.stderr, flush=True)
    try:
        shutdown_local_model()
    finally:
        os._exit(0)


def _save_upload(file_storage):
    filename = secure_filename(file_storage.filename or "")
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = ".jpg"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        file_storage.save(temp_file.name)
        return temp_file.name
    finally:
        temp_file.close()


def _process_document(image_path, config, template_name=None):
    selected_template = template_name or default_template_name(config)
    data, usage = extract_document(image_path, config, selected_template)
    return data, usage


@app.route("/extract", methods=["POST"])
def extract_data():
    if "file" not in request.files:
        return jsonify({"status": "error", "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "error": "No file selected"}), 400

    image_path = None
    started_at = time.perf_counter()
    try:
        image_path = _save_upload(file)
        template_name = request.form.get("template") or request.args.get("template")
        data, usage = _process_document(image_path, CONFIG, template_name)
        duration_seconds = round(time.perf_counter() - started_at, 3)

        response = {
            "status": "success",
            "template": template_name or default_template_name(CONFIG),
            "data": data,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        if usage:
            response["usage"] = usage
        return jsonify(response)
    except Exception as exc:
        return jsonify({
            "status": "error",
            "error": str(exc),
            "duration_seconds": round(time.perf_counter() - started_at, 3),
            "timestamp": datetime.now().astimezone().isoformat(),
        }), 500
    finally:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass


@app.route("/", methods=["GET", "POST"])
def web_form():
    result = None
    usage = None
    error = None
    timestamp = None
    duration_seconds = None
    templates = available_templates(CONFIG)
    selected_template = request.form.get("template") or default_template_name(CONFIG)

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Pilih berkas gambar terlebih dahulu."
        else:
            image_path = None
            started_at = time.perf_counter()
            try:
                image_path = _save_upload(file)
                result, usage = _process_document(image_path, CONFIG, selected_template)
                duration_seconds = round(time.perf_counter() - started_at, 3)
                timestamp = datetime.now().astimezone().isoformat()
            except Exception as exc:
                error = str(exc)
            finally:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except OSError:
                        pass

    return render_template(
        "web.html",
        result=result,
        usage=usage,
        error=error,
        timestamp=timestamp,
        duration_seconds=duration_seconds,
        templates=templates,
        selected_template=selected_template,
        postman_url=url_for("download_postman_collection"),
    )


@app.route("/downloads/postman", methods=["GET"])
def download_postman_collection():
    return send_from_directory(
        os.path.join(BASE_DIR, "postman"),
        "information_extraction.postman_collection.json",
        as_attachment=True,
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    server_config = CONFIG.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    try:
        port = int(server_config.get("port", 8000))
    except (TypeError, ValueError):
        port = 8000

    debug_value = server_config.get("debug", True)
    if isinstance(debug_value, str):
        debug = debug_value.lower() in {"1", "true", "yes", "on"}
    else:
        debug = bool(debug_value)

    use_reloader = bool(server_config.get("use_reloader", False))
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
