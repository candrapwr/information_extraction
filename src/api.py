import os
import sys
import tempfile
from datetime import datetime

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

# Allow running the script directly via `python src/api.py`
src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(src_root)

from src.preprocess import preprocess_image
from src.ocr import extract_text, extract_mrz, extract_llm_data
from src.parser import (
    load_config,
    normalize_ktp_result,
    parse_ktp,
    parse_passport,
    validate_result,
)

app = Flask(__name__)
BASE_DIR = src_root


def _process_document(image_path, doc_type, provider, config):
    """Run the OCR pipeline and return (result, usage, preprocessed_path, doc_type, provider, is_valid)."""
    cfg = config or {}
    doc_type_normalized = (doc_type or "ktp").lower()
    default_provider = cfg.get("ocr", {}).get("default_provider", "pytesseract")
    provider_normalized = (provider or default_provider).lower()

    preprocessed_path = None
    if provider_normalized == 'pytesseract':
        preprocessed_path = preprocess_image(image_path, config=cfg, provider=provider_normalized)
    ocr_input_path = preprocessed_path or image_path

    usage = None
    if provider_normalized == 'llm':
        result, usage = extract_llm_data(image_path, doc_type_normalized, cfg)
        if doc_type_normalized == 'ktp':
            result = normalize_ktp_result(result)
    else:
        lang = cfg.get('tesseract', {}).get('lang', 'eng')
        text = extract_text(
            ocr_input_path,
            lang,
            provider=provider_normalized,
            config=cfg,
        )
        if doc_type_normalized == 'passport':
            mrz_data = extract_mrz(ocr_input_path)
            result = parse_passport(mrz_data, text)
        else:
            result = normalize_ktp_result(parse_ktp(text, cfg))

    is_valid = validate_result(result, doc_type_normalized)
    return result, usage, preprocessed_path, doc_type_normalized, provider_normalized, is_valid

@app.route('/extract', methods=['POST'])
def extract_data():
    """API endpoint to extract data from KTP or passport."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save temporary file
    image_path = f"temp_{file.filename}"
    file.save(image_path)

    preprocessed_path = None
    try:
        # Load config & determine provider
        config = load_config()
        provider = request.form.get('provider')
        doc_type = request.form.get('type', 'ktp')
        (
            result,
            usage,
            preprocessed_path,
            _,
            _,
            is_valid,
        ) = _process_document(image_path, doc_type, provider, config)

        response = {
            "status": "success",
            "data": result,
            "valid": is_valid,
            "timestamp": datetime.now().astimezone().isoformat()
        }
        if usage:
            response["usage"] = usage
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().astimezone().isoformat(),
        }), 500
    finally:
        for path in {image_path, preprocessed_path}:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.route('/', methods=['GET', 'POST'])
def web_form():
    """Render a minimal web UI for uploading documents."""
    config = load_config()
    providers = ['pytesseract', 'easyocr', 'llm']
    doc_types = ['ktp', 'passport']

    default_provider = config.get('ocr', {}).get('default_provider', 'pytesseract')
    selected_provider = (request.form.get('provider') or default_provider).lower()
    selected_doc_type = (request.form.get('type') or 'ktp').lower()

    result = None
    usage = None
    error = None
    timestamp = None
    is_valid = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = "Pilih berkas gambar terlebih dahulu."
        else:
            _, ext = os.path.splitext(file.filename)
            if not ext:
                ext = '.jpg'
            preprocessed_path = None
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                    temp_path = temp_file.name
                    file.save(temp_file.name)

                (
                    result,
                    usage,
                    preprocessed_path,
                    selected_doc_type,
                    selected_provider,
                    is_valid,
                ) = _process_document(temp_path, selected_doc_type, selected_provider, config)
                timestamp = datetime.now().astimezone().isoformat()
            except Exception as exc:
                error = str(exc)
            finally:
                for path in {temp_path, preprocessed_path}:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

    return render_template(
        "web.html",
        providers=providers,
        doc_types=doc_types,
        selected_provider=selected_provider,
        selected_doc_type=selected_doc_type,
        default_provider=default_provider,
        result=result,
        usage=usage,
        error=error,
        timestamp=timestamp,
        is_valid=is_valid,
        postman_url=url_for("download_postman_collection"),
    )


@app.route("/downloads/postman", methods=["GET"])
def download_postman_collection():
    """Serve the Postman collection for the API."""
    return send_from_directory(
        os.path.join(BASE_DIR, "postman"),
        "information_extraction.postman_collection.json",
        as_attachment=True,
    )

if __name__ == '__main__':
    config = load_config()
    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port_override = os.getenv('OCR_API_PORT')
    try:
        port = int(port_override) if port_override else int(server_config.get('port', 8888))
    except (TypeError, ValueError):
        port = 5000
    debug_value = os.getenv('OCR_API_DEBUG', server_config.get('debug', True))
    if isinstance(debug_value, str):
        debug = debug_value.lower() in {'1', 'true', 'yes', 'on'}
    else:
        debug = bool(debug_value)
    app.run(host=host, port=port, debug=debug)
