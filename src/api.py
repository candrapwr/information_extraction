import os
import sys
from datetime import datetime

from flask import Flask, request, jsonify

# Allow running the script directly via `python src/api.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_image
from src.ocr import extract_text, extract_mrz, extract_llm_data
from src.parser import load_config, parse_ktp, parse_passport, normalize_ktp_result

app = Flask(__name__)

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
        provider = (request.form.get('provider') or config.get('ocr', {}).get('default_provider', 'pytesseract')).lower()

        # Preprocess image when appropriate
        if provider != 'llm':
            preprocessed_path = preprocess_image(image_path, config=config)

        doc_type = request.form.get('type', 'ktp')
        ocr_input_path = preprocessed_path or image_path

        usage = None
        if provider == 'llm':
            result, usage = extract_llm_data(image_path, doc_type, config)
            if doc_type.lower() == 'ktp':
                result = normalize_ktp_result(result)
        else:
            text = extract_text(
                ocr_input_path,
                config['tesseract']['lang'],
                provider=provider,
                config=config,
            )
            if doc_type == 'passport':
                mrz_data = extract_mrz(ocr_input_path)
                result = parse_passport(mrz_data, text)
            else:
                result = parse_ktp(text, config)

        response = {
            "status": "success",
            "data": result,
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

if __name__ == '__main__':
    config = load_config()
    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port_override = os.getenv('OCR_API_PORT')
    try:
        port = int(port_override) if port_override else int(server_config.get('port', 5000))
    except (TypeError, ValueError):
        port = 5000
    debug_value = os.getenv('OCR_API_DEBUG', server_config.get('debug', True))
    if isinstance(debug_value, str):
        debug = debug_value.lower() in {'1', 'true', 'yes', 'on'}
    else:
        debug = bool(debug_value)
    app.run(host=host, port=port, debug=debug)
