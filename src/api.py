import os
import sys

from flask import Flask, request, jsonify

# Allow running the script directly via `python src/api.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_image
from src.ocr import extract_text, extract_mrz
from src.parser import load_config, parse_ktp, parse_passport

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
    
    try:
        # Preprocess image
        preprocessed_path = preprocess_image(image_path)
        
        # Load config
        config = load_config()
        
        # Extract and parse data
        doc_type = request.form.get('type', 'ktp')  # 'ktp' or 'passport'
        if doc_type == 'passport':
            mrz_data = extract_mrz(preprocessed_path)
            text = extract_text(preprocessed_path, config['tesseract']['lang'])
            result = parse_passport(mrz_data, text)
        else:
            text = extract_text(preprocessed_path, config['tesseract']['lang'])
            result = parse_ktp(text, config)
        
        # Clean up temporary files
        os.remove(image_path)
        os.remove(preprocessed_path)
        
        return jsonify({
            "status": "success",
            "data": result,
            "timestamp": "2025-09-24T12:42:00+07:00"  # WIB
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
