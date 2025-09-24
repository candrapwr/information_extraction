import sys
from src.preprocess import preprocess_image
from src.ocr import extract_text, extract_mrz
from src.parser import load_config, parse_ktp, parse_passport
import json

def main(image_path, doc_type="ktp"):
    """Main function to extract data from image."""
    try:
        # Preprocess image
        preprocessed_path = preprocess_image(image_path)
        
        # Load config
        config = load_config()
        
        # Extract and parse data
        if doc_type == "passport":
            mrz_data = extract_mrz(preprocessed_path)
            text = extract_text(preprocessed_path, config['tesseract']['lang'])
            result = parse_passport(mrz_data, text)
        else:
            text = extract_text(preprocessed_path, config['tesseract']['lang'])
            result = parse_ktp(text, config)
        
        # Output JSON
        output = {
            "status": "success",
            "data": result,
            "timestamp": "2025-09-24T12:42:00+07:00"
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        output = {
            "status": "error",
            "error": str(e),
            "timestamp": "2025-09-24T12:42:00+07:00"
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [ktp|passport]")
        sys.exit(1)
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "ktp"
    main(image_path, doc_type)