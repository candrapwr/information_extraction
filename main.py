import sys
from datetime import datetime
from src.preprocess import preprocess_image
from src.ocr import extract_text, extract_mrz, extract_llm_data
from src.parser import (
    load_config,
    normalize_ktp_result,
    parse_ktp,
    parse_passport,
    validate_result,
)
import json

def main(image_path, doc_type="ktp", provider=None):
    """Main function to extract data from image."""
    try:
        # Load config
        config = load_config()

        provider = (provider or config.get('ocr', {}).get('default_provider', 'pytesseract')).lower()

        # Preprocess image when using OCR engines that benefit from it
        preprocessed_path = None
        if provider == "pytesseract":
            preprocessed_path = preprocess_image(image_path, config=config, provider=provider)
        ocr_input_path = preprocessed_path or image_path
        
        # Extract and parse data
        usage = None
        doc_type_normalized = (doc_type or "ktp").lower()
        if provider == "llm":
            result, usage = extract_llm_data(image_path, doc_type_normalized, config)
            if doc_type_normalized == "ktp":
                result = normalize_ktp_result(result)
        else:
            text = extract_text(
                ocr_input_path,
                config['tesseract']['lang'],
                provider=provider,
                config=config,
            )
            if doc_type_normalized == "passport":
                mrz_data = extract_mrz(ocr_input_path)
                result = parse_passport(mrz_data, text)
            else:
                result = normalize_ktp_result(parse_ktp(text, config))

        is_valid = validate_result(result, doc_type_normalized)

        # Output JSON
        output = {
            "status": "success",
            "data": result,
            "valid": is_valid,
            "timestamp": datetime.now().astimezone().isoformat()
        }
        if usage:
            output["usage"] = usage
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
    except Exception as e:
        output = {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().astimezone().isoformat()
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [ktp|passport] [pytesseract|easyocr|llm]")
        sys.exit(1)
    image_path = sys.argv[1]
    doc_type = sys.argv[2] if len(sys.argv) > 2 else "ktp"
    provider_arg = sys.argv[3] if len(sys.argv) > 3 else None
    main(image_path, doc_type, provider_arg)
