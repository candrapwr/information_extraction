import json
import sys
from datetime import datetime

from src.config import load_config
from src.local_model import default_template_name, extract_document, preload_local_model


def main(image_path, template_name=None):
    try:
        config = load_config()
        preload_local_model(config)
        selected_template = template_name or default_template_name(config)
        data, usage = extract_document(image_path, config, selected_template)

        output = {
            "status": "success",
            "template": selected_template,
            "data": data,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        if usage:
            output["usage"] = usage
        print(json.dumps(output, indent=2, ensure_ascii=False))
    except Exception as exc:
        output = {
            "status": "error",
            "error": str(exc),
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print("Usage: python main.py <image_path> [template]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
