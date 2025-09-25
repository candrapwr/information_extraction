import math
import os

import cv2


def _compute_scale_factor(width, height, max_width=None, max_height=None):
    scale_w = (max_width / width) if max_width and width > max_width else 1.0
    scale_h = (max_height / height) if max_height and height > max_height else 1.0
    return min(scale_w, scale_h, 1.0)


def preprocess_image(image_path, config=None):
    """Preprocess image to improve OCR accuracy and control image size."""
    cfg = (config or {}).get("preprocess", {})
    max_width = cfg.get("max_width")
    max_height = cfg.get("max_height")
    max_filesize_mb = cfg.get("max_filesize_mb")
    jpeg_quality = int(cfg.get("jpeg_quality", 90))

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid.")

    original_height, original_width = img.shape[:2]
    scale = _compute_scale_factor(original_width, original_height, max_width, max_height)

    if max_filesize_mb:
        try:
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        except OSError:
            file_size_mb = None
        if file_size_mb and file_size_mb > max_filesize_mb:
            ratio = max_filesize_mb / file_size_mb
            scale = min(scale, math.sqrt(max(ratio, 0)))

    if scale < 1.0:
        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)

    # Contrast enhancement using CLAHE
    clip_limit = float(cfg.get("clahe_clip", 2.0))
    grid_size = cfg.get("clahe_grid", [8, 8])
    tile_grid = tuple(grid_size) if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2 else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    enhanced = clahe.apply(denoised)

    temp_path = "temp_preprocessed.jpg"
    imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(jpeg_quality, 100))]
    cv2.imwrite(temp_path, enhanced, imwrite_params)
    return temp_path
