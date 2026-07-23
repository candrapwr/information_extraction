"""KTP-specific preprocessing: detect the card, crop it, verify the "NIK"
marker, and resize to a uniform 856x540 (1.585:1) before the image is sent
to the LLM.

Only depends on OpenCV + NumPy + Pillow (already installed). No external OCR.
If the card or the "NIK" marker cannot be found, :class:`KTPDetectionError`
is raised so the caller (api.py) can return a clean HTTP 400.

Card detection uses an edge-density approach that is robust to the range of
inputs we see: tight scans, table photos, and phone photos of someone holding
a card. A KTP is a region of dense text/edges; we locate the densest blob,
merge it with morphology, and bound it with a rotated rectangle. When the
blob is a clean quadrilateral we additionally use a perspective transform to
deskew; otherwise we fall back to a rotated bounding-box crop.
"""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class KTPDetectionError(RuntimeError):
    """Raised when the image cannot be confirmed as a KTP."""

    def __init__(self, reason: str, message: Optional[str] = None):
        self.reason = reason
        super().__init__(message or reason)


# NIK marker sits in the upper-middle band of an Indonesian KTP: below the
# header/province row and above the photo + body fields.
_NIK_BAND_Y = (0.18, 0.34)
_NIK_BAND_X = (0.33, 0.97)


def _project_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(os.path.join(base_dir, path))


def _config(cfg: Optional[Dict], key: str, default):
    return ((cfg or {}).get("ktp_preprocess") or {}).get(key, default)


def _load_cv(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise KTPDetectionError("unreadable_image", f"Cannot read image: {path}")
    return img


class _StepSaver:
    """Optional per-request folder that stores an image at each pipeline step
    so the result can be inspected visually."""

    def __init__(self, config: Dict, enabled: bool, basename: str):
        self.enabled = enabled
        self.dir: Optional[str] = None
        self.steps: Dict[str, str] = {}
        if not enabled:
            return
        debug_dir = _config(config, "debug_dir", "./debug_ktp")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{stamp}_{uuid.uuid4().hex[:6]}_{basename}"
        self.dir = _project_path(os.path.join(debug_dir, folder))
        os.makedirs(self.dir, exist_ok=True)

    def save(self, index: int, name: str, image, quality: int = 90):
        if not self.enabled or self.dir is None:
            return
        path = os.path.join(self.dir, f"{index:02d}_{name}.jpg")
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            buf.tofile(path)
            self.steps[name] = path

    def save_original(self, src_path: str):
        if not self.enabled or self.dir is None:
            return
        dst = os.path.join(self.dir, "00_input.jpg")
        try:
            img = _load_cv(src_path)
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                buf.tofile(dst)
                self.steps["input"] = dst
        except Exception:
            pass

    def fail(self, reason: str, message: str):
        if not self.enabled or self.dir is None:
            return
        with open(os.path.join(self.dir, "_FAIL.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"reason: {reason}\nmessage: {message}\n")


def _apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    planes[0] = clahe.apply(planes[0])
    merged = cv2.merge(planes)
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_corners_from_rect(rect):
    box = cv2.boxPoints(rect)
    return _order_points(box.astype("float32"))


def _perspective_from_corners(img, quad, ideal_ratio: float):
    (tl, tr, br, bl) = quad
    cw = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    ch = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if ch <= 0 or cw <= 0:
        return None
    out_h = int(round(ch))
    out_w = int(round(ch * ideal_ratio))
    src = np.array([tl, tr, br, bl], dtype="float32")
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h))


def _edge_density_blob(img, ratio_min: float, ratio_max: float, edges_debug: List):
    """Find the densest text/card blob via an edge-density map.

    Works for tight scans, table photos, and phone photos. Returns the largest
    rotated bounding rectangle (cv2.minAreaRect) whose side ratio matches a
    card, or None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_debug.clear()
    edges_debug.append(edges)

    h, w = edges.shape[:2]
    img_area = float(h * w)
    block = max(20, min(h, w) // 24)
    block -= block % 2
    bh, bw = h // block, w // block
    if bh < 4 or bw < 4:
        return None
    # Block-mean density via integral image (fast).
    integral = cv2.integral(edges, sdepth=cv2.CV_64F)
    density = np.zeros((bh, bw), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            y0, y1 = i * block, (i + 1) * block
            x0, x1 = j * block, (j + 1) * block
            density[i, j] = (integral[y1, x1] - integral[y0, x1]
                             - integral[y1, x0] + integral[y0, x0]) / float(block * block)

    thr = max(np.percentile(density, 82), density.mean() * 1.3)
    mask = (density > thr).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    kernel_len = max(40, min(h, w) // 12)
    kernel_len -= kernel_len % 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, kernel_len))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.05 * img_area:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if rw <= 0 or rh <= 0:
            continue
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        ratio = long_side / short_side
        if ratio_min <= ratio <= ratio_max:
            candidates.append((area, ratio, rect, c))
    if not candidates:
        return None
    # Prefer blobs whose ratio is close to 1.585 and that are large.
    candidates.sort(
        key=lambda it: (abs(it[1] - 1.585), -it[0] / img_area)
    )
    return candidates[0][2]


def _build_nik_templates():
    """Render the word "NIK" as small binary templates at multiple scales.
    Matched only against the NIK band of the card — not a full OCR pass, so it
    stays cheap (<50ms)."""
    templates = []
    for font in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX):
        for scale in (0.7, 0.85, 1.0, 1.2):
            for thick in (2, 3):
                (tw, th), baseline = cv2.getTextSize("NIK", font, scale, thick)
                canvas = np.zeros((th + baseline + 6, tw + 6), dtype=np.uint8)
                cv2.putText(canvas, "NIK", (3, th + 3), font, scale, 255, thick, cv2.LINE_AA)
                templates.append(canvas)
    return templates


_NIK_TEMPLATES = _build_nik_templates()


def _count_digit_blobs(binary) -> int:
    """Count connected components that look like digit glyphs (taller than
    wide, small). A KTP's NIK row produces many of these in its band."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    count = 0
    for i in range(1, n):
        cw, ch = stats[i, 2], stats[i, 3]
        area = stats[i, 4]
        if 2 < cw < ch * 1.6 and ch > 4 and 15 < area < 4000:
            count += 1
    return count


def _nik_band_roi(card_img, y_band=_NIK_BAND_Y):
    h, w = card_img.shape[:2]
    if h < 20 or w < 40:
        return None
    y1 = int(h * y_band[0])
    y2 = int(h * y_band[1])
    x1 = int(w * _NIK_BAND_X[0])
    x2 = int(w * _NIK_BAND_X[1])
    roi = card_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    scale = max(1.0, 200.0 / max(gray.shape[0], 1))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _verify_nik(card_img, threshold: float) -> Tuple[float, int]:
    """Hybrid NIK verification.

    Returns (template_score, digit_count). A KTP is confirmed when EITHER the
    "NIK" template matches strongly OR the band contains a dense row of digit
    glyphs (the NIK number). The digit count discriminates real KTPs from
    random text far better than template matching alone.

    Because the NIK row's vertical position shifts a little depending on how
    the card was cropped/deskewed, we scan several vertical bands and keep the
    best digit count.
    """
    # Scan a few overlapping vertical bands to tolerate crop misalignment.
    band_candidates = [
        (0.16, 0.30),
        (0.18, 0.34),
        (0.20, 0.38),
        (0.22, 0.40),
    ]
    best_score = 0.0
    best_digits = 0
    for yb in band_candidates:
        gray = _nik_band_roi(card_img, yb)
        if gray is None:
            continue
        variants = [
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 10),
        ]
        for tmpl in _NIK_TEMPLATES:
            th, tw = tmpl.shape[:2]
            for variant in variants:
                if variant.shape[0] < th or variant.shape[1] < tw:
                    continue
                res = cv2.matchTemplate(variant, tmpl, cv2.TM_CCOEFF_NORMED)
                if res.size == 0:
                    continue
                score = float(res.max())
                if score > best_score:
                    best_score = score
                if best_score >= 0.95:
                    break
        digit_count = _count_digit_blobs(variants[1])
        if digit_count > best_digits:
            best_digits = digit_count
    return best_score, best_digits


def _resize_uniform(card_img, target_w: int, target_h: int, pad_color):
    th_ratio = target_w / target_h
    h, w = card_img.shape[:2]
    src_ratio = w / h
    if src_ratio > th_ratio:
        new_w = target_w
        new_h = max(1, round(target_w / src_ratio))
    else:
        new_h = target_h
        new_w = max(1, round(target_h * src_ratio))
    interp = cv2.INTER_AREA if (new_w < w and new_h < h) else cv2.INTER_CUBIC
    resized = cv2.resize(card_img, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _detect_and_crop(img, kcfg, saver) -> Tuple[np.ndarray, str, Dict]:
    """Detect the card and return (cropped_image, method, info)."""
    ratio_min = float(kcfg.get("card_ratio_min", 1.2))
    ratio_max = float(kcfg.get("card_ratio_max", 2.2))
    ideal_ratio = float(kcfg.get("ideal_ratio", 1.585))
    # When the input frame already looks like a card and the detected blob is
    # only a small sub-region (dense text inside the card, not the card edge),
    # we must NOT crop - cropping would zoom into a fragment of the KTP.
    frame_ratio_min = float(kcfg.get("frame_ratio_min", 1.4))
    frame_ratio_max = float(kcfg.get("frame_ratio_max", 1.8))
    blob_subregion_frac = float(kcfg.get("blob_subregion_frac", 0.45))
    h, w = img.shape[:2]
    img_area = float(h * w)
    frame_ratio = w / h
    info: Dict = {}

    edges_buf: List = []
    rect = _edge_density_blob(img, ratio_min, ratio_max, edges_buf)

    if rect is None:
        saver.save(2, "edges_no_card",
                   cv2.cvtColor(edges_buf[0], cv2.COLOR_GRAY2BGR) if edges_buf else img)
        # No card found: keep full frame, let NIK verification decide.
        info["frame_ratio"] = round(frame_ratio, 3)
        return img, "full_frame", info

    # Visualize detection (the rotated rectangle over the edge map).
    if edges_buf:
        vis2 = cv2.cvtColor(edges_buf[0], cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis2, [cv2.boxPoints(rect).astype(int)], -1, (0, 0, 255), 3)
        saver.save(2, "edges", vis2)

    blob_area = cv2.contourArea(cv2.boxPoints(rect).astype(int))
    blob_frac = blob_area / img_area
    info["frame_ratio"] = round(frame_ratio, 3)
    info["card_area_frac"] = round(blob_frac, 3)

    # Guard against over-crop: the frame is already card-shaped but the blob
    # only covers a fragment of it -> the blob is text inside the card, not the
    # card edge. Keep the full frame instead.
    frame_already_card = frame_ratio_min <= frame_ratio <= frame_ratio_max
    if frame_already_card and blob_frac < blob_subregion_frac:
        info["card_ratio"] = round(frame_ratio, 3)
        info["skip_reason"] = "frame_already_card"
        return img, "full_frame", info

    corners = _four_corners_from_rect(rect)
    # Try a perspective deskew using the rotated-rect corners.
    deskewed = _perspective_from_corners(img, corners, ideal_ratio)
    if deskewed is not None and deskewed.shape[0] >= 50 and deskewed.shape[1] >= 50:
        dh, dw = deskewed.shape[:2]
        info["card_ratio"] = round(dw / max(dh, 1), 3)
        saver.save(3, "cropped", deskewed)
        return deskewed, "perspective", info

    return img, "full_frame", info


def preprocess_ktp(
    image_path: str,
    config: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """Run KTP preprocessing and return (new_image_path, timing metadata).

    Raises :class:`KTPDetectionError` if the KTP / NIK marker is not found.
    """
    config = config or {}
    kcfg = config.get("ktp_preprocess") or {}
    target_w = int(kcfg.get("target_width", 856))
    target_h = int(kcfg.get("target_height", 540))
    verify_nik = bool(kcfg.get("verify_nik", True))
    nik_threshold = float(kcfg.get("nik_confidence_threshold", 0.45))
    nik_strong_threshold = float(kcfg.get("nik_strong_threshold", 0.7))
    nik_digit_min = int(kcfg.get("nik_digit_min", 6))
    nik_digit_max = int(kcfg.get("nik_digit_max", 45))
    pad_color = tuple(int(c) for c in kcfg.get("pad_color", [255, 255, 255]))
    jpeg_quality = int(kcfg.get("jpeg_quality", 90))
    save_steps = bool(kcfg.get("save_steps", True))

    basename = os.path.splitext(os.path.basename(image_path))[0][:24] or "ktp"
    saver = _StepSaver(config, save_steps, basename)
    timings: Dict = {}

    try:
        t0 = time.perf_counter()
        img = _load_cv(image_path)
        saver.save_original(image_path)

        clahe = _apply_clahe(img)
        saver.save(1, "after_clahe", clahe)

        # Steps 2-3: detect + crop the card.
        cropped, crop_method, card_info = _detect_and_crop(clahe, kcfg, saver)
        timings["ktp_crop_seconds"] = round(time.perf_counter() - t0, 3)
        timings["ktp_crop_method"] = crop_method
        timings.update({f"ktp_{k}": v for k, v in card_info.items()})

        # Step 4: verify the "NIK" marker (template + digit-row hybrid).
        nik_score = 1.0
        nik_digits = 99
        if verify_nik:
            tn = time.perf_counter()
            nik_score, nik_digits = _verify_nik(cropped, nik_threshold)
            timings["ktp_nik_seconds"] = round(time.perf_counter() - tn, 3)
            timings["ktp_nik_score"] = round(nik_score, 3)
            timings["ktp_nik_digits"] = int(nik_digits)
            ch, cw = cropped.shape[:2]
            y1 = int(ch * _NIK_BAND_Y[0])
            y2 = int(ch * _NIK_BAND_Y[1])
            x1 = int(cw * _NIK_BAND_X[0])
            x2 = int(cw * _NIK_BAND_X[1])
            saver.save(4, "nik_region", cropped[y1:y2, x1:x2])

            # Confirmed if:
            #  - the band shows a realistic row of digit glyphs (the 16-digit
            #    NIK number), bounded above to reject pure-noise images, OR
            #  - the "NIK" template matches strongly AND there is at least
            #    some digit content (avoids false positives on plain text).
            digit_row_ok = nik_digit_min <= nik_digits <= nik_digit_max
            template_ok = nik_score >= nik_strong_threshold and nik_digits >= nik_digit_min
            confirmed = digit_row_ok or template_ok
            if not confirmed:
                msg = (
                    f"Tidak terdeteksi teks/angka 'NIK' pada gambar "
                    f"(score={nik_score:.2f}, digits={nik_digits}). "
                    f"Pastikan gambar adalah KTP yang jelas."
                )
                saver.fail("nik_not_found", msg)
                raise KTPDetectionError("nik_not_found", msg)

        # Step 5: resize to uniform target.
        tf = time.perf_counter()
        final = _resize_uniform(cropped, target_w, target_h, pad_color)
        saver.save(5, f"final_{target_w}x{target_h}", final, quality=jpeg_quality)
        timings["ktp_resize_seconds"] = round(time.perf_counter() - tf, 3)
        timings["ktp_output_width"] = target_w
        timings["ktp_output_height"] = target_h
        timings["ktp_debug_dir"] = saver.dir or ""

        tmp_path = image_path + ".ktp.jpg"
        ok, buf = cv2.imencode(".jpg", final, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise KTPDetectionError("encode_failed", "Gagal meng-encode hasil preprocessing.")
        buf.tofile(tmp_path)
        timings["ktp_total_seconds"] = round(time.perf_counter() - t0, 3)
        return tmp_path, timings

    except KTPDetectionError:
        raise
    except Exception as exc:
        saver.fail("preprocess_error", str(exc))
        raise KTPDetectionError("preprocess_error", str(exc)) from exc
