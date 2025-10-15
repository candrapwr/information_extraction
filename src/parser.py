import re
from difflib import SequenceMatcher

import yaml

KTP_FIELDS = [
    "address",
    "birth_date",
    "birth_place",
    "city",
    "gender",
    "kecamatan",
    "kelurahan_desa",
    "marital_status",
    "name",
    "nik",
    "province",
    "religion",
    "rt_rw",
]

PASSPORT_FIELDS = [
    "passport_number",
    "name",
    "nationality",
    "date_of_birth",
    "gender",
    "expiration_date",
    "country_code",
]

FIELD_VARIANTS = {
    "province": [["PROVINSI"], ["PROVINSI."]],
    "city": [["KABUPATEN"], ["KOTA"], ["KOTAMADYA"]],
    "nik": [["NIK"]],
    "name": [["NAMA"]],
    "birth": [["TEMPAT", "TGL", "LAHIR"], ["TEMPAT", "LAHIR"], ["TEMPATL", "LAHIR"], ["TEMPALTGL", "LAHIR"], ["TEMPATTGL", "LAHIR"], ["TEMPAT/TGL", "LAHIR"]],
    "gender": [["JENIS", "KELAMIN"], ["JNS", "KELAMIN"], ["JENIS", "KEL"]],
    "address": [["ALAMAT"], ["ALMAT"], ["ALAMAT"], ["ALAMAT"], ["ALANAT"], ["ATOMAT"], ["ALAMAP"]],
    "rt_rw": [["RT/RW"], ["RT", "RW"], ["RTRW"], ["RT", "TRW"]],
    "kelurahan_desa": [["KEL", "DESA"], ["KELURAHAN"], ["DESA"], ["KELDESA"], ["KEL/ DESA"], ["KEL/DEEA"], ["KEL/DESA"]],
    "kecamatan": [["KECAMATAN"], ["KEC"]],
    "religion": [["AGAMA"], ["AGAM"]],
    "marital_status": [["STATUS", "PERKAWIN"], ["STATUS", "KAWIN"], ["STATUS", "PERK"]],
}

_BLOCK_KEYWORDS = {
    "PROVINSI",
    "KOTA",
    "KABUPATEN",
    "KOTAMADYA",
    "NIK",
    "NAMA",
    "TEMPAT",
    "TGL",
    "LAHIR",
    "JENIS",
    "KELAMIN",
    "GOL",
    "DARAH",
    "ALAMAT",
    "RT",
    "RW",
    "KELDESA",
    "KEL",
    "DESA",
    "KECAMATAN",
    "AGAMA",
    "STATUS",
    "PERKAWINAN",
    "PEKERJAAN",
    "KEWARGANEGARAAN",
    "BERLAKU",
    "PEREMPUAN",
    "WANITA",
    "PRIA",
    "LAKI",
    "POKERJAAN",
}

_RELIGION_KEYWORDS = {
    "ISLAM",
    "KRISTEN",
    "PROTESTAN",
    "KATOLIK",
    "HINDU",
    "BUDDHA",
    "KONGHUCU",
    "KEPERCAYAAN",
}


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_token(token: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", token.upper())


def _tokens_with_clean(line: str):
    tokens = [tok for tok in re.split(r"\s+", line) if tok]
    cleaned = [_clean_token(tok) for tok in tokens]
    return tokens, cleaned


def _is_similar(a: str, b: str, threshold: float = 0.72) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _match_label(clean_tokens, label_tokens):
    position = 0
    for label in label_tokens:
        label_clean = _clean_token(label)
        matched = False
        for idx in range(position, len(clean_tokens)):
            if _is_similar(clean_tokens[idx], label_clean):
                position = idx + 1
                matched = True
                break
        if not matched:
            return None
    return position


def _remove_leading_delimiters(tokens):
    while tokens:
        token = tokens[0]
        if token in {":", "=", "-", ".", "/"}:
            tokens = tokens[1:]
            continue
        if len(token) == 1 and not token.isalpha():
            tokens = tokens[1:]
            continue
        break
    return tokens


def _extract_after_tokens(line: str, label_tokens):
    tokens, cleaned = _tokens_with_clean(line)
    end_idx = _match_label(cleaned, label_tokens)
    if end_idx is None:
        return None
    remaining = _remove_leading_delimiters(tokens[end_idx:])
    value = " ".join(remaining).strip(" :=-")
    return value or None


def _find_first_matching_line(lines, start_idx):
    for line in lines[start_idx:]:
        cleaned = _clean_token(line)
        if not line:
            continue
        if any(char.isdigit() for char in line):
            continue
        if ":" in line or "=" in line:
            continue
        if len(cleaned) < 3:
            continue
        return _normalize_whitespace(line)
    return None


def _iter_following_lines(lines, start_idx, max_offset=6):
    for offset in range(1, max_offset + 1):
        next_idx = start_idx + offset
        if next_idx >= len(lines):
            break
        candidate = lines[next_idx].strip()
        if candidate:
            yield candidate


def _is_probable_label(line: str) -> bool:
    tokens, cleaned = _tokens_with_clean(line)
    if not tokens:
        return False
    for variants in FIELD_VARIANTS.values():
        for variant in variants:
            if _match_label(cleaned, variant) is not None:
                return True
    return False


def _contains_block_keyword(text: str) -> bool:
    tokens = re.split(r"\s+", text.upper())
    for tok in tokens:
        cleaned = re.sub(r"[^A-Z]", "", tok)
        if cleaned and any(block in cleaned for block in _BLOCK_KEYWORDS):
            return True
    return False


def _candidate_value_for(field: str, line: str):
    raw = line.strip()
    stripped = raw.lstrip(" :=-")
    stripped = re.sub(r"^[^A-Z0-9]+", "", stripped)
    if not stripped:
        return None
    if _is_probable_label(stripped):
        return None
    upper = stripped.upper()
    if field == "nik":
        digits = re.findall(r"\d{16}", re.sub(r"[^0-9]", "", stripped))
        if digits:
            return max(digits, key=len)
        digits = re.findall(r"\d{4,}", stripped)
        if digits:
            return max(digits, key=len)
    elif field == "name":
        candidate = re.sub(r"^[^A-Z]+", "", stripped)
        upper_candidate = candidate.upper()
        if candidate and not _contains_block_keyword(upper_candidate) and not any(char.isdigit() for char in candidate) and len(candidate) >= 3:
            return candidate
    elif field == "gender":
        match = re.search(r"(LAKI[-\s]*LAKI|PEREMPUAN|PRIA|WANITA|LAKI|L|P)", upper)
        if match:
            token = match.group(1)
            if token in {"L", "LAKI"}:
                token = "LAKI-LAKI"
            if token == "P":
                token = "PEREMPUAN"
            return token
    elif field == "address":
        if not _contains_block_keyword(upper) and (upper.startswith(("JL", "JLN", "JALAN")) or any(char.isdigit() for char in stripped) or len(stripped) >= 10):
            return stripped
    elif field == "rt_rw":
        match = re.search(r"(\d{1,3})\s*[/|-]\s*(\d{1,3})", stripped)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    elif field in {"kelurahan_desa", "kecamatan"}:
        candidate = re.sub(r"[^A-Z\s]", "", upper).strip()
        if candidate:
            tokens = [tok for tok in candidate.split() if tok]
            if any(tok in {"JL", "JALAN", "JLN"} for tok in tokens):
                return None
            if not _contains_block_keyword(candidate) and len(candidate) >= 3 and len(tokens) <= 3:
                return candidate
    elif field == "religion":
        token = re.sub(r"[^A-Z]", "", upper)
        if token:
            for keyword in _RELIGION_KEYWORDS:
                if keyword in token:
                    return keyword
        return None
    elif field == "marital_status":
        candidate = re.sub(r"[^A-Z\s]", "", upper).strip()
        if candidate and not _contains_block_keyword(candidate) and not any(char.isdigit() for char in candidate):
            return candidate
    return None


def _should_replace(field: str, existing: str, new_value: str, record: dict) -> bool:
    if not existing or existing == "Not found":
        return True
    if existing == new_value:
        return False
    if _contains_block_keyword(existing):
        return True
    if field == "name":
        keywords = {"PROVINSI", "KOTA", "KABUPATEN", "KOTAMADYA"}
        if any(keyword in existing for keyword in keywords) and not any(keyword in new_value for keyword in keywords):
            return True
        if len(new_value.split()) >= 2 and len(existing.split()) <= 1:
            return True
        city = record.get("city")
        if city and existing == city and new_value != city:
            return True
    if field in {"kelurahan_desa", "kecamatan"}:
        if any(keyword in existing for keyword in {"JL", "JALAN"}) and not any(keyword in new_value for keyword in {"JL", "JALAN"}):
            return True
    if field == "address":
        if not any(char.isdigit() for char in existing) and any(char.isdigit() for char in new_value):
            return True
    return False


def _heuristic_parse_ktp(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data = {}
    province_idx = None
    pending_fields = []

    def store(field, value):
        if not value:
            return
        normalized = _normalize_whitespace(value)
        if any(char.isalpha() for char in normalized):
            normalized = normalized.upper()
        existing = data.get(field)
        if not existing or _should_replace(field, existing, normalized, data):
            data[field] = normalized
        if field in pending_fields:
            pending_fields.remove(field)

    def queue_pending(field):
        if field not in data and field not in pending_fields:
            pending_fields.append(field)

    for idx, line in enumerate(lines):
        for field in pending_fields[:]:
            candidate_value = _candidate_value_for(field, line)
            if candidate_value:
                if field == "kecamatan" and data.get("kelurahan_desa") == candidate_value:
                    continue
                store(field, candidate_value)
                if field in pending_fields:
                    pending_fields.remove(field)

        tokens, cleaned = _tokens_with_clean(line)
        upper_line = re.sub(r"[^A-Z0-9/,:=\-\s]", "", line.upper())

        if "religion" not in data:
            upper_compact = re.sub(r"[^A-Z]", "", upper_line)
            for keyword in _RELIGION_KEYWORDS:
                if upper_compact == keyword:
                    store("religion", keyword)
                    break

        existing_name = data.get("name")
        if existing_name:
            city_value = data.get("city")
            if (city_value and existing_name == city_value) or _contains_block_keyword(existing_name):
                candidate = _candidate_value_for("name", line)
                if candidate and candidate != existing_name:
                    store("name", candidate)

        if any(_match_label(cleaned, variant) is not None for variant in FIELD_VARIANTS["province"]):
            variant = next(variant for variant in FIELD_VARIANTS["province"] if _match_label(cleaned, variant) is not None)
            province_idx = idx
            value = _extract_after_tokens(line, variant)
            store("province", value or line)

        city_variant = None
        for variant in FIELD_VARIANTS["city"]:
            if _match_label(cleaned, variant) is not None:
                city_variant = variant
                break
        if city_variant:
            value = _extract_after_tokens(line, city_variant)
            store("city", value or line)

        if any(_match_label(cleaned, variant) is not None for variant in FIELD_VARIANTS["nik"]):
            numbers = re.findall(r"\d{4,}", upper_line)
            if numbers:
                store("nik", max(numbers, key=len))
                if "name" not in data:
                    for candidate in lines[idx + 1: idx + 4]:
                        cleaned_candidate = candidate.lstrip(":= ").strip()
                        if cleaned_candidate and cleaned_candidate == cleaned_candidate.upper() and not any(char.isdigit() for char in cleaned_candidate):
                            store("name", cleaned_candidate)
                            break
            else:
                for candidate in _iter_following_lines(lines, idx, max_offset=6):
                    if _is_probable_label(candidate):
                        continue
                    candidate_digits = re.findall(r"\d{4,}", candidate)
                    if candidate_digits:
                        store("nik", max(candidate_digits, key=len))
                        break
                if "name" not in data:
                    for candidate in _iter_following_lines(lines, idx, max_offset=6):
                        if _is_probable_label(candidate):
                            continue
                        cleaned_candidate = candidate.lstrip(":= ").strip()
                        if cleaned_candidate and cleaned_candidate == cleaned_candidate.upper() and not any(char.isdigit() for char in cleaned_candidate):
                            store("name", cleaned_candidate)
                            break

        if any(_match_label(cleaned, variant) is not None for variant in FIELD_VARIANTS["name"]):
            variant = next(variant for variant in FIELD_VARIANTS["name"] if _match_label(cleaned, variant) is not None)
            value = _extract_after_tokens(line, variant)
            if value:
                store("name", value)
            else:
                queue_pending("name")

        birth_variant = None
        for variant in FIELD_VARIANTS["birth"]:
            if _match_label(cleaned, variant) is not None:
                birth_variant = variant
                break
        if birth_variant:
            value = _extract_after_tokens(line, birth_variant)
            if value:
                parts = [part.strip() for part in value.split(",") if part.strip()]
                if parts:
                    store("birth_place", parts[0])
                if len(parts) > 1:
                    date_match = re.search(r"\d{2}-\d{2}-\d{4}", parts[1])
                    if date_match:
                        store("birth_date", date_match.group(0))
            else:
                match = re.search(r"([A-Z\s]+),\s*(\d{2}-\d{2}-\d{4})", upper_line)
                if match:
                    store("birth_place", match.group(1).strip().replace("TEMPAT", "").strip())
                    store("birth_date", match.group(2))

        gender_variant = None
        for variant in FIELD_VARIANTS["gender"]:
            if _match_label(cleaned, variant) is not None:
                gender_variant = variant
                break
        if gender_variant:
            value = _extract_after_tokens(line, gender_variant)
            if value:
                store("gender", value)
            else:
                for candidate in _iter_following_lines(lines, idx, max_offset=6):
                    if _is_probable_label(candidate):
                        continue
                    match = re.search(r"(LAKI[-\s]*LAKI|PEREMPUAN|PRIA|WANITA|LAKI-LAKI|LAKI LAKI|PEREMPUAN|LAKI|L|P)", candidate.upper())
                    if match:
                        store("gender", match.group(1))
                        break
                queue_pending("gender")
        gender_match = re.search(r"JENIS\s*KELAMIN[^A-Z0-9]+([A-Z]+)", upper_line)
        if gender_match:
            store("gender", gender_match.group(1))

        address_variant = None
        for variant in FIELD_VARIANTS["address"]:
            if _match_label(cleaned, variant) is not None:
                address_variant = variant
                break
        if address_variant:
            value = _extract_after_tokens(line, address_variant)
            if value:
                store("address", value)
            else:
                queue_pending("address")

        rt_variant = None
        for variant in FIELD_VARIANTS["rt_rw"]:
            if _match_label(cleaned, variant) is not None:
                rt_variant = variant
                break
        if rt_variant:
            value = _extract_after_tokens(line, rt_variant)
            if not value:
                digits = re.findall(r"\d{1,3}", upper_line)
                if len(digits) >= 2:
                    value = f"{digits[0]}/{digits[1]}"
            if value:
                store("rt_rw", value)
            else:
                queue_pending("rt_rw")

        kel_variant = None
        for variant in FIELD_VARIANTS["kelurahan_desa"]:
            if _match_label(cleaned, variant) is not None:
                kel_variant = variant
                break
        if kel_variant:
            value = _extract_after_tokens(line, kel_variant)
            if value:
                store("kelurahan_desa", value)
            else:
                queue_pending("kelurahan_desa")

        kec_variant = None
        for variant in FIELD_VARIANTS["kecamatan"]:
            if _match_label(cleaned, variant) is not None:
                kec_variant = variant
                break
        if kec_variant:
            value = _extract_after_tokens(line, kec_variant)
            if value:
                store("kecamatan", value)
            else:
                queue_pending("kecamatan")

        religion_variant = None
        for variant in FIELD_VARIANTS["religion"]:
            if _match_label(cleaned, variant) is not None:
                religion_variant = variant
                break
        if religion_variant:
            value = _extract_after_tokens(line, religion_variant)
            if value:
                store("religion", value)
            else:
                queue_pending("religion")

        marital_variant = None
        for variant in FIELD_VARIANTS["marital_status"]:
            if _match_label(cleaned, variant) is not None:
                marital_variant = variant
                break
        if marital_variant:
            value = _extract_after_tokens(line, marital_variant)
            if value:
                store("marital_status", value)
            else:
                queue_pending("marital_status")

    if province_idx is not None and "city" not in data:
        candidate = _find_first_matching_line(lines, province_idx + 1)
        if candidate:
            data["city"] = candidate

    if "address" not in data:
        for idx, line in enumerate(lines):
            if "RT" in line.upper() and "RW" in line.upper() and idx > 0:
                prev_line = lines[idx - 1]
                if prev_line and ":" not in prev_line and "=" not in prev_line:
                    store("address", prev_line)
                break

    if "birth_date" not in data or "birth_place" not in data:
        match = re.search(r"([A-Z\s]{3,}),\s*(\d{2}-\d{2}-\d{4})", re.sub(r"[^A-Z0-9,\s-]", "", text.upper()))
        if match:
            place = match.group(1).strip()
            date = match.group(2)
            if "TEMPAT" in place:
                place = place.split()[-1]
            store("birth_place", place)
            store("birth_date", date)

    return data


def _is_field_present(value):
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False
        if normalized.lower() == "not found":
            return False
    return True


def validate_result(data, doc_type):
    """Return True when all expected fields are present with meaningful values."""
    if not isinstance(data, dict):
        return False
    fields = []
    doc_type_lower = (doc_type or "").lower()
    if doc_type_lower == "passport":
        fields = PASSPORT_FIELDS
    else:
        fields = KTP_FIELDS
    return all(_is_field_present(data.get(field)) for field in fields)

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_ktp(text, config):
    """Parse KTP data using regex from config."""
    fields_config = config['templates']['ktp']['fields']
    result = {field: "Not found" for field in fields_config}
    for field, pattern in fields_config.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1)
            if isinstance(value, str):
                value = value.splitlines()[0]
            result[field] = str(value).strip()
    heuristics = _heuristic_parse_ktp(text)
    for key, value in heuristics.items():
        if value and result.get(key, "Not found") == "Not found":
            result[key] = value
    for field in KTP_FIELDS:
        result.setdefault(field, "Not found")
    return result


def normalize_ktp_result(data):
    """Ensure KTP result dict contains all expected fields with default fallbacks."""
    normalized = {}
    source = data or {}
    for field in KTP_FIELDS:
        raw_value = source.get(field, "Not found")
        if raw_value is None:
            normalized[field] = "Not found"
            continue
        if isinstance(raw_value, str):
            value = raw_value.strip()
            normalized[field] = value if value else "Not found"
        else:
            normalized[field] = raw_value
    return normalized


def parse_passport(mrz_data, text):
    """Parse passport data (MRZ + additional text)."""
    result = {}
    if mrz_data:
        result = {
            "passport_number": mrz_data.get('number', "Not found"),
            "name": f"{mrz_data.get('names', '')} {mrz_data.get('surname', '')}".strip(),
            "nationality": mrz_data.get('nationality', "Not found"),
            "date_of_birth": mrz_data.get('date_of_birth', "Not found"),
            "gender": mrz_data.get('sex', "Not found"),
            "expiration_date": mrz_data.get('expiration_date', "Not found"),
            "country_code": mrz_data.get('country', "Not found")
        }
    # Add parsing for non-MRZ fields if needed
    return result
