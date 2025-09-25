import re
from difflib import SequenceMatcher

import yaml

KTP_FIELDS = [
    "province",
    "city",
    "nik",
    "name",
    "birth_place",
    "birth_date",
    "gender",
    "blood_type",
    "address",
    "rt_rw",
    "kelurahan_desa",
    "kecamatan",
    "religion",
    "marital_status",
    "occupation",
    "nationality",
    "valid_until",
]

FIELD_VARIANTS = {
    "province": [["PROVINSI"], ["PROVINSI."]],
    "city": [["KABUPATEN"], ["KOTA"], ["KOTAMADYA"]],
    "nik": [["NIK"]],
    "name": [["NAMA"]],
    "birth": [["TEMPAT", "TGL", "LAHIR"], ["TEMPAT", "LAHIR"], ["TEMPATL", "LAHIR"], ["TEMPALTGL", "LAHIR"], ["TEMPATTGL", "LAHIR"], ["TEMPAT/TGL", "LAHIR"]],
    "gender": [["JENIS", "KELAMIN"], ["JNS", "KELAMIN"], ["JENIS", "KEL"]],
    "blood_type": [["GOL", "DARAH"], ["GOL", "DRAH"], ["GOL", "DRH"], ["GOLDARAH"]],
    "address": [["ALAMAT"], ["ALMAT"], ["ALAMAT"], ["ALAMAT"], ["ALANAT"], ["ATOMAT"], ["ALAMAP"]],
    "rt_rw": [["RT/RW"], ["RT", "RW"], ["RTRW"], ["RT", "TRW"]],
    "kelurahan_desa": [["KEL", "DESA"], ["KELURAHAN"], ["DESA"], ["KELDESA"], ["KEL/ DESA"], ["KEL/DEEA"], ["KEL/DESA"]],
    "kecamatan": [["KECAMATAN"], ["KEC"]],
    "religion": [["AGAMA"], ["AGAM"]],
    "marital_status": [["STATUS", "PERKAWIN"], ["STATUS", "KAWIN"], ["STATUS", "PERK"]],
    "occupation": [["PEKERJAAN"], ["PEKERJN"], ["PEKERJA"]],
    "nationality": [["KEWARGANEGARAAN"], ["KEWARGANEGARAN"], ["KEWARGANEG"]],
    "valid_until": [["BERLAKU", "HINGGA"], ["BERLAKU"], ["BERLAK"], ["BERLAKU", "H"], ["BERLKU", "HINGGA"]],
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


def _heuristic_parse_ktp(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    data = {}
    province_idx = None

    def store(field, value):
        if value and field not in data:
            normalized = _normalize_whitespace(value)
            if any(char.isalpha() for char in normalized):
                normalized = normalized.upper()
            data[field] = normalized

    for idx, line in enumerate(lines):
        tokens, cleaned = _tokens_with_clean(line)
        upper_line = re.sub(r"[^A-Z0-9/,:=\-\s]", "", line.upper())

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

        if any(_match_label(cleaned, variant) is not None for variant in FIELD_VARIANTS["name"]):
            variant = next(variant for variant in FIELD_VARIANTS["name"] if _match_label(cleaned, variant) is not None)
            value = _extract_after_tokens(line, variant)
            store("name", value or line)

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

        gender_match = re.search(r"JENIS\s*KELAMIN[^A-Z0-9]+([A-Z]+)", upper_line)
        if gender_match:
            store("gender", gender_match.group(1))
        blood_match = re.search(r"GOL\s*DARAH[^A-Z0-9]+([A-Z0-9+-]+)", upper_line)
        if blood_match:
            store("blood_type", blood_match.group(1))

        address_variant = None
        for variant in FIELD_VARIANTS["address"]:
            if _match_label(cleaned, variant) is not None:
                address_variant = variant
                break
        if address_variant:
            value = _extract_after_tokens(line, address_variant)
            store("address", value)

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
            store("rt_rw", value)

        kel_variant = None
        for variant in FIELD_VARIANTS["kelurahan_desa"]:
            if _match_label(cleaned, variant) is not None:
                kel_variant = variant
                break
        if kel_variant:
            value = _extract_after_tokens(line, kel_variant)
            store("kelurahan_desa", value)

        kec_variant = None
        for variant in FIELD_VARIANTS["kecamatan"]:
            if _match_label(cleaned, variant) is not None:
                kec_variant = variant
                break
        if kec_variant:
            value = _extract_after_tokens(line, kec_variant)
            store("kecamatan", value)

        religion_variant = None
        for variant in FIELD_VARIANTS["religion"]:
            if _match_label(cleaned, variant) is not None:
                religion_variant = variant
                break
        if religion_variant:
            value = _extract_after_tokens(line, religion_variant)
            store("religion", value)

        marital_variant = None
        for variant in FIELD_VARIANTS["marital_status"]:
            if _match_label(cleaned, variant) is not None:
                marital_variant = variant
                break
        if marital_variant:
            value = _extract_after_tokens(line, marital_variant)
            store("marital_status", value)

        occupation_variant = None
        for variant in FIELD_VARIANTS["occupation"]:
            if _match_label(cleaned, variant) is not None:
                occupation_variant = variant
                break
        if occupation_variant:
            value = _extract_after_tokens(line, occupation_variant)
            store("occupation", value)

        nationality_variant = None
        for variant in FIELD_VARIANTS["nationality"]:
            if _match_label(cleaned, variant) is not None:
                nationality_variant = variant
                break
        if nationality_variant:
            value = _extract_after_tokens(line, nationality_variant)
            if value:
                store("nationality", value.split()[0])

        valid_variant = None
        for variant in FIELD_VARIANTS["valid_until"]:
            if _match_label(cleaned, variant) is not None:
                valid_variant = variant
                break
        if valid_variant:
            value = _extract_after_tokens(line, valid_variant)
            if value:
                date_match = re.search(r"\d{2}-\d{2}-\d{4}", value)
                store("valid_until", date_match.group(0) if date_match else value)

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
