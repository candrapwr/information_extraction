<div align="center">

# 🪪 Local GGUF OCR

**Ekstraksi KTP Indonesia memakai model vision lokal — tanpa cloud, tanpa OCR eksternal.**

Upload foto KTP (selfie, scan, atau foto meja) → crop otomatis → verifikasi NIK → resize seragam → model vision membaca → JSON terstruktur.

[Fitur](#-fitur) · [Cara Kerja](#-cara-kerja) · [Instalasi](#-instalasi) · [Penggunaan](#-penggunaan) · [Konfigurasi](#️-konfigurasi)

</div>

---

## ✨ Fitur

- 🔒 **100% lokal** — `llama-cpp-python` load model GGUF dari `./model`, tidak ada request ke cloud
- ✂️ **Auto-crop KTP** — deteksi kartu via edge-density (OpenCV), luruskan kartu miring, anti over-crop
- 🔍 **Verifikasi NIK** — konfirmasi gambar benar-benar KTP via template matching + deteksi baris 16-digit; bukan KTP → error HTTP 400
- 📐 **5 level kualitas** — Very Low → Very High, atur trade-off kecepatan vs detail teks
- 🧠 **Template schema** — KTP (15 field Bahasa Indonesia) & paspor; output dinormalisasi otomatis
- 🧹 **Value cleaner** — validasi nilai terstruktur (jenis_kelamin) & hapus label bocor secara otomatis
- 🖥️ **Web + REST API** — UI AJAX dengan loading state & timer, plus endpoint `/extract`
- ⌨️ **CLI** — ekstraksi langsung dari terminal
- 🐛 **Debug output** — setiap step preprocessing disimpan untuk inspeksi visual

## 🎯 Kenapa bukan OCR biasa?

OCR engine (Tesseract, EasyOCR) butuh dependency berat dan sering kacau di KTP yang miring/kotor. Pendekatan ini berbeda:

| Tahap | Tools | Tujuan |
|-------|-------|--------|
| **Preprocess** | OpenCV (sudah ada) | crop + luruskan + verifikasi KTP |
| **Baca** | Model vision 3B lokal | pahami layout + ekstrak field sekaligus |

Hasilnya: input macam apa pun (foto selfie, scan, foto meja) masuk sebagai **KTP yang bersih dan seragam** ke model → akurasi tinggi, beban model minim.

## 🧩 Cara Kerja

```
┌─────────────┐    ┌──────────────────────────────┐    ┌─────────────┐    ┌──────┐
│  Upload     │───▶│  KTP Preprocessing           │───▶│  Model      │───▶│ JSON │
│  (apa pun)  │    │  1. CLAHE contrast           │    │  Vision 3B  │    │      │
└─────────────┘    │  2. deteksi kartu (edge)     │    │  (lokal)    │    └──────┘
                   │  3. crop + deskew            │    └─────────────┘
                   │  4. verifikasi NIK ✅/❌      │
                   │  5. resize sesuai kualitas   │
                   └──────────────────────────────┘
```

- **Gagal verifikasi NIK** (mis. foto bukan KTP) → HTTP **400** dengan alasan spesifik, sebelum membuang waktu ke model
- **Sudah tight-crop?** algoritma otomatis skip crop (anti over-crop)
- **Debug** disimpan per-request di `debug_ktp/` untuk Anda cek

## 📐 Level Kualitas

Atur trade-off kecepatan vs detail teks kecil. Tersedia di web, API, dan CLI.

| Level | Ukuran | Cocok untuk |
|-------|--------|-------------|
| Very Low | 472×298 | Cepat maksimal, KTP jelas/besar |
| Low | 588×372 | Cepat, teks kecil mungkin sedikit degradasi |
| **Medium** *(default)* | **856×540** | **Akurasi penuh — sweet spot teruji** |
| High | 1024×646 | Lebih detail, sedikit lebih lambat |
| Very High | 1280×808 | Detail maksimal, paling lambat |

> Indikatif (M1, model 3B): `low` ~12s vs `very_high` ~25s per inference. Semua rasio konsisten 1.585 (KTP fisik).

## 📦 Instalasi

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Folder `model/` tidak perlu dicommit. Jika file model belum ada, aplikasi **otomatis download** dari Hugging Face saat startup pertama.

> **Mac Apple Silicon:** `llama-cpp-python` jauh lebih cepat jika dibuild dengan Metal:
> ```bash
> CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python
> ```

## 🚀 Penggunaan

### Web UI
```bash
python src/api.py
```
Buka `http://localhost:8000/` → pilih file → pilih template → pilih **kualitas** → **Ekstrak**.

### REST API
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@/path/to/ktp.jpg" \
  -F "template=ktp" \
  -F "quality=medium"
```
Parameter `quality` opsional (`very_low|low|medium|high|very_high`), default `medium`.

**Respons sukses (HTTP 200):**
```json
{
  "status": "success",
  "template": "ktp",
  "duration_seconds": 16.8,
  "timings": {
    "ktp_quality": "medium",
    "ktp_output_width": 856,
    "ktp_output_height": 540,
    "ktp_crop_method": "perspective",
    "ktp_nik_digits": 16,
    "ktp_nik_score": 0.63,
    "inference_seconds": 15.1
  },
  "data": {
    "provinsi": "DAERAH ISTIMEWA YOGYAKARTA",
    "kabupaten_kota": "GUNUNGKIDUL",
    "nik": "3403162806030001",
    "nama": "ANGGI PRATAMA",
    "tempat_tgl_lahir": "WONOGIRI, 28-06-2003",
    "jenis_kelamin": "LAKI-LAKI",
    "alamat": "PUTAT",
    "rt_rw": "001/011",
    "kel_desa": "SONGBANYU",
    "kecamatan": "GIRISUBO",
    "agama": "ISLAM",
    "status_perkawinan": "BELUM KAWIN",
    "pekerjaan": "NELAYAN/PERIKANAN",
    "kewarganegaraan": "WNI",
    "berlaku_hingga": "SEUMUR HIDUP"
  }
}
```

**Respons gagal — bukan KTP (HTTP 400):**
```json
{
  "status": "error",
  "reason": "nik_not_found",
  "error": "Tidak terdeteksi teks/angka 'NIK' pada gambar (score=0.00, digits=0). Pastikan gambar adalah KTP yang jelas.",
  "template": "ktp"
}
```

### CLI
```bash
python main.py <image_path> [template] [quality]
# contoh:
python main.py /path/to/ktp.jpg ktp low
```
`template` default dari config (`ktp`). `quality` default `medium`.

## ⚙️ Konfigurasi

File: `config/config.yaml`. Salin dari `config.example.yaml`.

### Model & inference
```yaml
local_model:
  model_path: "./model/Nanonets-OCR-s-Q4_0.gguf"
  chat_handler: "qwen2.5-vl"
  ctx_size: 8192
  n_gpu_layers: -1          # offload penuh ke GPU (Metal/CUDA)
  n_threads: 4
  flash_attn: true
  op_offload: true
  max_tokens: 512
  temperature: 0
  default_template: "ktp"
  prompt: "Extract fields from this Indonesian KTP (ID card). For each key, write only the value found on the card. Use null if a field is missing or unreadable. Ignore the 'Gol. Darah' label entirely, do not extract it. Output JSON only."
```

### KTP preprocessing (khusus template `ktp`)
```yaml
ktp_preprocess:
  enabled: true
  target_width: 856         # output default (= level "medium")
  target_height: 540
  ideal_ratio: 1.585        # rasio fisik KTP (85.6mm × 54mm)
  card_ratio_min: 1.2
  card_ratio_max: 2.2
  frame_ratio_min: 1.4      # anti over-crop: frame sudah rasio KTP + blob kecil
  frame_ratio_max: 1.8      #           -> skip crop (tight scan tidak rusak)
  blob_subregion_frac: 0.45
  verify_nik: true          # tolak kalau NIK tidak terdeteksi
  nik_confidence_threshold: 0.45
  nik_strong_threshold: 0.70
  nik_digit_min: 6
  nik_digit_max: 45
  save_steps: true          # simpan setiap step ke debug_ktp/
  debug_dir: "./debug_ktp"
  pad_color: [255, 255, 255]
  jpeg_quality: 90
```

> **Catatan tentang quality:** `target_width/height` dipakai sebagai default. Saat parameter `quality` diberikan (via web/API/CLI), ukuran dari preset quality akan **menimpa** nilai ini.

### Template output
```yaml
templates:
  ktp:
    provinsi: null
    kabupaten_kota: null
    nik: null
    nama: null
    tempat_tgl_lahir: null
    jenis_kelamin: null
    alamat: null
    rt_rw: null
    kel_desa: null
    kecamatan: null
    agama: null
    status_perkawinan: null
    pekerjaan: null
    kewarganegaraan: null
    berlaku_hingga: null
  passport:
    passport_number: null
    name: null
    nationality: null
    date_of_birth: null
    gender: null
    expiration_date: null
    country_code: null
```

> **Catatan:**
> - Key template KTP memakai **Bahasa Indonesia (snake-case)** yang mirip label KTP asli → model 3B lebih presisi mengaitkan key dengan field kartu.
> - Field `gol_darah` **dihilangkan** dari schema (posisinya di kartu sebaris dgn Jenis Kelamin, sering kosong, bikin model rancu). Prompt juga eksplisit menyuruh abaikan label itu.
> - Value cleaner otomatis memvalidasi field terstruktur: `jenis_kelamin` hanya menerima `LAKI-LAKI`/`PEREMPUAN`/`L`/`P`; label bocor & sentinel kosong (`-`, `N/A`) → `null`.

## 🐛 Inspeksi Debug

Setiap request KTP menyimpan step-by-step ke `debug_ktp/<timestamp>_<random>/`:

```
00_input.jpg            # gambar asli
01_after_clahe.jpg      # setelah enhance kontras
02_edges.jpg            # visualisasi deteksi kartu (kotak merah)
03_cropped.jpg          # hasil crop + deskew (jika terdeteksi)
04_nik_region.jpg       # area NIK yang diverifikasi
05_final_WxH.jpg        # output final (ukuran sesuai quality) yg masuk model
_FAIL.txt               # (hanya jika gagal) berisi alasan
```

## 🗂️ Struktur

```text
.
├── config/
│   ├── config.yaml              # konfigurasi aktif (copy dari example)
│   └── config.example.yaml      # template konfigurasi
├── model/                       # GGUF models (auto-download, gitignored)
├── postman/
│   └── information_extraction.postman_collection.json
├── sampel/                      # contoh gambar KTP untuk testing (gitignored)
├── src/
│   ├── api.py                   # Flask web + REST API
│   ├── config.py                # loader config
│   ├── ktp_preprocess.py        # crop + verifikasi NIK + resize + quality
│   ├── local_model.py           # load model + inference + prompt + cleaner
│   └── templates/
│       └── web.html             # UI web (AJAX)
├── debug_ktp/                   # output debug preprocessing (gitignored)
├── main.py                      # entry point CLI
├── requirements.txt
└── README.md
```

## 🔧 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `llama-cpp-python belum terinstall` | `pip install -r requirements.txt` di virtualenv |
| Model gagal load | pastikan dua file `.gguf` di `model/` ada & cocok |
| Crash saat shutdown (Mac) | rebuild Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall llama-cpp-python` |
| KTP valid ditolak | turunkan `nik_confidence_threshold` / `nik_digit_min` di config |
| Bukan KTP lolos | naikkan `nik_digit_min` atau `nik_confidence_threshold` |
| Teks kecil salah baca | naikkan level `quality` (high/very_high) |
| Out of memory | turunkan `ctx_size` atau `n_gpu_layers`, pakai quantization lebih kecil |
| Over-crop pada tight scan | kecilkan `blob_subregion_frac` (mis. 0.35) |

## 📄 License

[MIT](LICENSE)
