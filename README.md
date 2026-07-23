<div align="center">

# 🪪 Local GGUF OCR

**Ekstraksi KTP Indonesia memakai model vision lokal — tanpa cloud, tanpa OCR eksternal.**

Upload foto KTP (selfie, scan, atau foto meja) → crop otomatis → verifikasi NIK → resize seragam → model vision membaca → JSON terstruktur.

[Fitur](#-fitur) · [Demo](#-cara-kerja) · [Instalasi](#-instalasi) · [Penggunaan](#-penggunaan) · [Konfigurasi](#-konfigurasi)

</div>

---

## ✨ Fitur

- 🔒 **100% lokal** — `llama-cpp-python` load model GGUF dari `./model`, tidak ada request ke cloud
- ✂️ **Auto-crop KTP** — deteksi kartu via edge-density (OpenCV), luruskan kartu miring, anti over-crop
- 🔍 **Verifikasi NIK** — konfirmasi gambar benar-benar KTP via template matching + deteksi baris 16-digit; bukan KTP → error
- 📐 **Resize seragam 856×540** — semua input dinormalkan ke rasio KTP sebelum masuk model (ringan + akurat)
- 🧠 **Template schema** — KTP (16 field Bahasa Indonesia) & paspor; output dinormalisasi otomatis
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
┌─────────────┐    ┌──────────────────────────┐    ┌─────────────┐    ┌──────────┐
│  Upload     │───▶│  KTP Preprocessing       │───▶│  Model      │───▶│  JSON    │
│  (apa pun)  │    │  1. CLAHE contrast       │    │  Vision 3B  │    │  field   │
└─────────────┘    │  2. deteksi kartu        │    │  (lokal)    │    └──────────┘
                   │  3. crop + deskew        │    └─────────────┘
                   │  4. verifikasi NIK ✅/❌  │
                   │  5. resize 856×540       │
                   └──────────────────────────┘
```

- **Gagal verifikasi NIK** (mis. foto bukan KTP) → HTTP **400** dengan alasan spesifik, sebelum membuang waktu ke model
- **Sudah tight-crop?** algoritma otomatis skip crop (anti over-crop)
- **Debug** disimpan per-request di `debug_ktp/` untuk Anda cek

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
Buka `http://localhost:8000/` → pilih file → pilih template → **Ekstrak**.

### REST API
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@/path/to/ktp.jpg" \
  -F "template=ktp"
```

**Respons sukses (HTTP 200):**
```json
{
  "status": "success",
  "template": "ktp",
  "duration_seconds": 16.8,
  "timings": {
    "ktp_crop_method": "perspective",
    "ktp_nik_digits": 16,
    "ktp_nik_score": 0.63,
    "inference_seconds": 15.1
  },
  "data": {
    "provinsi": "DAERAH ISTIMEWA YOGYAKARTA",
    "kabupaten_kota": "GUNUNGKIDUL",
    "nik": "1111111111111111",
    "nama": "LUNA ANGGI PRATAMA",
    "tempat_tgl_lahir": "WONOGIRI, 28-06-2003",
    "jenis_kelamin": "LAKI-LAKI",
    "agama": "ISLAM",
    "pekerjaan": "NELAYAN/PERIKANAN"
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
python main.py /path/to/ktp.jpg ktp
```

## ⚙️ Konfigurasi

File: `config/config.yaml`. Salin dari `config.example.yaml`.

### Model & inference
```yaml
local_model:
  model_path: "./model/Nanonets-OCR-s-Q4_0.gguf"
  chat_handler: "qwen2.5-vl"
  ctx_size: 8192
  n_gpu_layers: -1          # offload penuh ke GPU (Metal/CUDA)
  flash_attn: true
  op_offload: true
  max_tokens: 512
  temperature: 0
  default_template: "ktp"
  prompt: "Extract fields from this Indonesian KTP (ID card). For each key, write only the value found on the card. Use null if a field is missing or unreadable. Output JSON only."
```

### KTP preprocessing (khusus template `ktp`)
```yaml
ktp_preprocess:
  enabled: true
  target_width: 856         # output seragam
  target_height: 540
  ideal_ratio: 1.585        # rasio fisik KTP (85.6mm × 54mm)
  card_ratio_min: 1.2
  card_ratio_max: 2.2
  frame_ratio_min: 1.4      # anti over-crop: frame sudah rasio KTP + blob kecil
  frame_ratio_max: 1.8      #           -> skip crop (tight scan tidak rusak)
  blob_subregion_frac: 0.45
  verify_nik: true          # tolak kalau NIK tidak terdeteksi
  nik_confidence_threshold: 0.45
  nik_digit_min: 6
  nik_digit_max: 45
  save_steps: true          # simpan setiap step ke debug_ktp/
  debug_dir: "./debug_ktp"
```

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
    gol_darah: null
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
    # ... (lihat config.example.yaml)
```

> **Catatan:** Key template KTP sengaja memakai **Bahasa Indonesia (snake-case)** yang mirip label KTP asli — ini membuat model 3B lebih presisi mengaitkan key dengan field kartu.

## 🐛 Inspeksi Debug

Setiap request KTP menyimpan step-by-step ke `debug_ktp/<timestamp>_<random>/`:

```
00_input.jpg            # gambar asli
01_after_clahe.jpg      # setelah enhance kontras
02_edges.jpg            # visualisasi deteksi kartu (kotak merah)
03_cropped.jpg          # hasil crop + deskew
04_nik_region.jpg       # area NIK yang diverifikasi
05_final_856x540.jpg    # output final yg masuk model
_FAIL.txt               # (hanya jika gagal) berisi alasan
```

## 🗂️ Struktur

```text
.
├── config/
│   ├── config.yaml              # konfigurasi (copy dari example)
│   └── config.example.yaml
├── model/                       # GGUF models (auto-download, gitignored)
├── postman/
│   └── information_extraction.postman_collection.json
├── sampel/                      # contoh gambar KTP untuk testing
├── src/
│   ├── api.py                   # Flask web + REST API
│   ├── config.py                # loader config
│   ├── ktp_preprocess.py        # crop + verifikasi NIK + resize
│   ├── local_model.py           # load model + inference + prompt
│   └── templates/
│       └── web.html             # UI web
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
| Lambat di Mac | rebuild dengan Metal: `CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall llama-cpp-python` |
| KTP valid ditolak | turunkan `nik_confidence_threshold` / `nik_digit_min` di config |
| Bukan KTP lolos | naikkan `nik_digit_min` atau `nik_confidence_threshold` |
| Out of memory | turunkan `ctx_size` atau `n_gpu_layers`, pakai quantization lebih kecil |
| Over-crop pada tight scan | kecilkan `blob_subregion_frac` (mis. 0.35) |

## 📄 License

[MIT](LICENSE)
