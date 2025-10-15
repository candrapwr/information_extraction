# KTP & Passport OCR

Pipeline untuk mengekstrak informasi dari KTP Indonesia dan paspor internasional menggunakan Python, OpenCV, serta beberapa engine OCR (pytesseract, EasyOCR, atau API LLM). Proyek ini menyediakan skrip CLI serta layanan Flask sederhana untuk integrasi ke aplikasi lain.

## Fitur Utama
- Ekstraksi elemen inti KTP: NIK, nama, alamat lengkap, RT/RW, kelurahan/desa, kecamatan, kota/kabupaten, provinsi, tempat & tanggal lahir, jenis kelamin, agama, serta status perkawinan.
- Pembacaan MRZ paspor menggunakan *passporteye* serta OCR tambahan untuk informasi non-MRZ.
- Pra-pemrosesan citra (grayscale + Otsu thresholding) agar hasil OCR lebih stabil.
- Otomatis mengecilkan resolusi/ukuran file yang terlalu besar sebelum OCR.
- Pilihan engine OCR: pytesseract (default), EasyOCR, atau API LLM (misal Google Gemini) dengan parameter yang sama di CLI/API.
- Heuristik toleran terhadap hasil OCR yang noisy (mis-read huruf, tanda baca hilang, dll.).
- Konfigurasi fleksibel melalui `config/config.yaml` untuk jalur Tesseract, bahasa OCR, dan pola regex.

## Prasyarat
- Python 3.9+ (dikembangkan dengan 3.13).
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) beserta data bahasa yang relevan. Sangat disarankan meng-instal paket bahasa Indonesia (`ind.traineddata`).
- Dependensi Python dari `requirements.txt`.

## Instalasi
```bash
# opsional: buat virtualenv
python -m venv venv
source venv/bin/activate

# instal dependensi
pip install -r requirements.txt
```

### Instalasi Tesseract
- **macOS** (Homebrew):
  ```bash
  brew install tesseract
  ```
  Setelah terpasang, jalankan `which tesseract`. Pada Mac Apple Silicon biasanya `tesseract` berada di `/opt/homebrew/bin/tesseract`, sedangkan Mac Intel di `/usr/local/bin/tesseract`.

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr tesseract-ocr-ind
  ```
  Jalankan `which tesseract` untuk memastikan executable berada di `/usr/bin/tesseract` (default pada sebagian besar distro).

### Konfigurasi Tesseract
Perbarui `config/config.yaml` agar `tesseract.path` menunjuk ke hasil `which tesseract`. Misal:
```yaml
tesseract:
  path: "/opt/homebrew/bin/tesseract"
  lang: "eng+ind"
```
Jika Anda menghapus baris `path`, aplikasi otomatis memakai lokasi yang ditemukan oleh `shutil.which("tesseract")`.

Salin template `config/config.example.yaml` menjadi `config/config.yaml`, lalu sesuaikan nilainya (termasuk API key jika menggunakan mode LLM). Edit `config/config.yaml` untuk menyesuaikan bahasa (`tesseract.lang`) maupun opsi lain. Untuk mode EasyOCR dan LLM, sesuaikan juga bagian `easyocr` serta `llm` sesuai kebutuhan.

## Struktur Proyek
```
.
├── config/
│   └── config.yaml          # Jalur Tesseract + pola regex
├── src/
│   ├── api.py               # Flask API
│   ├── preprocess.py        # Fungsi pra-pemrosesan citra
│   ├── ocr.py               # Wrapper pytesseract/EasyOCR/LLM + MRZ
│   └── parser.py            # Parser KTP & paspor
├── main.py                  # Entry-point CLI
├── requirements.txt
└── README.md
```

## Penggunaan CLI
```bash
python main.py <path_gambar> [ktp|passport] [pytesseract|easyocr|llm]

# contoh
python main.py data/debby_ktp.jpg ktp           # default: pytesseract
python main.py data/debby_ktp.jpg ktp easyocr   # pakai EasyOCR
python main.py data/debby_ktp.jpg ktp llm       # pakai API LLM
python main.py data/sample_passport.jpg passport
```

Keluaran berupa JSON yang mencakup status, data hasil ekstraksi, indikator `valid`, serta timestamp.

## Mode Web Sederhana
```bash
python src/api.py
```
Buka browser ke `http://localhost:8000/` (atau port sesuai konfigurasi). Form web memungkinkan Anda memilih berkas gambar, tipe dokumen (`ktp`/`passport`), serta engine OCR (`pytesseract`, `easyocr`, `llm`). Hasil ekstraksi akan tampil sebagai JSON pada halaman yang sama.

### Contoh Keluaran KTP
```json
{
  "status": "success",
  "data": {
    "address": "JL KECAPL V",
    "birth_date": "24-12-1980",
    "birth_place": "JAKARTA",
    "city": "JAKARTA SELATAN",
    "gender": "Not found",
    "kecamatan": "JAGAKARSA",
    "kelurahan_desa": "DAGAKARSA",
    "marital_status": "BELUM KAWIN",
    "name": "AKU",
    "nik": "0074096112900001",
    "province": "DKI JAKARTA",
    "religion": "ZISLAM",
    "rt_rw": "2008 / 005"
  },
  "valid": false,
  "timestamp": "..."
}
```
> Beberapa nilai dapat tetap "Not found" bila bahasa/data latih Tesseract belum terpasang atau kualitas gambar kurang baik. Setelah `ind.traineddata` terpasang, akurasi label seperti jenis kelamin dan golongan darah meningkat.

## Menjalankan API Flask
```bash
python src/api.py
```

Endpoint yang tersedia:

| Method | Endpoint   | Deskripsi                                        |
|--------|------------|---------------------------------------------------|
| POST   | `/extract` | Upload berkas gambar & tipe dokumen (`ktp`/`passport`). |

Contoh permintaan menggunakan `curl`:
```bash
curl -X POST http://localhost:5000/extract \
  -F "file=@data/debby_ktp.jpg" \
  -F "type=ktp" \
  -F "provider=easyocr"
```

Respons API mengikuti format JSON yang sama dengan CLI.

**Port & host:** ubah lewat `config.yaml` (`server.host`, `server.port`, `server.debug`) atau override port/debug sementara via environment variable `OCR_API_PORT` dan `OCR_API_DEBUG` sebelum menjalankan `python src/api.py`.

## Mode OCR
- **pytesseract**: mode default yang memanfaatkan instalasi Tesseract lokal. Pastikan `tesseract.path` dan `tesseract.lang` sudah benar.
- **easyocr**: gunakan ketika ingin memanfaatkan model EasyOCR. Atur daftar bahasa dan opsi GPU di `config.yaml`.
- **llm**: memanggil endpoint LLM (contoh Google Gemini) untuk mengembalikan JSON terstruktur. Pastikan environment variable untuk API key sesuai (`GEMINI_API_KEY` secara default) dan endpoint/model sudah disetel.
  Respons akan menyertakan objek `usage` bila penyedia LLM mengembalikan informasi konsumsi token, dan pipeline otomatis mengirim satu permintaan dummy (gambar kosong → `{}`) sebelum permintaan utama agar sesi model siap.

## Konfigurasi
- **Tesseract**: perbarui `tesseract.path` bila executable tidak berada di `/usr/bin/tesseract`.
- **Bahasa OCR**: `tesseract.lang` menerima string bahasa dipisah `+` (contoh `eng+ind`). Modul akan otomatis menggunakan bahasa yang tersedia bila sebagian belum terpasang.
- **Pra-pemrosesan**: atur `preprocess.max_width`, `max_height`, `max_filesize_mb`, `jpeg_quality`, dan parameter CLAHE untuk mengendalikan resolusi/ukuran hasil preprocess.
- **Server Flask**: sesuaikan `server.host`, `server.port`, `server.debug` atau gunakan env var `OCR_API_PORT`/`OCR_API_DEBUG` saat menjalankan API.
- **EasyOCR**: atur daftar bahasa (`easyocr.lang`) dan penggunaan GPU (`easyocr.gpu`).
- **LLM**: isi `llm.endpoint`, `llm.model`, serta `llm.api_key_env` atau `llm.api_key`. Prompt default dapat ditimpa melalui `llm.prompts`, dan Anda dapat menyesuaikan balasan dummy via `llm.dummy_response` bila diperlukan.
- **Regex Template**: `templates.ktp.fields` berisi pola dasar. Parser juga memakai heuristik tambahan (`src/parser.py`) untuk menangani variasi label / kesalahan OCR.

## Tips Kualitas OCR
- Gunakan gambar beresolusi tinggi dengan pencahayaan merata.
- Hindari distorsi perspektif; lakukan crop agar kartu memenuhi frame.
- Jika hasil terlalu noisy, pertimbangkan filter tambahan (blur ringan, adaptive threshold) sebelum memberi ke pipeline.

## Troubleshooting
- **`TesseractNotFoundError`**: pastikan Tesseract sudah terinstal dan jalurnya benar di `config.yaml`.
- **Bahasa tidak tersedia**: jalankan `tesseract --list-langs` untuk mengecek bahasa yang terpasang, lalu instal paket yang hilang (misal `sudo apt-get install tesseract-ocr-ind`).
- **`ImportError: No module named 'src'`**: jalankan skrip melalui `python src/api.py` atau `python -m src.api` dari root proyek; modul path sudah ditangani otomatis di `src/api.py`.
- **Fontconfig / Matplotlib cache warning**: set `MPLCONFIGDIR` ke direktori yang writable atau abaikan; peringatan tidak mempengaruhi hasil OCR.

## Roadmap
- Tambah dukungan format KTP model terbaru (e-KTP dengan QR code).
- Normalisasi hasil (title case, format tanggal ISO).
- Tambah test suite + dataset contoh untuk regresi otomatis.
