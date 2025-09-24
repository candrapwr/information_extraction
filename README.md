# KTP & Passport OCR

Pipeline untuk mengekstrak informasi dari KTP Indonesia dan paspor internasional menggunakan Python, OpenCV, dan Tesseract. Proyek ini menyediakan skrip CLI serta layanan Flask sederhana untuk integrasi ke aplikasi lain.

## Fitur Utama
- Ekstraksi seluruh elemen penting KTP (NIK, alamat lengkap, RT/RW, kelurahan/desa, kecamatan, agama, status kawin, pekerjaan, kewarganegaraan, masa berlaku, dsb.).
- Pembacaan MRZ paspor menggunakan *passporteye* serta OCR tambahan untuk informasi non-MRZ.
- Pra-pemrosesan citra (grayscale + Otsu thresholding) agar hasil OCR lebih stabil.
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

Edit `config/config.yaml` agar `tesseract.path` menunjuk ke executable Tesseract di sistem Anda dan sesuaikan daftar bahasa (`lang`).

## Struktur Proyek
```
.
├── config/
│   └── config.yaml          # Jalur Tesseract + pola regex
├── src/
│   ├── api.py               # Flask API
│   ├── preprocess.py        # Fungsi pra-pemrosesan citra
│   ├── ocr.py               # Wrapper pytesseract + MRZ
│   └── parser.py            # Parser KTP & paspor
├── main.py                  # Entry-point CLI
├── requirements.txt
└── README.md
```

## Penggunaan CLI
```bash
python main.py <path_gambar> [ktp|passport]

# contoh
python main.py data/debby_ktp.jpg ktp
python main.py data/sample_passport.jpg passport
```

Keluaran berupa JSON yang mencakup status, data hasil ekstraksi, serta timestamp.

### Contoh Keluaran KTP
```json
{
  "status": "success",
  "data": {
    "province": "DKI JAKARTA",
    "city": "JAKARTA SELATAN",
    "nik": "0074096112900001",
    "name": "AKU",
    "birth_place": "JAKARTA",
    "birth_date": "24-12-1980",
    "gender": "Not found",
    "blood_type": "Not found",
    "address": "JL KECAPL V",
    "rt_rw": "2008 / 005",
    "kelurahan_desa": "DAGAKARSA",
    "kecamatan": "JAGAKARSA",
    "religion": "ZISLAM",
    "marital_status": "BELUM KAWIN",
    "occupation": "KARYAWAN SWASTA SAKARTA SELATAN",
    "nationality": "WNI",
    "valid_until": "21-12-2016"
  }
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
  -F "type=ktp"
```

Respons API mengikuti format JSON yang sama dengan CLI.

## Konfigurasi
- **Tesseract**: perbarui `tesseract.path` bila executable tidak berada di `/usr/bin/tesseract`.
- **Bahasa OCR**: `tesseract.lang` menerima string bahasa dipisah `+` (contoh `eng+ind`). Modul akan otomatis menggunakan bahasa yang tersedia bila sebagian belum terpasang.
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

## Lisensi
Proyek ini tidak menyertakan lisensi eksplisit. Tambahkan lisensi pilihan Anda sebelum dipublikasikan.

