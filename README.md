# Local GGUF OCR

API dan web sederhana untuk mengekstrak informasi dari gambar dokumen memakai model lokal GGUF. Project ini hanya memakai satu jalur inference: `llama-cpp-python` memuat model dari folder `./model` langsung di proses aplikasi.

Tidak ada Tesseract, EasyOCR, MRZ parser, atau provider eksternal. Pilihan yang tersedia hanya template output agar format JSON sesuai dokumen.

## Fitur
- Load model GGUF lokal saat aplikasi Flask start.
- Upload gambar melalui web atau endpoint API.
- Web app memakai AJAX, loading animation, dan timer proses tanpa reload halaman.
- Model membaca gambar dan mengembalikan satu JSON object sesuai template.
- Output dinormalisasi ke schema template: key yang hilang menjadi `null`, key ekstra dibuang.
- CLI sederhana untuk menjalankan ekstraksi dari terminal.
- Konfigurasi ringkas di `config/config.yaml`.

## Struktur
```text
.
├── config/
│   ├── config.yaml
│   └── config.example.yaml
├── model/                  # auto-created, ignored by git
├── postman/
│   └── information_extraction.postman_collection.json
├── src/
│   ├── api.py
│   ├── config.py
│   ├── local_model.py
│   └── templates/
│       └── web.html
├── main.py
├── requirements.txt
└── README.md
```

## Instalasi
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Folder `model/` tidak perlu dicommit. Jika file model belum ada, aplikasi akan otomatis download dari Hugging Face saat startup pertama.

Untuk Mac Apple Silicon, `llama-cpp-python` biasanya lebih cepat jika dibuild dengan Metal. Jika install biasa terasa lambat, reinstall dengan opsi Metal sesuai dokumentasi `llama-cpp-python`.

## Konfigurasi
File utama: `config/config.yaml`.

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: true
  use_reloader: false

local_model:
  enabled: true
  preload_on_start: true
  backend: "llama_cpp_python"
  model_path: "./model/Nanonets-OCR-s-Q4_0.gguf"
  mmproj_path: "./model/Nanonets-OCR-s-mmproj-F16.gguf"
  model_source:
    auto_download: true
    repo_id: "unsloth/Nanonets-OCR-s-GGUF"
    revision: "main"
    files:
      model: "Nanonets-OCR-s-Q4_0.gguf"
      mmproj: "mmproj-F16.gguf"
  model_name: "Nanonets-OCR-s"
  chat_handler: "qwen2.5-vl"
  ctx_size: 8192
  n_gpu_layers: -1
  n_threads: 4
  n_threads_batch: 4
  n_batch: 2048
  n_ubatch: 512
  flash_attn: true
  op_offload: true
  max_tokens: 512
  temperature: 0
  json_mode: true
  verbose: false
  use_mmap: true
  use_mlock: false
  warmup_on_start: true
  default_template: "ktp"
  prompt: "Return JSON matching schema exactly. Same keys only. Values only, no labels. Use null if unreadable."

templates:
  ktp:
    province: null
    city: null
    nik: null
    name: null
    birth_place: null
    birth_date: null
    gender: null
    blood_type: null
    address: null
    rt_rw: null
    kelurahan_desa: null
    kecamatan: null
    religion: null
    marital_status: null
    occupation: null
    nationality: null
    valid_until: null
  passport:
    passport_number: null
    name: null
    nationality: null
    date_of_birth: null
    gender: null
    expiration_date: null
    country_code: null
```

Prompt sengaja pendek dan hanya berisi instruction, nama template, serta schema JSON. Template juga hanya berisi elemen output agar tidak membingungkan user. Aplikasi tetap memaksa hasil akhir mengikuti field di template: key yang hilang menjadi `null`, key ekstra dibuang.

## Auto Download Model
Saat startup, app mengecek `local_model.model_path` dan `local_model.mmproj_path`. Jika salah satu belum ada dan `local_model.model_source.auto_download: true`, file akan didownload dari Hugging Face:

```yaml
local_model:
  model_source:
    repo_id: "unsloth/Nanonets-OCR-s-GGUF"
    files:
      model: "Nanonets-OCR-s-Q4_0.gguf"
      mmproj: "mmproj-F16.gguf"
```

File remote `mmproj-F16.gguf` disimpan ke path lokal `./model/Nanonets-OCR-s-mmproj-F16.gguf` agar kompatibel dengan config aplikasi.

## Menjalankan Web/API
```bash
python src/api.py
```

Model wajib dimuat saat `python src/api.py` dijalankan. Startup akan memuat GGUF, menjalankan warmup vision kecil, lalu Flask baru listen setelah muncul log `Local GGUF model ready.`.

`use_mmap: true` menjaga penggunaan RAM lebih rendah karena file model dipetakan oleh OS. `warmup_on_start: true` tetap memaksa inisialisasi inference dan mmproj/vision sebelum request pertama.

Setting performa dibuat mendekati log `llama-server`: `n_threads: 4`, `n_threads_batch: 4`, `n_batch: 2048`, `flash_attn: true`, dan `op_offload: true`. Default tetap memakai `qwen2.5-vl` supaya prompt yang dikirim tetap pendek sesuai config. `mtmd` bisa lebih mirip `llama-server`, tetapi pada model ini ia menambahkan prompt OCR bawaan yang panjang.

Untuk stop server, tekan `Ctrl+C`. App memasang shutdown handler yang menutup object `llama-cpp-python` lebih dulu agar proses Metal/ggml tidak crash saat interpreter exit.

Buka:
```text
http://localhost:8000/
```

Endpoint API:
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@/path/to/document.jpg" \
  -F "template=ktp"
```

Contoh respons:
```json
{
  "status": "success",
  "template": "ktp",
  "duration_seconds": 2.418,
  "data": {
    "name": "BUDI",
    "nik": "1234567890123456"
  },
  "timestamp": "2026-07-23T10:00:00+07:00"
}
```

## CLI
```bash
python main.py /path/to/document.jpg ktp
```

Argumen template opsional. Jika tidak diisi, CLI memakai `local_model.default_template`.

## Postman
Koleksi tersedia di:
```text
postman/information_extraction.postman_collection.json
```

Import koleksi tersebut, lalu isi form-data `file` dan `template` pada request `Extract Document`.

## Troubleshooting
- `llama-cpp-python belum terinstall`: jalankan `pip install -r requirements.txt` di virtualenv project.
- Model gagal load: pastikan dua file `.gguf` di folder `model/` ada dan cocok.
- Handler tidak ditemukan: upgrade `llama-cpp-python`, atau ubah `local_model.chat_handler` ke handler yang tersedia.
- Out of memory: turunkan `ctx_size`, ubah `n_gpu_layers`, atau gunakan model quantization yang lebih kecil.

## License
Project ini dirilis di bawah lisensi [MIT](LICENSE).
