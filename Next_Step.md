# Status: Integrasi Metaflow (lokal) + GPU Training via Modal

Konteks: pipeline ABSA (`run_experiment.py` / `workflow/flow.py`) butuh GPU untuk
tahap training. Laptop tidak punya GPU, sebelumnya training dilakukan manual
di Google Colab (klik "Run" di notebook) — akibatnya Metaflow tidak bisa
mencatat metadata run karena seluruh flow (termasuk step ringan) ikut
dieksekusi di VM Colab yang ephemeral, bukan di laptop.

Solusi yang sedang dibangun: **Metaflow tetap dijalankan lokal** (supaya
`.metaflow/` tersimpan di repo), tapi step training didelegasikan secara
programatik ke GPU cloud **Modal** (bukan Colab) saat tidak ada GPU lokal —
tanpa klik manual, cocok untuk dipicu otomatis di production nantinya.

## Sudah dilakukan

### 1. Environment lokal
- venv baru di `model_ABSA/.venv`, Python **3.12.13** via pyenv (bukan 3.11 —
  `numpy==2.5.0` di `requirements.txt` butuh Python ≥3.12, sama seperti base
  image `mlflow/Dockerfile` yang pakai `python:3.12-slim`).
- Modal CLI terinstal & terautentikasi (akun `rezafh19`, free credit $1).
- Dependency training (`torch`, `transformers`, dll dari `requirements.txt`,
  minus `dvc`/`dvc-s3`) ikut diinstal di venv lokal juga — **wajib**, karena
  `evaluate_step`/`validate_model.py` sudah lebih dulu `import torch`
  langsung untuk reload checkpoint + inferensi CPU (bukan kebutuhan baru dari
  Modal). Di Mac, `pip install torch` otomatis dapat build CPU/MPS (ringan,
  bukan build CUDA raksasa seperti di Linux).

### 2. Infra MLflow lokal
- Port MLflow dipindah dari **5000 → 5001** (bentrok dengan AirPlay Receiver
  macOS) — diubah di `docker-compose.yaml`,
  `configs/experiment_indobert_baseline.yaml`, `workflow/pipeline_config.yaml`.
- Cloudflare Quick Tunnel dibuat untuk expose MLflow lokal ke internet (supaya
  Modal bisa log ke sana): **`https://semi-factor-repairs-yea.trycloudflare.com`**
  — ⚠️ ephemeral, mati kalau proses `cloudflared` atau laptop restart, perlu
  dibuat ulang lalu update `.env` + Modal Secret dengan URL baru.
- Nama bucket B2 diperbaiki: `mlflow-artifact` (bukan `mlflow-artifacts`,
  singular) — sudah diperbaiki di `.env` dan `experiments.artifact_location`
  di Postgres (dua eksperimen yang kadung salah sudah di-`UPDATE`).
- `.env` (gitignored, tidak pernah di-commit) sekarang berisi kredensial B2
  dua kali dengan nama berbeda: `B2_*` (dipakai `docker-compose.yaml`) dan
  alias `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`MLFLOW_S3_ENDPOINT_URL`/
  `MLFLOW_TRACKING_URI` (dipakai proses Python yang jalan langsung di laptop,
  mis. boto3/mlflow saat download artifact).

### 3. Kode
- **`modal_app.py`** (baru): `modal.App("absa-training")`, image Python 3.12
  + dependency training, fungsi GPU `train_remote(model_config, data)` yang
  memanggil `run_train_model()` lalu upload bundle checkpoint
  (`best_model.pt` + tokenizer + config) sebagai MLflow artifact di path
  `checkpoint/`. Sudah **di-deploy** ke Modal (`modal deploy modal_app.py`).
- **`workflow/stages/train_model.py`**: `run_train_model()` sekarang jadi
  dispatcher — cek `torch.cuda.is_available()`:
  - Ada GPU lokal → `_train_local()` (kode lama, tidak berubah).
  - Tidak ada GPU → `_train_remote()`: panggil `modal.Function.from_name(
    "absa-training", "train_remote").remote(...)`, lalu unduh checkpoint
    balik ke `save_dir` lokal via `mlflow.artifacts.download_artifacts()`
    supaya `evaluate_step`/`register_model` tetap bisa pakai file lokal
    seperti biasa.

### 4. Pengujian
- Smoke test langsung (`modal run modal_app.py`, versi awal train+evaluate
  digabung) **berhasil penuh**: training di GPU T4, early stopping epoch 9,
  Test Sentiment F1 0.5424, checkpoint ~498MB ke B2 — semua sukses.
- Setelah dipecah (train-only di Modal + wiring ke `train_model.py`),
  training di Modal **masih berhasil**, tapi tahap unduh checkpoint balik ke
  lokal **gagal**: `403 Forbidden` saat `HeadObject` — Application Key B2
  yang dipakai punya capability list+write tapi **tidak punya `readFiles`**,
  jadi bisa upload tapi tidak bisa download balik.

### 5. Git
- Semua perubahan di atas sudah di-commit ke branch baru
  **`development/dummy_experiments`** (sudah di-push ke GitHub). Branch
  `main` tidak tersentuh. Commit pakai identitas git lokal user sendiri,
  tanpa atribusi AI apa pun.

## Belum dilakukan / langkah selanjutnya

1. **[BLOCKER]** Perbaiki capability Application Key B2 — buka dashboard
   Backblaze B2 → App Keys → tambahkan `readFiles` pada key yang ada, atau
   buat key baru scoped ke bucket `mlflow-artifact` dengan akses **Read and
   Write**. Setelah dapat key yang benar, update `.env` + jalankan ulang
   `modal secret create absa-mlflow-creds ... --force`.
2. Re-test `run_train_model()` end-to-end (dispatcher lokal → Modal → unduh
   checkpoint) sampai `best_model.pt` benar-benar ada di `save_dir` lokal.
3. Jalankan **seluruh flow Metaflow** secara lokal:
   `python workflow/flow.py run` — pastikan `.metaflow/` ter-generate di
   repo, dan `evaluate_step` → `validate_model` → `register_model` jalan
   normal memakai checkpoint hasil Modal.
4. Hardening untuk production:
   - Ganti Cloudflare Quick Tunnel dengan *named tunnel* / domain tetap
     (quick tunnel tidak cocok untuk production, URL berubah tiap restart).
   - Rotasi kredensial B2 (sempat sempat tertulis plaintext di
     `Untitled0.ipynb` — rotasi ditunda atas permintaan user untuk keperluan
     testing, sebaiknya dilakukan sebelum production).
   - Tentukan mekanisme trigger terjadwal di production (cron/systemd/GitHub
     Actions) — `@schedule` bawaan Metaflow butuh Metaflow Service yang
     belum di-deploy, jadi tidak aktif dengan setup lokal saat ini.
   - Pantau pemakaian free credit Modal ($1) seiring bertambahnya run
     training.
5. (Opsional) Notebook Colab (`Untitled0.ipynb`) sudah tidak relevan lagi
   untuk alur training — bisa diarsipkan/dihapus setelah alur Modal
   terkonfirmasi stabil.
