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

1. Re-test `run_train_model()` end-to-end (dispatcher lokal → Modal → unduh
   checkpoint) sampai `best_model.pt` benar-benar ada di `save_dir` lokal.
2. Jalankan **seluruh flow Metaflow** secara lokal:
   `python workflow/flow.py run` — pastikan `.metaflow/` ter-generate di
   repo, dan `evaluate_step` → `validate_model` → `register_model` jalan
   normal memakai checkpoint hasil Modal.
3. Hardening untuk production:
   - Ganti Cloudflare Quick Tunnel dengan *named tunnel* / domain tetap
     (quick tunnel tidak cocok untuk production, URL berubah tiap restart).
   - Tentukan mekanisme trigger terjadwal di production (cron/systemd/GitHub
     Actions) — `@schedule` bawaan Metaflow butuh Metaflow Service yang
     belum di-deploy, jadi tidak aktif dengan setup lokal saat ini.
   - Pantau pemakaian free credit Modal ($1) seiring bertambahnya run
     training.
4. (Opsional) Notebook Colab (`Untitled0.ipynb`) sudah tidak relevan lagi
   untuk alur training — bisa diarsipkan/dihapus setelah alur Modal
   terkonfirmasi stabil.

## Update — Migrasi artifact store dari Backblaze B2 ke Cloudflare R2

Alasan: masalah capability Application Key B2 (list+write tapi tanpa
`readFiles`, lihat riwayat blocker di atas) jadi pemicu untuk pindah
sepenuhnya ke Cloudflare R2, sekalian menyederhanakan (tunneling MLflow
dan akses R2 sama-sama lewat ekosistem Cloudflare).

Karena integrasi B2 sebelumnya **tidak pernah pakai `b2sdk` native** —
semua lewat MLflow `S3ArtifactRepository` (boto3) yang murni dikonfigurasi
dari environment variable — migrasi ke R2 hanya perlu ganti konfigurasi,
bukan kode:
- `docker-compose.yaml`: env var `B2_*` diganti `R2_*`, ditambah
  `AWS_DEFAULT_REGION=auto` (wajib untuk R2 API).
- `.env`: `B2_ENDPOINT_URL`/`B2_BUCKET_NAME`/`B2_KEY_ID`/`B2_APPLICATION_KEY`
  diganti `R2_ENDPOINT_URL`/`R2_BUCKET_NAME`/`R2_ACCESS_KEY_ID`/
  `R2_SECRET_ACCESS_KEY` (nilai R2 asli masih perlu diisi manual — lihat
  checklist di bawah).
- `modal_app.py`, `inference_example.py`: hanya docstring yang disebut B2,
  diubah ke R2 (tidak ada perubahan kode).
- `requirements.txt`/`mlflow/Dockerfile`: tidak berubah, `boto3` yang sudah
  ada dipakai ulang apa adanya.

**Checklist yang perlu disiapkan pengguna sebelum training/inference jalan lagi:**
1. Di dashboard Cloudflare → R2 → buat bucket baru (mis. `mlflow-artifact`,
   nama bebas asal konsisten dengan `.env`).
2. Buat R2 API Token (R2 → Manage API Tokens → Create API Token) dengan
   permission **Object Read & Write**, scoped ke bucket di atas — jangan
   ulangi kesalahan B2 yang lupa kasih izin read.
3. Catat `Account ID` (terlihat di dashboard R2, dipakai untuk endpoint
   `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`), `Access Key ID`, dan
   `Secret Access Key` dari token yang baru dibuat.
4. Isi ke `.env` lokal: `R2_ENDPOINT_URL`, `R2_BUCKET_NAME`,
   `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`.
5. Untuk proses Python yang jalan langsung (bukan lewat docker-compose,
   mis. `workflow/stages/train_model.py` saat mengunduh checkpoint hasil
   Modal secara lokal): set juga `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `MLFLOW_S3_ENDPOINT_URL`, `AWS_DEFAULT_REGION=auto` di environment shell
   (nilainya sama dengan `R2_*` di atas) — kode membaca `os.environ`
   langsung, tidak ada loader `.env` otomatis di repo ini.
6. Update Modal Secret dengan kredensial baru:
   `modal secret create absa-mlflow-creds MLFLOW_TRACKING_URI=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... MLFLOW_S3_ENDPOINT_URL=... AWS_DEFAULT_REGION=auto --force`
7. `docker compose up -d --build` ulang container `mlflow` supaya
   `--default-artifact-root` memakai bucket R2 yang baru untuk experiment
   berikutnya (experiment lama di Postgres masih menunjuk artifact_location
   `s3://mlflow-artifact` lama di B2 — kalau bucket lama masih ada datanya
   tidak hilang, tapi run baru sebaiknya jadi experiment baru supaya tidak
   tercampur dua storage).
8. Setelah R2 terkonfirmasi jalan: cabut/nonaktifkan Application Key B2 lama
   di dashboard Backblaze (kredensial lama sudah tidak dipakai kode manapun
   lagi di repo ini).
9. `.dvc/config` (remote dataset DVC) masih mengarah ke MinIO lokal, sama
   sekali terpisah dari B2/R2 — di luar cakupan migrasi ini kecuali memang
   ingin disatukan ke R2 juga.

## Update — Metaflow UI (self-hosted, via docker-compose)

Tujuan: dashboard visual (DAG, timeline tiap step, riwayat run, artifact
browser) — bukan lagi cuma `python workflow/flow.py list-runs`/`show` di
terminal.

**Riset penting sebelum implementasi:** di Docker Hub, HANYA
`netflixoss/metaflow_metadata_service` yang punya image publik siap pakai.
UI backend (`ui_backend_service`) dan frontend (`metaflow-ui`) **tidak
punya image publik** — keduanya harus di-build dari source clone repo
Netflix. Sudah diputuskan: clone manual ke folder `.metaflow-ui-stack/`
yang di-gitignore (bukan git submodule), konsisten dengan pola `.venv`
yang juga tidak pernah masuk git.

**Konsekuensi arsitektur:** supaya UI backend bisa membaca artifact/DAG,
datastore Metaflow untuk run yang mau ditampilkan di UI ini **harus S3**
(reuse bucket R2 yang sama dengan MLflow, prefix `metaflow/` supaya tidak
tercampur) — bukan lagi `.metaflow/` lokal. Ini **opt-in lewat environment
variable**: kalau variabel `METAFLOW_*` di bawah tidak di-set, `python
workflow/flow.py run` tetap jalan seperti biasa ke `.metaflow/` lokal
(mode lama tidak rusak). Konsekuensinya: run lama yang sudah ada di
`.metaflow/` lokal TIDAK otomatis muncul di UI ini (provider metadata dan
datastore-nya berbeda) — hanya run baru yang dijalankan dengan env var
service yang tercatat di sini.

### Langkah setup

1. Clone kedua repo Netflix, **pin ke tag yang sama dengan image metadata
   service** (`v2.5.1`) supaya skema DB antara metadata service (image
   publik) dan ui-backend (dibangun dari source) tidak mismatch. **Wajib**
   `-c core.autocrlf=false` di Windows — kalau tidak, `Dockerfile.ui_service`
   gagal build karena shell script bawaan repo (`download_ui.sh`) ikut
   dikonversi ke CRLF oleh git dan shebang-nya jadi tidak dikenali Linux:
   ```
   git clone -c core.autocrlf=false --branch v2.5.1 https://github.com/Netflix/metaflow-service .metaflow-ui-stack/metaflow-service
   git clone -c core.autocrlf=false https://github.com/Netflix/metaflow-ui .metaflow-ui-stack/metaflow-ui
   ```
2. Buat database `metaflow` di Postgres yang sudah ada (volume sudah
   terisi dari setup MLflow sebelumnya, jadi init script otomatis di
   `postgres-init/` tidak akan jalan ulang — harus manual sekali):
   ```
   docker compose exec postgres psql -U mlflow -d mlflow -c "CREATE DATABASE metaflow OWNER mlflow;"
   ```
3. Pastikan `.env` sudah berisi `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`,
   `R2_ENDPOINT_URL`, `R2_BUCKET_NAME` (sudah ada dari migrasi R2 di atas —
   dipakai ulang oleh `metaflow-ui-backend` untuk baca datastore).
4. Build & jalankan tiga service baru:
   ```
   docker compose up -d --build metaflow-metadata metaflow-ui-backend metaflow-ui
   ```
5. **Migrasi skema DB berjalan OTOMATIS** saat container `metaflow-metadata`
   start (terlihat di `docker logs metaflow-metadata`: `Running initial
   migration.. ... goose: no migrations to run`) — tidak perlu `curl
   /upgrade` manual. Verifikasi saja statusnya:
   ```
   curl http://localhost:8082/db_schema_status   # is_up_to_date: true
   ```
6. **Metaflow tidak jalan native di Windows** (`import fcntl` di
   `metaflow/sidecar/sidecar_subprocess.py` — modul POSIX-only, tidak ada di
   Windows). `python workflow/flow.py run` **harus dijalankan dari WSL**,
   bukan PowerShell/CMD langsung — konsisten dengan `.venv` yang memang
   sudah dibuat dari WSL (`home = /usr/bin` di `.venv/pyvenv.cfg`). Docker
   Desktop dengan backend WSL2 sudah meneruskan `localhost` antara Windows
   dan WSL, jadi service di docker-compose tetap bisa diakses dari WSL via
   `http://localhost:8080` dst tanpa konfigurasi tambahan.
7. Jalankan pipeline dari WSL dengan env var yang mengarahkan ke Metaflow
   Service + datastore R2:
   ```bash
   set -a; source .env; set +a   # atau export manual satu-satu
   export METAFLOW_SERVICE_URL=http://localhost:8080
   export METAFLOW_DEFAULT_METADATA=service
   export METAFLOW_DEFAULT_DATASTORE=s3
   export METAFLOW_DATASTORE_SYSROOT_S3="s3://${R2_BUCKET_NAME}/metaflow"
   export METAFLOW_S3_ENDPOINT_URL="${R2_ENDPOINT_URL}"
   export AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
   export AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
   export AWS_DEFAULT_REGION=auto
   .venv/bin/python3.12 workflow/flow.py run
   ```
   Tidak ada perubahan kode di `workflow/flow.py`/`trigger.py` — variabel
   `METAFLOW_*` dibaca otomatis oleh library Metaflow sendiri sebelum
   FlowSpec apa pun dieksekusi.
8. Buka dashboard: http://localhost:3000 (atau langsung http://localhost:8083,
   `ui_backend_service` ternyata membundel build rilis frontend-nya sendiri
   via `download_ui.sh` — jadi ada dua UI yang jalan, keduanya valid).

### Sudah diverifikasi end-to-end (smoke test, 2026-07-06)

Flow uji 2-step (`start` → `end`) berhasil tercatat penuh: metadata di
Postgres (`GET /flows/SmokeTestFlow/runs` menunjukkan status `completed`)
dan artifact di R2 (`s3://<bucket>/metaflow/`). Dua isu ditemukan & sudah
diperbaiki selama proses:

- **Line ending CRLF pada `download_ui.sh`** (`Dockerfile.ui_service`)
  bikin build `metaflow-ui-backend` gagal (`exit code 127`, shebang
  `#!/bin/bash\r` tidak dikenali Linux) — penyebabnya `core.autocrlf=true`
  di git config Windows. **Fix:** clone kedua repo Netflix dengan
  `git clone -c core.autocrlf=false ...` (bukan clone biasa).
- **Direktori kerja saat `run` menentukan performa packaging Metaflow.**
  Metaflow mem-package seluruh working directory (minus direktori
  berawalan titik, yang otomatis dikecualikan) sebagai code bundle sebelum
  eksekusi. Karena clone Netflix awalnya di folder `metaflow-ui-stack/`
  (tanpa titik) di root repo, `python workflow/flow.py run` macet
  **belasan menit** (bukan gagal, cuma sangat lambat — Metaflow menelusuri
  seluruh isi clone tersebut). **Fix:** folder diganti nama jadi
  `.metaflow-ui-stack/` (berawalan titik → otomatis dikecualikan) — run
  yang sama selesai dalam ~60 detik setelahnya. `.venv/` sudah aman by
  default karena juga berawalan titik. **Pelajaran:** direktori non-hidden
  apa pun yang ditaruh di root repo (terutama yang isinya banyak file,
  seperti clone repo lain) berisiko bikin packaging Metaflow lambat —
  hindari, atau taruh di luar root repo / beri awalan titik.

### Belum dilakukan / perlu diverifikasi

- Baru dites dengan flow 2-step tanpa dependency berat. Belum dites dengan
  `workflow/flow.py` yang sesungguhnya (torch, transformers, dll ikut
  ter-package? — kemungkinan besar tidak masalah karena hanya modul lokal
  yang diimpor `pipeline.*`/`model.*`/`preprocessing.*` yang ikut ter-package,
  bukan seluruh `site-packages`, tapi tetap perlu dikonfirmasi).
- Kredensial R2 yang sama dipakai dua sistem berbeda (MLflow artifact +
  Metaflow datastore) di bucket yang sama, prefix beda — pastikan token R2
  scoped ke bucket (bukan cuma satu prefix) supaya kedua prefix bisa diakses.
- Belum ada mekanisme supaya env var `METAFLOW_*` di atas otomatis ter-set
  tiap kali mau pakai mode service (saat ini harus di-export manual tiap
  sesi shell baru) — kalau nanti terasa mengganggu, pertimbangkan wrapper
  script kecil (bukan `.env` loader otomatis, sesuai konvensi repo ini).
