"""
Contoh Inference — Muat Model dari MLflow Model Registry
===========================================================
Skrip contoh untuk aplikasi konsumen: memuat model ABSA versi terbaru
pada stage tertentu dari MLflow Model Registry (artifact bundle fisiknya
tersimpan di Cloudflare R2), lalu menjalankan prediksi pada teks baru.

Environment variable yang dibutuhkan (lihat .env):
  MLFLOW_TRACKING_URI     — URL MLflow tracking server
  AWS_ACCESS_KEY_ID       — samakan dengan R2_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY   — samakan dengan R2_SECRET_ACCESS_KEY
  MLFLOW_S3_ENDPOINT_URL  — samakan dengan R2_ENDPOINT_URL
  AWS_DEFAULT_REGION      — "auto" (wajib untuk R2)

Empat variabel terakhir wajib ada karena MLflow server di project ini
dijalankan dengan --no-serve-artifacts (lihat docker-compose.yaml) —
artinya client mengunduh artifact LANGSUNG dari R2 via boto3, bukan
lewat tracking server.

Skema versi/stage
------------------
Setiap kali tahap register_model mendaftarkan model baru dengan nama
registry yang sama (mis. "absa_indobert"), MLflow otomatis membuat versi
baru yang terus bertambah (v1, v2, v3, ...) — tidak pernah menimpa versi
lama. URI "models:/<nama>/<stage>" TIDAK terikat ke nomor versi tertentu;
ia selalu di-resolve ke versi TERTINGGI yang sedang berstatus stage
tersebut. Jadi aplikasi konsumen tidak perlu mengubah URL saat model baru
didaftarkan — cukup tetap memakai "models:/absa_indobert/Staging" (atau
"/Production" untuk model yang sudah dipromosikan manual ke produksi).
"""

import os

import mlflow.pyfunc


def load_absa_model(stage: str = "Staging"):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    registry_name = os.environ.get("MLFLOW_REGISTRY_NAME", "absa_indobert")
    model_uri = f"models:/{registry_name}/{stage}"

    print(f"Memuat model dari {model_uri} ...")
    return mlflow.pyfunc.load_model(model_uri)


if __name__ == "__main__":
    model = load_absa_model(stage="Staging")

    contoh_teks = [
        "Aplikasinya bagus tapi harga langganan mahal",
        "UI nya jelek dan sering error pas mau login",
    ]
    hasil = model.predict(contoh_teks)

    # predict() mengembalikan kontrak mentah {aspek: {label, score}} — reshaping
    # ke bentuk API (snake_case key, aspek dominan, rollup sentiment) adalah
    # tanggung jawab adapter di aplikasi konsumen, bukan model ini.
    for teks, pred in zip(contoh_teks, hasil):
        print(f"\nTeks: {teks}")
        for aspek, hasil_aspek in pred.items():
            print(f"  {aspek:<25}: {hasil_aspek['label']} ({hasil_aspek['score']})")
