"""
Tahap Ekstraksi Data
====================
Mengambil data produksi ter-anotasi dari komponen monitoring, memeriksa
kecukupan volumenya, dan membentuk dataset pelatihan terbaru.

Interface Contract dengan Komponen Monitoring
---------------------------------------------
Komponen monitoring diharapkan mengekspos endpoint berikut:

  POST {monitoring_api_url}/annotations/export
  Body   : {"since": "<ISO 8601 timestamp>"}
  Response (200):
    {
      "data": [
        {"Komentar": "...", "Content Quality": 1, "UI/UX": -1, ...},
        ...
      ],
      "count": <int>
    }

  Parameter input  : `since` — timestamp sejak data terakhir diambil
  Parameter output : kumpulan data teranotasi (skema kolom identik dengan
                     dataset pelatihan) sejak timestamp tersebut.
"""

import os
import datetime
import subprocess

import pandas as pd
import requests


# ── State file: pencatat timestamp ekstraksi terakhir ─────────────────────────
_DEFAULT_STATE_FILE = os.path.join(
    os.path.dirname(__file__), '..', '.state', 'last_extraction_timestamp'
)


# ── Interface Contract: Monitoring API ────────────────────────────────────────

def fetch_annotated_data_since(since_timestamp: str, api_url: str, timeout: int = 30) -> pd.DataFrame:
    """
    Ambil data teranotasi dari komponen monitoring sejak timestamp yang diberikan.

    Parameters
    ----------
    since_timestamp : str
        ISO 8601 timestamp (misal: "2025-01-01T00:00:00Z") sejak data
        terakhir diambil.
    api_url : str
        Base URL komponen monitoring (misal: "http://monitoring-service:8001").
    timeout : int
        Timeout request dalam detik.

    Returns
    -------
    pd.DataFrame
        Dataset teranotasi dengan kolom identik skema dataset pelatihan.
    """
    endpoint = f"{api_url.rstrip('/')}/annotations/export"
    response = requests.post(endpoint, json={"since": since_timestamp}, timeout=timeout)
    response.raise_for_status()
    records = response.json().get("data", [])
    return pd.DataFrame(records)


# ── State Management ──────────────────────────────────────────────────────────

def load_last_timestamp(state_file: str) -> str:
    """Muat timestamp ekstraksi terakhir dari file state."""
    state_file = os.path.normpath(state_file)
    if os.path.isfile(state_file):
        with open(state_file, 'r') as f:
            ts = f.read().strip()
            if ts:
                return ts
    # Default: satu tahun ke belakang jika belum pernah dijalankan
    return (datetime.datetime.utcnow() - datetime.timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')


def save_current_timestamp(state_file: str) -> str:
    """Catat timestamp saat ini sebagai timestamp ekstraksi terbaru."""
    state_file = os.path.normpath(state_file)
    ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w') as f:
        f.write(ts)
    return ts


# ── Data Sufficiency ──────────────────────────────────────────────────────────

def check_data_sufficiency(
    df_new: pd.DataFrame,
    existing_dataset_path: str,
    min_new_samples: int,
    min_ratio: float,
) -> dict:
    """
    Periksa apakah data baru cukup untuk membentuk dataset pelatihan terbaru.

    Kriteria kecukupan (OR — cukup memenuhi salah satu):
      - Jumlah data baru >= min_new_samples, ATAU
      - Rasio data baru terhadap dataset sebelumnya >= min_ratio

    Returns
    -------
    dict:
      sufficient  : bool
      n_new       : int
      n_existing  : int
      ratio       : float
      reason      : str
    """
    n_new = len(df_new)

    try:
        n_existing = len(pd.read_csv(existing_dataset_path))
    except FileNotFoundError:
        n_existing = 0

    ratio = n_new / n_existing if n_existing > 0 else float('inf')
    sufficient = (n_new >= min_new_samples) or (ratio >= min_ratio)

    if sufficient:
        reason = (
            f"Data baru ({n_new} sampel, rasio {ratio:.1%}) "
            f"memenuhi kriteria kecukupan."
        )
    else:
        reason = (
            f"Data baru ({n_new} sampel, rasio {ratio:.1%}) "
            f"belum memenuhi kriteria (min {min_new_samples} sampel "
            f"atau rasio {min_ratio:.1%})."
        )

    return {
        'sufficient' : sufficient,
        'n_new'      : n_new,
        'n_existing' : n_existing,
        'ratio'      : ratio,
        'reason'     : reason,
    }


# ── Dataset Merging & Versioning ──────────────────────────────────────────────

def merge_and_version_dataset(
    df_new: pd.DataFrame,
    existing_dataset_path: str,
) -> None:
    """
    Gabungkan data baru dengan dataset sebelumnya dan catat versi baru via DVC.
    Dataset hasil gabungan ditulis kembali ke path yang sama sehingga konfigurasi
    model tidak perlu diubah.
    """
    df_existing = pd.read_csv(existing_dataset_path)
    df_merged   = pd.concat([df_existing, df_new], ignore_index=True)
    df_merged.drop_duplicates(inplace=True)
    df_merged.to_csv(existing_dataset_path, index=False)

    print(f"  Dataset digabungkan: {len(df_existing)} + {len(df_new)} "
          f"→ {len(df_merged)} baris (setelah deduplikasi)")

    subprocess.run(['dvc', 'add', existing_dataset_path], check=True)
    subprocess.run(
        ['git', 'add', f'{existing_dataset_path}.dvc', '.gitignore'], check=True
    )
    subprocess.run(
        ['git', 'commit', '-m',
         f'data: tambah {len(df_new)} sampel produksi baru ke dataset pelatihan'],
        check=True,
    )
    subprocess.run(['dvc', 'push'], check=True)
    print(f"  Versi dataset terbaru tercatat via DVC: {existing_dataset_path}")


# ── Notification ──────────────────────────────────────────────────────────────

def notify_annotators(sufficiency_result: dict, webhook_url: str) -> None:
    """
    Kirim notifikasi ke annotator bahwa volume data belum mencukupi.
    Menggunakan generic incoming webhook (kompatibel Slack, Teams, dsb).
    """
    if not webhook_url:
        print(f"  [NOTIFIKASI — log only] {sufficiency_result['reason']}")
        print("  Webhook belum dikonfigurasi; set notification.annotator_webhook_url "
              "di workflow/pipeline_config.yaml untuk mengirim notifikasi nyata.")
        return

    message = {
        "text": (
            ":warning: *ABSA Retraining Pipeline — Data Produksi Belum Mencukupi*\n"
            f"{sufficiency_result['reason']}\n"
            "Mohon tambahkan anotasi sebelum siklus retraining berikutnya. "
            "Pipeline tetap dijalankan dengan dataset yang ada."
        )
    }
    try:
        resp = requests.post(webhook_url, json=message, timeout=10)
        resp.raise_for_status()
        print("  Notifikasi terkirim ke annotator.")
    except Exception as exc:
        print(f"  [PERINGATAN] Gagal mengirim notifikasi: {exc}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_extract_data(workflow_config: dict, model_config: dict) -> dict:
    """
    Jalankan tahap ekstraksi data secara lengkap.

    Alur:
      1. Muat timestamp ekstraksi terakhir
      2. Ambil data teranotasi dari monitoring API sejak timestamp tersebut
      3. Periksa kecukupan data
      4. Jika cukup  → gabungkan dengan dataset sebelumnya & versi via DVC
      5. Jika belum  → kirim notifikasi (pipeline tetap dilanjutkan)
      6. Catat timestamp ekstraksi saat ini

    Returns
    -------
    dict:
      dataset_path    : str  — path dataset yang akan digunakan retraining
      n_new_samples   : int  — jumlah sampel baru dari monitoring
      data_sufficient : bool — apakah volume data memenuhi kriteria
      extraction_ts   : str  — timestamp ekstraksi ini (ISO 8601)
    """
    ext_cfg   = workflow_config['data_extraction']
    suf_cfg   = workflow_config['data_sufficiency']
    notif_cfg = workflow_config['notification']

    existing_path = model_config['data']['path']

    if not ext_cfg.get('enabled', False):
        retraining_input = model_config.get('retraining_input') or {}
        n_new = int(retraining_input.get('new_sample_count') or 0)
        n_existing = int(retraining_input.get('pool_rows_before') or 0)
        combined_rows = int(retraining_input.get('combined_rows_used') or 0)
        ratio = n_new / n_existing if n_existing > 0 else (float('inf') if n_new else 0.0)
        sufficient = (
            n_new >= int(suf_cfg['min_new_samples'])
            or ratio >= float(suf_cfg['min_ratio_of_existing'])
        )

        print("  [DILEWATI] data_extraction.enabled=false — menggunakan snapshot "
              "dataset yang sudah dibentuk dan dibekukan oleh BE_ABSA.")
        print(
            f"  Metadata snapshot → data baru: {n_new} | pool sebelumnya: "
            f"{n_existing} | gabungan: {combined_rows or 'tidak diketahui'} | "
            f"versi: {retraining_input.get('dataset_version') or 'tidak diketahui'}"
        )
        print(
            f"  Kecukupan data: {'CUKUP' if sufficient else 'BELUM CUKUP'} — "
            f"{n_new} sampel baru ({ratio:.1%} dari pool sebelumnya); "
            f"ambang {suf_cfg['min_new_samples']} sampel atau "
            f"{float(suf_cfg['min_ratio_of_existing']):.1%}."
        )
        return {
            'dataset_path'   : existing_path,
            'n_new_samples'  : n_new,
            'data_sufficient': sufficient,
            'extraction_ts'  : None,
        }

    state_file = ext_cfg.get('state_file', _DEFAULT_STATE_FILE)
    since_ts   = load_last_timestamp(state_file)
    print(f"  Mengambil data produksi sejak: {since_ts}")

    df_new = fetch_annotated_data_since(
        since_timestamp = since_ts,
        api_url         = ext_cfg['monitoring_api_url'],
        timeout         = ext_cfg.get('monitoring_api_timeout_seconds', 30),
    )
    print(f"  Data baru diterima: {len(df_new)} sampel")

    sufficiency = check_data_sufficiency(
        df_new               = df_new,
        existing_dataset_path = existing_path,
        min_new_samples      = suf_cfg['min_new_samples'],
        min_ratio            = suf_cfg['min_ratio_of_existing'],
    )
    status_label = 'CUKUP' if sufficiency['sufficient'] else 'BELUM CUKUP'
    print(f"  Kecukupan data: {status_label} — {sufficiency['reason']}")

    if sufficiency['sufficient']:
        merge_and_version_dataset(df_new, existing_path)
    else:
        if notif_cfg.get('enable_notifications'):
            notify_annotators(sufficiency, notif_cfg.get('annotator_webhook_url', ''))
        else:
            print(f"  [NOTIFIKASI — log only] {sufficiency['reason']}")
        print("  Pipeline tetap dilanjutkan dengan dataset yang ada.")

    extraction_ts = save_current_timestamp(state_file)

    return {
        'dataset_path'   : existing_path,
        'n_new_samples'  : sufficiency['n_new'],
        'data_sufficient': sufficiency['sufficient'],
        'extraction_ts'  : extraction_ts,
    }
