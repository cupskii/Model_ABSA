"""
Tahap Perbandingan dengan Model Champion
==========================================
Menggantikan gerbang validasi otomatis (ambang batas, vs stage Production,
vs stage Staging) yang dulu ada di workflow/stages/validate_model.py.
Sekarang murni informasional: muat model yang sedang memegang alias MLflow
`champion` (kalau ada), evaluasi ulang pada test split RUN INI (bukan
metrik lama yang mungkin dihitung dari test split dataset yang berbeda),
lalu laporkan dua metrik utamanya berdampingan dengan model baru — supaya
developer bisa membandingkan sendiri di UI sebelum memutuskan menekan
tombol "Promote". Tidak ada pass/fail di sini sama sekali.
"""

import os
import sys
import glob

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

import torch
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from model.checkpoint_io import load_model_from_checkpoint
from pipeline.evaluate_model import compute_test_metrics


def _download_and_reconstruct(model_uri: str, device: torch.device) -> tuple:
    """Unduh checkpoint dari MLflow (by alias URI) dan rekonstruksi model+tokenizer.

    Sama seperti _reevaluate_checkpoint yang dulu ada di validate_model.py —
    lihat catatan di sana soal kenapa dicari via glob (bukan basename tetap).
    """
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
    matches = glob.glob(os.path.join(local_dir, 'artifacts', '*', 'best_model.pt'))
    if not matches:
        raise FileNotFoundError(
            f"best_model.pt tidak ditemukan di artifact '{model_uri}' (local_dir={local_dir})"
        )
    checkpoint_dir = os.path.dirname(matches[0])
    return load_model_from_checkpoint(checkpoint_dir, device)


def run_compare_champion(model_config: dict, workflow_config: dict, data: dict, run_id: str | None = None) -> dict:
    """
    Bandingkan model champion saat ini (alias MLflow) dengan model baru,
    pada test split yang sama (`data`).

    Sebelumnya cuma dua metrik agregat (Sentiment/Detection F1) yang
    dikembalikan — metrik per-aspek dari compute_test_metrics() dibuang.
    Developer bisa menekan "Promote" atas model yang agregatnya naik padahal
    salah satu dari 5 aspek justru regresi. Sekarang seluruh dict flat
    (agregat + per-aspek) dikembalikan apa adanya, dan kalau `run_id`
    diberikan juga dicatat ke MLflow run yang sama dengan prefiks
    "champion_" supaya trigger.py bisa mengambilnya lewat MLflow API
    (bukan regex atas teks log) tanpa mengubah run historis milik champion
    itu sendiri — ini reevaluasi pada test split RUN INI, bukan skor asli
    champion saat pertama kali dilatih.

    Returns
    -------
    dict:
      champion_exists : bool
      champion_version: str | None
      metrics         : dict | None — flat, sama format dengan compute_test_metrics()
                         (test_mean_sentiment_f1, test_mean_detect_f1, dan
                         test_{aspek}_{detect_f1,sentiment_f1,pair_f1,aspect_detect_f1})
      reason          : str — penjelasan (dipakai kalau champion_exists=False)
    """
    mlflow_wf  = workflow_config.get('mlflow', {})
    mlflow_mdl = model_config.get('mlflow', {})

    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or (
        mlflow_wf.get('tracking_uri') or mlflow_mdl.get('tracking_uri', 'http://localhost:5000')
    )
    mlflow.set_tracking_uri(tracking_uri)

    registry_name = mlflow_wf.get('registry_name', mlflow_mdl.get('registry_name', 'absa_indobert'))
    alias         = mlflow_wf.get('champion_alias', 'champion')

    client = MlflowClient()
    try:
        champion_version_info = client.get_model_version_by_alias(registry_name, alias)
    except MlflowException as exc:
        return {
            'champion_exists': False,
            'champion_version': None,
            'metrics': None,
            'reason': f"Belum ada model dengan alias '{alias}' di registry '{registry_name}' ({exc.error_code}).",
        }
    except Exception as exc:
        return {
            'champion_exists': False,
            'champion_version': None,
            'metrics': None,
            'reason': f"Gagal mengakses MLflow Model Registry: {exc}",
        }

    print(f"  Model champion ditemukan: {registry_name} v{champion_version_info.version} — mengunduh & mengevaluasi ulang...")
    model_uri = f"models:/{registry_name}@{alias}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        champion_model, champion_tokenizer, champion_cfg = _download_and_reconstruct(model_uri, device)
    except Exception as exc:
        return {
            'champion_exists': True,
            'champion_version': champion_version_info.version,
            'metrics': None,
            'reason': f"Model champion v{champion_version_info.version} ada, tetapi gagal diunduh/dimuat: {exc}",
        }

    champion_metrics = compute_test_metrics(champion_model, champion_tokenizer, champion_cfg, data)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})

    return {
        'champion_exists': True,
        'champion_version': champion_version_info.version,
        'metrics': champion_metrics,
        'reason': f"Model champion v{champion_version_info.version} berhasil dievaluasi ulang.",
    }
