"""
Tahap Pelatihan Model (Workflow Wrapper)
=========================================
Membungkus pipeline/train_model.py menjadi satu unit eksekusi yang dapat
diisolasi dan diulang secara independen dalam automated ML workflow pipeline.

Tanggung jawab tambahan dibanding pipeline/train_model.py:
  - Membuka sesi MLflow run untuk mencatat seluruh metadata eksperimen
  - Mengembalikan dict ringan (tanpa objek model) agar dapat disimpan sebagai
    Metaflow artifact lintas step — model disimpan ke disk oleh pipeline/train_model.py
    dan dimuat ulang oleh tahap evaluate_model.

GPU lokal vs Modal
------------------
run_train_model() mendeteksi torch.cuda.is_available() dan memilih otomatis:
  - Ada GPU lokal   → latih langsung (_train_local), seperti sebelumnya.
  - Tidak ada GPU   → delegasikan ke GPU cloud Modal (_train_remote), lihat
    modal_app.py. Checkpoint yang dihasilkan Modal diunduh kembali ke
    save_dir lokal sehingga evaluate_step/validate_model/register_model
    tetap bekerja tanpa perubahan.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.pytorch
import torch

from model.absa_model import set_seed
from pipeline.train_model import train_model
from run_experiment import flatten_config, get_git_commit


def run_train_model(model_config: dict, data: dict) -> dict:
    """
    Latih model ABSA — otomatis memilih lokal (kalau ada GPU) atau delegasi
    ke GPU cloud Modal (kalau tidak ada GPU lokal, mis. laptop tanpa GPU).

    Parameters
    ----------
    model_config : dict — konfigurasi model dari YAML eksperimen
    data         : dict — output run_prepare_data() (df_train, df_val, df_test, class_weights)

    Returns
    -------
    dict:
      run_id          : str   — MLflow run ID (untuk dilanjutkan di evaluate_model)
      save_dir        : str   — direktori checkpoint model
      best_val_f1     : float — Sentiment F1 terbaik pada validation set
      best_val_det_f1 : float — Detection F1 pada epoch terbaik
    """
    if torch.cuda.is_available():
        return _train_local(model_config, data)
    return _train_remote(model_config, data)


def _train_remote(model_config: dict, data: dict) -> dict:
    """
    Delegasikan pelatihan ke GPU cloud Modal (modal_app.py::train_remote)
    saat tidak ada GPU lokal tersedia. Model dilatih dan checkpoint-nya
    diunggah ke MLflow oleh Modal; di sini checkpoint diunduh kembali ke
    save_dir lokal agar strukturnya identik dengan hasil training lokal.
    """
    import modal

    print("  Tidak ada GPU lokal terdeteksi — melatih via GPU cloud Modal...")
    train_fn = modal.Function.from_name('absa-training', 'train_remote')
    train_result = train_fn.remote(model_config, data)

    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        mlflow_cfg.get('tracking_uri', 'http://localhost:5000'),
    )
    mlflow.set_tracking_uri(tracking_uri)

    save_dir = train_result['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Mengunduh checkpoint dari MLflow run {train_result['run_id'][:8]} ke {save_dir}/ ...")
    downloaded_dir = mlflow.artifacts.download_artifacts(
        run_id        = train_result['run_id'],
        artifact_path = 'checkpoint',
        dst_path      = save_dir,
    )
    # download_artifacts menaruh isi artifact_path di save_dir/checkpoint/ —
    # ratakan ke save_dir langsung agar sama dengan struktur hasil training lokal.
    if os.path.normpath(downloaded_dir) != os.path.normpath(save_dir):
        for fname in os.listdir(downloaded_dir):
            os.replace(os.path.join(downloaded_dir, fname), os.path.join(save_dir, fname))
        os.rmdir(downloaded_dir)

    print(f"  Pelatihan (Modal) selesai. Best Val Sentiment F1: {train_result['best_val_f1']:.4f}")
    return train_result


def _train_local(model_config: dict, data: dict) -> dict:
    """
    Latih model dan catat eksperimen ke MLflow, langsung di proses ini
    (dipanggil saat GPU lokal tersedia — laptop ber-GPU, atau di dalam
    kontainer Modal yang memang menyediakan GPU).

    Membuka satu MLflow run yang akan dilanjutkan oleh tahap evaluate_model
    (via mlflow.start_run(run_id=...)) untuk mencatat metrik test set dalam
    run yang sama.
    """
    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        mlflow_cfg.get('tracking_uri', 'http://localhost:5000'),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(model_config['experiment']['name'])

    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'

    seed = model_config['experiment'].get('seed', 42)
    set_seed(seed)

    run_name = model_config['experiment'].get('run_name', model_config['experiment']['name'])

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Catat metadata versi
        mlflow.set_tag('git_commit',     get_git_commit())
        mlflow.set_tag('model_name',     model_config['representation']['model_name'])
        mlflow.set_tag('model_revision', model_config['representation'].get('model_revision', 'main'))
        mlflow.set_tag('mlflow.note.content', model_config['experiment'].get('description', ''))
        mlflow.set_tag('triggered_by', os.environ.get('ABSA_TRIGGER_REASON', 'scheduled'))

        mlflow.log_param('experiment.seed', seed)
        for k, v in flatten_config(model_config).items():
            mlflow.log_param(k, str(v)[:500])

        # Latih model — pipeline/train_model.py mencatat metrik per epoch ke run aktif
        print(f"\n  MLflow Run ID: {run_id}")
        trained = train_model(model_config, data)

        mlflow.log_metric('best_val_sentiment_f1', trained['best_val_f1'])
        mlflow.log_metric('best_val_detection_f1', trained['best_val_det_f1'])

        # Log artefak kecil (bukan checkpoint .pt)
        _LARGE_EXTS = {'.bin', '.safetensors', '.pt', '.pth'}
        save_dir = trained['save_dir']
        if os.path.isdir(save_dir):
            for fname in os.listdir(save_dir):
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() not in _LARGE_EXTS:
                    mlflow.log_artifact(fpath, artifact_path='model_artifacts')

    print(f"  Pelatihan selesai. Best Val Sentiment F1: {trained['best_val_f1']:.4f}")

    # Kembalikan hanya metadata yang dapat diserialisasi Metaflow (bukan objek model)
    return {
        'run_id'         : run_id,
        'save_dir'       : save_dir,
        'best_val_f1'    : trained['best_val_f1'],
        'best_val_det_f1': trained['best_val_det_f1'],
    }
