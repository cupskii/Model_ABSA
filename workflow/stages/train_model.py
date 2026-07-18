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
"""

import os
import sys
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.pytorch
import torch

from model.absa_model import set_seed
from pipeline.train_model import train_model
from run_experiment import flatten_config, get_git_commit


def run_train_model(model_config: dict, data: dict, git_commit: Optional[str] = None,
                     run_id: Optional[str] = None) -> dict:
    """
    Latih model dan catat eksperimen ke MLflow.

    Membuka satu MLflow run yang akan dilanjutkan oleh tahap evaluate_model
    (via mlflow.start_run(run_id=...)) untuk mencatat metrik test set dalam
    run yang sama.

    Parameters
    ----------
    model_config : dict — konfigurasi model dari YAML eksperimen
    data         : dict — output run_prepare_data() (df_train, df_val, df_test, class_weights)
    git_commit   : str, opsional — commit hash yang sudah dihitung di mesin
                   pemanggil asli (lihat catatan di _train_local).
    run_id       : str, opsional — MLflow run ID yang sudah aktif untuk
                   dilanjutkan (dipakai run_experiment.py, yang membuka
                   run-nya sendiri sebelum memanggil training). Kalau None,
                   fungsi ini membuat run baru sendiri (dipakai flow
                   Metaflow, lihat workflow/flow.py — train_step tidak
                   membuka run terlebih dulu).

    Returns
    -------
    dict:
      run_id          : str   — MLflow run ID (untuk dilanjutkan di evaluate_model)
      save_dir        : str   — direktori checkpoint model
      best_val_f1     : float — Sentiment F1 terbaik pada validation set
      best_val_det_f1 : float — Detection F1 pada epoch terbaik
    """
    if torch.cuda.is_available():
        return _train_local(model_config, data, git_commit=git_commit, run_id=run_id)
    return _train_remote(model_config, data, git_commit=git_commit, run_id=run_id)


def _train_remote(model_config: dict, data: dict, git_commit: Optional[str] = None,
                   run_id: Optional[str] = None) -> dict:
    """
    Delegasikan pelatihan ke GPU cloud Modal (modal_app.py::train_remote)
    saat tidak ada GPU lokal tersedia. Model dilatih dan checkpoint-nya
    diunggah ke MLflow oleh Modal; di sini checkpoint diunduh kembali ke
    save_dir lokal agar strukturnya identik dengan hasil training lokal.

    git_commit dihitung DI SINI (mesin lokal yang punya .git sungguhan),
    lalu dikirim ke kontainer Modal — image Modal tidak menyertakan
    direktori .git maupun binary git, jadi menghitungnya di dalam
    kontainer selalu menghasilkan 'unknown'/commit yang salah.
    """
    import modal

    git_commit = git_commit or get_git_commit()

    print("  Tidak ada GPU lokal terdeteksi — melatih via GPU cloud Modal...")
    train_fn = modal.Function.from_name('absa-training', 'train_remote')
    train_result = train_fn.remote(model_config, data, git_commit, run_id)

    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or mlflow_cfg.get(
        'tracking_uri', 'http://localhost:5000',
    )
    mlflow.set_tracking_uri(tracking_uri)

    save_dir = train_result['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    print(f"  Mengunduh checkpoint dari MLflow run {train_result['run_id'][:8]} ke {save_dir}/ ...")

    checkpoint_uri = train_result.get('checkpoint_artifact_uri')
    if not checkpoint_uri:
        run = mlflow.MlflowClient().get_run(train_result['run_id'])
        checkpoint_uri = f"{run.info.artifact_uri.rstrip('/')}/checkpoint"

    print(f"  Artifact URI: {checkpoint_uri}")
    downloaded_dir = mlflow.artifacts.download_artifacts(
        artifact_uri = checkpoint_uri,
        dst_path     = save_dir,
    )
    # download_artifacts menaruh isi artifact_path di save_dir/checkpoint/ —
    # ratakan ke save_dir langsung agar sama dengan struktur hasil training lokal.
    if os.path.normpath(downloaded_dir) != os.path.normpath(save_dir):
        for fname in os.listdir(downloaded_dir):
            os.replace(os.path.join(downloaded_dir, fname), os.path.join(save_dir, fname))
        os.rmdir(downloaded_dir)

    print(f"  Pelatihan (Modal) selesai. Best Val Sentiment F1: {train_result['best_val_f1']:.4f}")
    return train_result


def _train_local(model_config: dict, data: dict, git_commit: Optional[str] = None,
                  run_id: Optional[str] = None) -> dict:
    """
    Latih model dan catat eksperimen ke MLflow, langsung di proses ini
    (dipanggil saat GPU lokal tersedia — laptop ber-GPU, atau di dalam
    kontainer Modal yang memang menyediakan GPU).

    Membuka satu MLflow run yang akan dilanjutkan oleh tahap evaluate_model
    (via mlflow.start_run(run_id=...)) untuk mencatat metrik test set dalam
    run yang sama.

    git_commit : str, opsional — kalau dikirim dari luar (lihat _train_remote),
    dipakai apa adanya karena kontainer Modal tidak punya .git untuk
    dihitung ulang; kalau None (benar-benar jalan lokal), dihitung di sini.
    run_id     : str, opsional — kalau diisi, lanjutkan run yang sudah dibuka
    pemanggil (mis. run_experiment.py, yang sudah mencatat seluruh metadata
    versi/param sebelum memanggil training) alih-alih membuat run baru dan
    mencatat metadata itu dua kali.
    """
    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or mlflow_cfg.get(
        'tracking_uri', 'http://localhost:5000',
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(model_config['experiment']['name'])

    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'

    seed = model_config['experiment'].get('seed', 42)
    set_seed(seed)

    run_name    = model_config['experiment'].get('run_name', model_config['experiment']['name'])
    continuing  = run_id is not None
    run_context = mlflow.start_run(run_id=run_id) if continuing else mlflow.start_run(run_name=run_name)

    with run_context as run:
        run_id = run.info.run_id

        if not continuing:
            # Metadata versi — kalau continuing, pemanggil sudah mencatat ini.
            mlflow.set_tag('git_commit',     git_commit or get_git_commit())
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
    result = {
        'run_id'         : run_id,
        'save_dir'       : save_dir,
        'best_val_f1'    : trained['best_val_f1'],
        'best_val_det_f1': trained['best_val_det_f1'],
    }
    return result
