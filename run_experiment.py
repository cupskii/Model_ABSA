import os
import json
import argparse
import subprocess

import yaml
import torch
import mlflow
import mlflow.pytorch

from model.absa_model        import set_seed
from pipeline.validate_data  import validate_data
from pipeline.prepare_data   import prepare_data
from pipeline.train_model    import train_model
from pipeline.evaluate_model import evaluate_model


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    """Ambil commit hash HEAD sebagai pointer versi kode."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except Exception:
        return 'unknown'



def flatten_config(cfg: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, key))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _save_model_checkpoint(ckpt_path: str, config: dict, run_id: str) -> None:
    """
    Simpan checkpoint model (.pt) ke lokasi persisten sesuai environment:
      - Colab  → Google Drive (transfer lokal dalam infrastruktur Google, cepat)
      - Lokal  → lewati (file sudah ada di save_dir)

    URI tujuan dicatat ke MLflow sebagai tag 'model_checkpoint_uri'.
    """
    if not os.path.isfile(ckpt_path):
        print("  [!] best_model.pt tidak ditemukan, simpan checkpoint dilewati.")
        return

    mlflow_cfg   = config.get('mlflow', {})
    run_name     = config['experiment'].get('run_name', run_id[:8])
    registry_name = mlflow_cfg.get('registry_name', 'absa_indobert')

    if _is_colab():
        # ── Simpan ke Google Drive ─────────────────────────────────────────
        gdrive_base = mlflow_cfg.get('gdrive_dir', 'absa_models')
        drive_root  = '/content/drive/MyDrive'

        if not os.path.isdir(drive_root):
            print("  Mounting Google Drive...")
            from google.colab import drive
            drive.mount('/content/drive')

        dest_dir  = os.path.join(drive_root, gdrive_base, registry_name, run_name)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, 'best_model.pt')

        print(f"  Menyalin model ke Google Drive → {dest_path} ...")
        import shutil
        shutil.copy2(ckpt_path, dest_path)

        gdrive_uri = f"gdrive://MyDrive/{gdrive_base}/{registry_name}/{run_name}/best_model.pt"
        mlflow.set_tag('model_checkpoint_uri',    gdrive_uri)
        mlflow.set_tag('model_checkpoint_run_id', run_id)
        print(f"  Tersimpan di Google Drive: {gdrive_uri}")

    else:
        # ── Training lokal: file sudah ada di save_dir, catat path-nya ──
        abs_path = os.path.abspath(ckpt_path)
        mlflow.set_tag('model_checkpoint_uri',    f"local://{abs_path}")
        mlflow.set_tag('model_checkpoint_run_id', run_id)
        print(f"  Checkpoint tersimpan lokal: {abs_path}")


# ── ORKESTRASI PIPELINE ────────────────────────────────────────────────────────

def run_experiment(config_path: str) -> dict:
    """
    Jalankan satu eksperimen lengkap dari berkas konfigurasi:
      1. Muat konfigurasi
      2. Buka sesi MLflow dan catat seluruh metadata
      3. Validasi data
      4. Persiapan data
      5. Pelatihan model
      6. Evaluasi model
      7. Catat metrik dan simpan artefak ke MLflow
      8. Tutup sesi MLflow

    Parameters
    ----------
    config_path : str — path ke berkas YAML konfigurasi eksperimen

    Returns
    -------
    dict — metrik evaluasi test set
    """
    config = load_config(config_path)

    seed = config['experiment'].get('seed', 42)
    set_seed(seed)

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(config['experiment']['name'])

    with mlflow.start_run(run_name=config['experiment'].get('run_name', config['experiment']['name'])) as run:
        run_id = run.info.run_id

        print(f"\n{'='*60}")
        print(f"EKSPERIMEN: {config['experiment']['name']}")
        print(f"Deskripsi : {config['experiment'].get('description', '-')}")
        print(f"MLflow Run ID: {run_id}")
        print(f"{'='*60}")

        # ── Catat metadata versi ───────────────────────────────────
        git_commit = get_git_commit()

        mlflow.set_tag('git_commit',     git_commit)
        mlflow.set_tag('model_name',     config['representation']['model_name'])
        mlflow.set_tag('model_revision', config['representation'].get('model_revision', 'main'))
        mlflow.set_tag('mlflow.note.content', config['experiment'].get('description', ''))
        mlflow.log_param('experiment.seed', seed)

        mlflow.log_artifact("requirements.txt")

        print(f"\nVersi kode  : {git_commit[:12]}")
        print(f"Model       : {config['representation']['model_name']} "
              f"@ {config['representation'].get('model_revision', 'main')}")

        # ── Catat seluruh parameter konfigurasi ───────────────────
        for k, v in flatten_config(config).items():
            mlflow.log_param(k, str(v)[:500])

        # ── 1. Validasi data ───────────────────────────────────────
        print("\n[1/4] Validasi data...")
        val_report = validate_data(config)
        if not val_report['passed']:
            raise RuntimeError(f"Validasi data gagal: {val_report['issues']}")
        print(f"  OK — {val_report['total_rows']} baris, semua pemeriksaan lulus")
        if val_report['issues']:
            print(f"  Peringatan: {val_report['issues']}")
        mlflow.log_param('data.n_rows', val_report['total_rows'])

        # ── 2. Persiapan data ──────────────────────────────────────
        print("\n[2/4] Persiapan data...")
        data = prepare_data(config)
        n_train = len(data['df_train'])
        n_test  = len(data['df_test'])
        n_val   = len(data['df_val'])
        print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
        mlflow.log_param('data.n_train', n_train)
        mlflow.log_param('data.n_val', n_val)
        mlflow.log_param('data.n_test',  n_test)

        # Simpan class weights sebagai artefak
        save_dir = config['model']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        cw_path = os.path.join(save_dir, 'class_weights.json')
        with open(cw_path, 'w', encoding='utf-8') as f:
            json.dump(data['class_weights'], f, indent=2, ensure_ascii=False)

        # ── 3. Pelatihan model ─────────────────────────────────────
        print("\n[3/4] Pelatihan model...")
        if torch.cuda.is_available():
            trained = train_model(config, data)
        else:
            from workflow.stages.train_model import run_train_model
            from model.checkpoint_io import load_model_from_checkpoint

            train_result = run_train_model(
                config, data, git_commit=git_commit, run_id=run_id,
            )
            device = torch.device('cpu')
            model, tokenizer, _ = load_model_from_checkpoint(train_result['save_dir'], device)
            trained = {
                'model'          : model,
                'tokenizer'      : tokenizer,
                'device'         : device,
                'save_dir'       : train_result['save_dir'],
                'best_val_f1'    : train_result['best_val_f1'],
                'best_val_det_f1': train_result['best_val_det_f1'],
            }

        mlflow.log_metric('best_val_sentiment_f1', trained['best_val_f1'])
        mlflow.log_metric('best_val_detection_f1', trained['best_val_det_f1'])

        print("\n[4/4] Evaluasi model pada test set...")
        metrics = evaluate_model(config, trained, data)
        mlflow.log_metrics(metrics)

        _LARGE_EXTS = {'.bin', '.safetensors', '.pt', '.pth'}
        if os.path.isdir(save_dir):
            for fname in os.listdir(save_dir):
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() not in _LARGE_EXTS:
                    mlflow.log_artifact(fpath, artifact_path='model_artifacts')

        mlflow.log_artifact(config_path, artifact_path='config')

        ckpt_path = os.path.join(save_dir, 'best_model.pt')
        _save_model_checkpoint(ckpt_path, config, run_id)

        print(f"\n{'='*60}")
        print("EKSPERIMEN SELESAI")
        print(f"{'='*60}")
        print(f"  Test Mean Sentiment F1 : {metrics.get('test_mean_sentiment_f1', 0):.4f}  <- metrik utama")
        print(f"  Test Mean Detection F1 : {metrics.get('test_mean_detect_f1', 0):.4f}")
        print(f"  Test Pair-based Micro F1     : {metrics.get('test_pair_micro_f1', 0):.4f}")
        print(f"  Test Aspect Detection F1     : {metrics.get('test_aspect_detection_f1', 0):.4f}")
        print(f"  MLflow Run ID          : {run_id}")
        print(f"{'='*60}\n")

    return metrics



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Jalankan satu eksperimen ABSA dari berkas konfigurasi YAML.')
    parser.add_argument(
        'config',
        type=str,
        help='Path ke berkas konfigurasi YAML (misal: configs/experiment_indobert_baseline.yaml)',
    )
    args = parser.parse_args()

    run_experiment(args.config)
