"""
Baseline Acak (Uniform Random) — ABSA
=======================================
Baseline pembanding paling dasar: setiap aspek diprediksi dengan peluang
sama rata di antara seluruh kelasnya (termasuk 'None'), tanpa melihat data
sama sekali. Dipakai sebagai lower bound saat membandingkan model IndoBERT.

Memakai test split yang SAMA dengan eksperimen model (prepare_data() pada
config data yang identik, lihat configs/baseline_random.yaml) supaya
perbandingan metrik valid — bukan test set yang berbeda secara kebetulan.

Jalankan:
  python run_baseline_random.py
  python run_baseline_random.py --config_path configs/baseline_random.yaml
"""

import os
import argparse

import numpy as np
import mlflow

from preprocessing.preprocessing_functions import FINAL_ASPECTS, NUM_CLASSES
from pipeline.prepare_data   import prepare_data
from pipeline.evaluate_model import compute_metrics_from_predictions
from run_experiment          import load_config, get_git_commit


def predict_random(df_test, seed: int) -> tuple:
    """
    Prediksi uniform random per aspek pada test set. Satu RNG dipakai
    berurutan lintas aspek (bukan RNG baru per aspek) supaya seluruh
    baseline reproducible dari satu seed saja.
    """
    rng = np.random.default_rng(seed)
    n   = len(df_test)

    all_preds  = {}
    all_labels = {}
    for asp in FINAL_ASPECTS:
        all_preds[asp]  = rng.integers(0, NUM_CLASSES[asp], size=n).tolist()
        all_labels[asp] = df_test[f'lbl_{asp}'].tolist()

    return all_preds, all_labels


def run_baseline_random(config_path: str) -> dict:
    config = load_config(config_path)
    seed   = config['experiment'].get('seed', 42)

    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or config['mlflow']['tracking_uri']
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config['experiment']['name'])

    print(f"\n{'='*60}")
    print("BASELINE ACAK (uniform random)")
    print(f"Deskripsi : {config['experiment'].get('description', '-')}")
    print(f"{'='*60}")

    print("\n[1/2] Persiapan data...")
    data = prepare_data(config)
    print(f"  Test: {len(data['df_test'])} baris")

    print("\n[2/2] Prediksi acak & evaluasi...")
    all_preds, all_labels = predict_random(data['df_test'], seed)
    metrics = compute_metrics_from_predictions(all_preds, all_labels)

    with mlflow.start_run(run_name=config['experiment'].get('run_name', 'baseline_random')) as run:
        mlflow.set_tag('baseline_type', 'random_uniform')
        mlflow.set_tag('git_commit',    get_git_commit())
        mlflow.set_tag('mlflow.note.content', config['experiment'].get('description', ''))
        mlflow.log_param('experiment.seed', seed)
        mlflow.log_param('data.path',   config['data']['path'])
        mlflow.log_param('data.n_test', len(data['df_test']))
        mlflow.log_metrics(metrics)
        run_id = run.info.run_id

    print(f"\n{'='*60}")
    print("BASELINE ACAK SELESAI")
    print(f"{'='*60}")
    print(f"  MLflow Run ID          : {run_id}")
    print(f"  Test Mean Sentiment F1 : {metrics['test_mean_sentiment_f1']:.4f}")
    print(f"  Test Mean Detection F1 : {metrics['test_mean_detect_f1']:.4f}")
    print(f"{'='*60}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline acak (uniform random) untuk ABSA')
    parser.add_argument(
        '--config_path',
        default='configs/baseline_random.yaml',
        help='Path konfigurasi baseline YAML',
    )
    args = parser.parse_args()

    run_baseline_random(args.config_path)
