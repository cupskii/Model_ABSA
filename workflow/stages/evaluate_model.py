import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import mlflow

from model.checkpoint_io import load_model_from_checkpoint
from pipeline.evaluate_model import evaluate_model


def run_evaluate_model(model_config: dict, train_result: dict, data: dict) -> dict:
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = train_result['save_dir']
    run_id   = train_result['run_id']

    print(f"  Memuat model dari: {save_dir}/best_model.pt")
    model, tokenizer, _ = load_model_from_checkpoint(save_dir, device)

    trained_reconstructed = {
        'model'    : model,
        'tokenizer': tokenizer,
        'device'   : device,
        'save_dir' : save_dir,
    }

    metrics = evaluate_model(model_config, trained_reconstructed, data)

    # Lanjutkan MLflow run yang sama untuk mencatat metrik test set
    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or mlflow_cfg.get(
        'tracking_uri', 'http://localhost:5000',
    )
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        # Log artefak evaluasi yang dihasilkan pipeline/evaluate_model.py
        for fname in ('classification_report.txt', 'confusion_matrix.png'):
            fpath = os.path.join(save_dir, fname)
            if os.path.isfile(fpath):
                mlflow.log_artifact(fpath, artifact_path='model_artifacts')

    print(f"  Test Mean Sentiment F1: {metrics.get('test_mean_sentiment_f1', 0):.4f} ← metrik utama")
    return metrics
