"""
Modal App — GPU Training Worker untuk ABSA Pipeline
=====================================================
Menggantikan alur manual Google Colab: fungsi di sini menjalankan tahap
train_step (workflow/stages/train_model.py) di GPU cloud Modal, dipanggil
secara sinkron dan programatik dari flow Metaflow — bukan diklik manual
dari notebook. Tahap evaluate/validate/register tetap berjalan lokal
seperti biasa, memuat ulang checkpoint dari disk.

Kredensial (MLFLOW_TRACKING_URI, kunci B2) diambil dari Modal Secret
'absa-mlflow-creds', tidak pernah ditulis di kode ini.

Penggunaan
----------
  # Smoke test langsung dari GPU Modal (tanpa Metaflow), dari folder model_ABSA:
  modal run modal_app.py

  # Deploy supaya bisa dipanggil terus-menerus dari flow Metaflow via
  # modal.Function.from_name("absa-training", "train_remote"):
  modal deploy modal_app.py
"""

import os

import modal

app = modal.App("absa-training")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "transformers==5.12.0",
        "numpy==2.5.0",
        "pandas==2.3.3",
        "scikit-learn==1.9.0",
        "iterative-stratification==0.1.9",
        "matplotlib==3.11.0",
        "mlflow==3.14.0",
        "boto3==1.43.36",
        "psutil==7.2.2",
        "PyYAML==6.0.3",
    )
    .add_local_python_source(
        "model", "preprocessing", "pipeline", "workflow", "run_experiment"
    )
)


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("absa-mlflow-creds")],
    timeout=1800,
)
def train_remote(model_config: dict, data: dict) -> dict:
    """
    Latih model ABSA di GPU Modal, catat ke MLflow, lalu unggah bundle
    checkpoint (best_model.pt, tokenizer, config.yaml) sebagai MLflow
    artifact supaya bisa diunduh kembali oleh proses lokal
    (workflow/stages/train_model.py) tanpa perlu akses ke disk Modal.

    run_train_model() mendeteksi torch.cuda.is_available() secara mandiri —
    di dalam kontainer Modal ini nilainya True, sehingga otomatis melatih
    langsung di GPU tanpa cabang logika tambahan di sini.

    Parameters
    ----------
    model_config : dict — konfigurasi eksperimen (hasil load_config YAML)
    data         : dict — output pipeline.prepare_data.prepare_data()
                   (df_train, df_val, df_test, class_weights)

    Returns
    -------
    dict — output run_train_model() (run_id, save_dir, best_val_f1, ...)
    """
    import mlflow

    from workflow.stages.train_model import run_train_model

    train_result = run_train_model(model_config, data)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_id=train_result["run_id"]):
        mlflow.log_artifacts(train_result["save_dir"], artifact_path="checkpoint")

    return train_result


@app.local_entrypoint()
def main(config_path: str = "configs/experiment_indobert_baseline.yaml"):
    """
    Smoke test: latih model penuh di GPU Modal, tanpa Metaflow dan tanpa
    evaluate (evaluate_step butuh torch, sengaja tidak diinstal di venv
    lokal ringan — lihat workflow/stages/evaluate_model.py untuk itu).
      modal run modal_app.py
      modal run modal_app.py --config-path configs/experiment_indobert_baseline.yaml
    """
    import yaml

    from pipeline.prepare_data import prepare_data

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    print(f"Menyiapkan data dari {config_path} ...")
    data = prepare_data(model_config)
    print(
        f"  Train: {len(data['df_train'])} | "
        f"Val: {len(data['df_val'])} | Test: {len(data['df_test'])}"
    )

    print("Memanggil GPU Modal untuk training ...")
    train_result = train_remote.remote(model_config, data)

    print("\n" + "=" * 60)
    print("HASIL TRAINING DI MODAL")
    print("=" * 60)
    print(f"  MLflow Run ID : {train_result['run_id']}")
    print(f"  Save Dir      : {train_result['save_dir']}")
    print(f"  Best Val F1   : {train_result['best_val_f1']:.4f}")
    print("=" * 60)
