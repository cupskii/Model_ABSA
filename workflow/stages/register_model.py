"""
Tahap Registrasi Model
=======================
Komponen baru dalam automated ML workflow pipeline.

Mendaftarkan model yang sudah divalidasi ke MLflow Model Registry sebagai
versi baru dengan stage "Staging". Model hanya didaftarkan jika tahap
validate_model menyatakan bahwa model lolos kedua pengujian.

Pendekatan registrasi: custom pyfunc flavor, bukan mlflow.pytorch.log_model()
-------------------------------------------------------------------------------
save_dir (hasil pipeline/train_model.py) sudah menjadi bundle mandiri yang
cukup untuk reproduksi/deployment tanpa dependensi eksternal:
  best_model.pt   — state_dict + config + metrik epoch terbaik
  config.yaml     — salinan config eksperimen dalam format yang mudah dibaca
  tokenizer/      — hasil tokenizer.save_pretrained(), tanpa akses HF Hub
  metrics.json    — metrik test set (ditulis di tahap ini)

  (riwayat metrik per epoch, termasuk breakdown per aspek, di-log langsung
  ke MLflow run via mlflow.log_metric — dapat dilihat sebagai chart di
  MLflow UI tanpa artefak file terpisah)

Bundle ini dibungkus wrapper model/absa_pyfunc.py (mlflow.pyfunc.PythonModel)
lewat mlflow.pyfunc.log_model(), lalu didaftarkan ke registry dengan
mlflow.register_model(). Pendekatan ini dipilih alih-alih mlflow.pytorch.log_model()
langsung karena:
  - mlflow.pytorch.log_model() mem-pickle objek ABSAModel secara utuh,
    sehingga artifact tetap terikat erat pada path modul Python persis
    saat training (model.absa_model.ABSAModel) — rapuh terhadap refactor
    atau perpindahan environment.
  - Wrapper pyfunc hanya mem-pickle dirinya sendiri (kode tipis); bobot
    model tetap file checkpoint biasa yang dimuat ulang eksplisit lewat
    ABSAModel(...) + load_state_dict() di load_context() — logika
    rekonstruksi tetap terlihat, tidak tersembunyi di balik pickle.
  - Flavor pyfunc memberi satu entrypoint standar (mlflow.pyfunc.load_model())
    yang bisa dipakai registry, `mlflow models serve`, maupun aplikasi
    konsumen — tanpa perlu menulis file MLmodel secara manual.
"""

import os
import sys
import json

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

from model.absa_pyfunc import ABSAPyfuncModel


_PYFUNC_CODE_PATHS = [
    os.path.join(_REPO_ROOT, 'model'),
    os.path.join(_REPO_ROOT, 'preprocessing'),
]


def run_register_model(
    model_validation: dict,
    train_result: dict,
    metrics: dict,
    workflow_config: dict,
    model_config: dict,
) -> dict:
    """
    Daftarkan model yang sudah divalidasi ke MLflow Model Registry.

    Parameters
    ----------
    model_validation : dict — output run_validate_model()
    train_result     : dict — output run_train_model() (run_id, save_dir, ...)
    metrics          : dict — output run_evaluate_model() (metrik test set)
    workflow_config  : dict — konfigurasi workflow dari pipeline_config.yaml
    model_config     : dict — konfigurasi model dari YAML eksperimen

    Returns
    -------
    dict:
      registered    : bool   — apakah model berhasil didaftarkan
      model_version : str    — versi model di registry (None jika tidak didaftarkan)
      model_stage   : str    — stage yang ditetapkan
      reason        : str    — penjelasan keputusan
    """
    if not model_validation['passed']:
        failure_summary = '; '.join(model_validation['failure_reasons'])
        print(f"  Model tidak didaftarkan: {failure_summary}")
        return {
            'registered'   : False,
            'model_version': None,
            'model_stage'  : None,
            'reason'       : f"Model tidak lolos validasi: {failure_summary}",
        }

    mlflow_wf  = workflow_config.get('mlflow', {})
    mlflow_mdl = model_config.get('mlflow', {})

    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or (
        mlflow_wf.get('tracking_uri') or mlflow_mdl.get('tracking_uri', 'http://localhost:5000')
    )
    registry_name  = mlflow_wf.get('registry_name', mlflow_mdl.get('registry_name', 'absa_indobert'))
    register_stage = mlflow_wf.get('register_stage', 'Staging')

    mlflow.set_tracking_uri(tracking_uri)

    run_id   = train_result['run_id']
    save_dir = train_result['save_dir']

    # Sertakan metrik test set ke dalam bundle artifact agar setiap versi
    # model di registry membawa metriknya sendiri tanpa perlu query terpisah
    # ke MLflow run.
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  Mengunggah bundle model dari {save_dir} ke MLflow run {run_id[:8]} (pyfunc)...")
    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path    = 'model',
            python_model     = ABSAPyfuncModel(),
            artifacts        = {'checkpoint': save_dir},
            code_paths       = _PYFUNC_CODE_PATHS,
            # Sengaja HANYA package yang benar-benar disentuh load_context()/
            # predict() (absa_pyfunc.py) — bukan requirements.txt training
            # penuh. numpy/pandas eksplisit di sini karena predict() memanggil
            # .numpy() dan mengecek isinstance(..., pd.DataFrame) langsung,
            # bukan cuma transitive dependency torch/mlflow. scikit-learn dan
            # iterative-stratification SENGAJA tidak disertakan — keduanya
            # training-only (dulu ikut ter-import transitif lewat
            # model/absa_model.py dan preprocessing/preprocessing_functions.py,
            # sudah diperbaiki jadi local import supaya tidak lagi bocor ke
            # environment serving ini).
            pip_requirements = [
                'torch>=2.0.0',
                'transformers==5.12.0',
                'numpy',
                'pandas',
            ],
        )

    model_uri = f"runs:/{run_id}/model"
    print(f"  Mendaftarkan {model_uri} ke registry '{registry_name}'...")
    model_version = mlflow.register_model(model_uri=model_uri, name=registry_name)

    client = MlflowClient()
    # archive_existing_versions=True aman dipakai di sini karena Uji 3 di
    # validate_model.py (_check_vs_staging_reeval) sudah memastikan model
    # baru benar-benar lebih baik daripada versi yang sedang di register_stage
    # sebelum sampai ke tahap ini — versi lama yang di-archive memang kalah
    # performanya, bukan sekadar tergeser posisi.
    client.transition_model_version_stage(
        name                    = registry_name,
        version                 = model_version.version,
        stage                   = register_stage,
        archive_existing_versions = True,
    )

    # Tambahkan deskripsi versi dengan metrik utama
    sentiment_f1 = metrics.get('test_mean_sentiment_f1', 0.0)
    detection_f1 = metrics.get('test_mean_detect_f1', 0.0)
    client.update_model_version(
        name        = registry_name,
        version     = model_version.version,
        description = (
            f"Test Sentiment F1: {sentiment_f1:.4f} | "
            f"Test Detection F1: {detection_f1:.4f} | "
            f"MLflow Run: {run_id[:8]}"
        ),
    )

    print(f"  Model terdaftar: {registry_name} v{model_version.version} → stage '{register_stage}'")

    return {
        'registered'   : True,
        'model_version': model_version.version,
        'model_stage'  : register_stage,
        'reason'       : (
            f"Model lolos validasi dan didaftarkan sebagai "
            f"{registry_name} v{model_version.version} ({register_stage})"
        ),
    }
