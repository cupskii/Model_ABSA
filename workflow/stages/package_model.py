
import os
import sys

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _REPO_ROOT)

import mlflow
import mlflow.pyfunc

from model.absa_pyfunc import ABSAPyfuncModel


_PYFUNC_CODE_PATHS = [
    os.path.join(_REPO_ROOT, 'model'),
    os.path.join(_REPO_ROOT, 'preprocessing'),
]


def run_package_model(train_result: dict, model_config: dict) -> dict:
    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or mlflow_cfg.get(
        'tracking_uri', 'http://localhost:5000',
    )
    mlflow.set_tracking_uri(tracking_uri)

    run_id   = train_result['run_id']
    save_dir = train_result['save_dir']

    print(f"  Mengunggah bundle model dari {save_dir} ke MLflow run {run_id[:8]} (pyfunc)...")
    with mlflow.start_run(run_id=run_id):
        mlflow.pyfunc.log_model(
            artifact_path    = 'model',
            python_model     = ABSAPyfuncModel(),
            artifacts        = {'checkpoint': save_dir},
            code_paths       = _PYFUNC_CODE_PATHS,
            # Sengaja HANYA package yang benar-benar disentuh load_context()/
            # predict() (absa_pyfunc.py) — bukan requirements.txt training
            # penuh. Lihat penjelasan lengkap di riwayat register_model.py.
            pip_requirements = [
                'torch>=2.0.0',
                'transformers==5.12.0',
                'numpy',
                'pandas',
            ],
        )

    model_uri = f"runs:/{run_id}/model"
    print(f"  Model ter-package di {model_uri} (belum didaftarkan/dipromosikan)")

    return {
        'packaged': True,
        'model_uri': model_uri,
    }
