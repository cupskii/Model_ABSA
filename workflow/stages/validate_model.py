import os
import sys
import glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import mlflow
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient

from model.checkpoint_io import load_model_from_checkpoint
from pipeline.evaluate_model import compute_test_metrics



def _check_threshold(metrics: dict, validation_cfg: dict) -> dict:
    """Periksa apakah metrik test set memenuhi nilai minimum yang ditetapkan."""
    min_sent = validation_cfg['min_sentiment_f1']
    min_det  = validation_cfg['min_detection_f1']

    actual_sent = metrics.get('test_mean_sentiment_f1', 0.0)
    actual_det  = metrics.get('test_mean_detect_f1', 0.0)

    passed = (actual_sent >= min_sent) and (actual_det >= min_det)

    reasons = []
    if actual_sent < min_sent:
        reasons.append(
            f"Sentiment F1 ({actual_sent:.4f}) < threshold ({min_sent})"
        )
    if actual_det < min_det:
        reasons.append(
            f"Detection F1 ({actual_det:.4f}) < threshold ({min_det})"
        )

    return {
        'passed' : passed,
        'reasons': reasons,
        'details': {
            'actual_sentiment_f1'  : actual_sent,
            'actual_detection_f1'  : actual_det,
            'min_sentiment_f1'     : min_sent,
            'min_detection_f1'     : min_det,
        },
    }



def _reevaluate_checkpoint(registry_name: str, stage: str, data: dict) -> dict:
    """
    Unduh checkpoint model dari MLflow Model Registry (by stage) dan evaluasi
    ulang pada test set (`data`) saat ini.

    Dipakai HANYA ketika dataset yang dipakai baseline berbeda dari dataset
    saat ini (lihat _check_vs_production/_check_vs_staging_reeval) — kalau
    dataset sama, metrik yang sudah tercatat di MLflow run cukup dipakai
    langsung tanpa unduh+inferensi ulang yang mahal.
    """
    model_uri = f"models:/{registry_name}/{stage}"
    local_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    # register_model.py meng-log save_dir via artifacts={'checkpoint': save_dir}
    # di mlflow.pyfunc.log_model() — tapi MLflow menyimpan foldernya memakai
    # basename asli save_dir (mis. "model_output" sesuai save_dir di config
    # eksperimen), BUKAN key dict 'checkpoint'. Nama basename itu bisa beda
    # antar versi kalau save_dir di config eksperimen pernah di-rename, jadi
    # dicari langsung folder yang berisi best_model.pt, bukan diasumsikan tetap.
    matches = glob.glob(os.path.join(local_dir, 'artifacts', '*', 'best_model.pt'))
    if not matches:
        raise FileNotFoundError(
            f"best_model.pt tidak ditemukan di artifact '{model_uri}' (local_dir={local_dir})"
        )
    checkpoint_dir = os.path.dirname(matches[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_model, baseline_tokenizer, baseline_cfg = load_model_from_checkpoint(checkpoint_dir, device)
    return compute_test_metrics(baseline_model, baseline_tokenizer, baseline_cfg, data)


def _get_production_metric(workflow_config: dict, model_config: dict) -> dict:
    mlflow_wf  = workflow_config.get('mlflow', {})
    mlflow_mdl = model_config.get('mlflow', {})

    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or (
        mlflow_wf.get('tracking_uri') or mlflow_mdl.get('tracking_uri', 'http://localhost:5000')
    )
    mlflow.set_tracking_uri(tracking_uri)

    registry_name     = mlflow_wf.get('registry_name', mlflow_mdl.get('registry_name', 'absa_indobert'))
    production_stage  = mlflow_wf.get('production_stage', 'Production')
    comparison_metric = workflow_config['model_validation']['comparison_metric']

    client = MlflowClient()
    try:
        versions = client.get_latest_versions(registry_name, stages=[production_stage])
    except MlflowException as exc:
        if 'RESOURCE_DOES_NOT_EXIST' in str(exc.error_code):
            return {
                'exists'      : False,
                'value'       : None,
                'dataset_path': None,
                'registry_name': registry_name,
                'stage'       : production_stage,
                'reason': (
                    f"Registered model '{registry_name}' belum ada di MLflow Model "
                    f"Registry (belum pernah ada eksperimen yang didaftarkan)."
                ),
            }
        return {
            'exists'      : False,
            'value'       : None,
            'dataset_path': None,
            'registry_name': registry_name,
            'stage'       : production_stage,
            'reason': f"Gagal mengakses MLflow Model Registry: {exc}",
        }
    except Exception as exc:
        return {
            'exists'      : False,
            'value'       : None,
            'dataset_path': None,
            'registry_name': registry_name,
            'stage'       : production_stage,
            'reason': f"Gagal mengakses MLflow Model Registry: {exc}",
        }

    if not versions:
        return {
            'exists'      : False,
            'value'       : None,
            'dataset_path': None,
            'registry_name': registry_name,
            'stage'       : production_stage,
            'reason': (
                f"Registered model '{registry_name}' ada, tetapi belum ada versi "
                f"berstage '{production_stage}'."
            ),
        }

    prod_run_id = versions[0].run_id
    try:
        prod_run     = client.get_run(prod_run_id)
        prod_metrics = prod_run.data.metrics
        return {
            'exists'      : True,
            'value'       : prod_metrics.get(comparison_metric),
            'dataset_path': prod_run.data.params.get('data.path'),
            'registry_name': registry_name,
            'stage'       : production_stage,
            'version'     : versions[0].version,
            'reason': f"Model produksi ditemukan: '{registry_name}' v{versions[0].version}",
        }
    except Exception as exc:
        return {
            'exists'      : False,
            'value'       : None,
            'dataset_path': None,
            'registry_name': registry_name,
            'stage'       : production_stage,
            'reason': f"Model produksi terdaftar tetapi gagal membaca run metrics: {exc}",
        }


def _check_vs_production(
    metrics: dict,
    production_lookup: dict,
    validation_cfg: dict,
    data: dict,
    model_config: dict,
) -> dict:
    comparison_metric   = validation_cfg['comparison_metric']
    require_improvement = validation_cfg.get('require_improvement', True)
    skip_if_no_prod      = validation_cfg.get('skip_comparison_if_no_production', True)

    new_value = metrics.get(comparison_metric, 0.0)

    if not production_lookup['exists']:
        if skip_if_no_prod:
            return {
                'passed'        : True,
                'reasons'       : [],
                'bootstrap_mode': True,
                'details': {
                    'production_model': None,
                    'new_value'       : new_value,
                    'prod_value'      : None,
                    'note'            : production_lookup['reason'],
                },
            }
        else:
            return {
                'passed'        : False,
                'reasons'       : [
                    f"{production_lookup['reason']} "
                    f"(skip_comparison_if_no_production=False — registrasi model "
                    f"produksi diwajibkan sebelum pipeline dapat lolos validasi)."
                ],
                'bootstrap_mode': True,
                'details': {
                    'production_model': None,
                    'new_value'       : new_value,
                    'prod_value'      : None,
                    'note'            : production_lookup['reason'],
                },
            }

    if not require_improvement:
        return {
            'passed'        : True,
            'reasons'       : [],
            'bootstrap_mode': False,
            'details': {
                'metric'     : comparison_metric,
                'new_value'  : new_value,
                'prod_value' : production_lookup['value'],
                'delta'      : None,
                'reevaluated': False,
            },
        }

    current_dataset_path = model_config.get('data', {}).get('path')
    prod_dataset_path    = production_lookup['dataset_path']
    same_dataset = prod_dataset_path is not None and prod_dataset_path == current_dataset_path

    if same_dataset:
        prod_metric_value = production_lookup['value']
        reevaluated       = False
    else:
        print(
            f"    Dataset model produksi berbeda dari dataset saat ini "
            f"('{prod_dataset_path}' vs '{current_dataset_path}') — mengunduh & "
            f"mengevaluasi ulang '{production_lookup['registry_name']}' "
            f"(stage {production_lookup['stage']}) pada test set saat ini..."
        )
        prod_metrics = _reevaluate_checkpoint(
            production_lookup['registry_name'], production_lookup['stage'], data
        )
        prod_metric_value = prod_metrics.get(comparison_metric, 0.0)
        reevaluated        = True

    passed = new_value > prod_metric_value
    reasons = (
        []
        if passed
        else [
            f"{comparison_metric} model baru ({new_value:.4f}) tidak lebih baik "
            f"dari produksi ({prod_metric_value:.4f})"
            + (" setelah re-evaluasi pada test set yang sama." if reevaluated else ".")
        ]
    )

    return {
        'passed'        : passed,
        'reasons'       : reasons,
        'bootstrap_mode': False,
        'details': {
            'metric'     : comparison_metric,
            'new_value'  : new_value,
            'prod_value' : prod_metric_value,
            'delta'      : new_value - prod_metric_value,
            'reevaluated': reevaluated,
        },
    }



def _check_vs_staging_reeval(
    metrics: dict,
    data: dict,
    workflow_config: dict,
    model_config: dict,
) -> dict:
    mlflow_wf  = workflow_config.get('mlflow', {})
    mlflow_mdl = model_config.get('mlflow', {})

    registry_name     = mlflow_wf.get('registry_name', mlflow_mdl.get('registry_name', 'absa_indobert'))
    register_stage    = mlflow_wf.get('register_stage', 'Staging')
    comparison_metric = workflow_config['model_validation']['comparison_metric']

    client = MlflowClient()
    try:
        versions = client.get_latest_versions(registry_name, stages=[register_stage])
    except MlflowException as exc:
        if 'RESOURCE_DOES_NOT_EXIST' in str(exc.error_code):
            versions = []
        else:
            return {
                'passed'        : False,
                'reasons'       : [f"Gagal mengakses MLflow Model Registry: {exc}"],
                'bootstrap_mode': False,
                'details'       : {},
            }

    if not versions:
        return {
            'passed'        : True,
            'reasons'       : [],
            'bootstrap_mode': True,
            'details': {
                'note': (
                    f"Belum ada versi berstage '{register_stage}' pada "
                    f"registered model '{registry_name}' — mode bootstrap."
                ),
            },
        }

    baseline_version = versions[0].version
    baseline_run     = client.get_run(versions[0].run_id)

    current_dataset_path  = model_config.get('data', {}).get('path')
    baseline_dataset_path = baseline_run.data.params.get('data.path')
    same_dataset = baseline_dataset_path is not None and baseline_dataset_path == current_dataset_path

    if same_dataset:
        print(f"    Dataset baseline '{register_stage}' v{baseline_version} sama dengan "
              f"dataset saat ini — memakai metrik yang sudah tercatat tanpa evaluasi ulang.")
        baseline_value = baseline_run.data.metrics.get(comparison_metric, 0.0)
        reevaluated    = False
    else:
        print(f"    Dataset baseline '{register_stage}' v{baseline_version} berbeda dari "
              f"dataset saat ini ('{baseline_dataset_path}' vs '{current_dataset_path}') — "
              f"mengunduh & mengevaluasi ulang pada test set saat ini...")
        baseline_metrics = _reevaluate_checkpoint(registry_name, register_stage, data)
        baseline_value   = baseline_metrics.get(comparison_metric, 0.0)
        reevaluated      = True

    new_value = metrics.get(comparison_metric, 0.0)
    passed = new_value > baseline_value

    reasons = [] if passed else [
        f"{comparison_metric} model baru ({new_value:.4f}) tidak lebih baik dari "
        f"baseline '{register_stage}' v{baseline_version} ({baseline_value:.4f})"
        + (" setelah re-evaluasi pada test set yang sama." if reevaluated else ".")
    ]

    return {
        'passed'        : passed,
        'reasons'       : reasons,
        'bootstrap_mode': False,
        'details': {
            'baseline_version': baseline_version,
            'metric'          : comparison_metric,
            'new_value'       : new_value,
            'baseline_value'  : baseline_value,
            'delta'           : new_value - baseline_value,
            'reevaluated'     : reevaluated,
        },
    }



def run_validate_model(metrics: dict, data: dict, workflow_config: dict, model_config: dict) -> dict:
    validation_cfg = workflow_config['model_validation']

    print("\n  [Uji 1] Memeriksa threshold metrik absolut...")
    threshold_result = _check_threshold(metrics, validation_cfg)
    status1 = 'LULUS' if threshold_result['passed'] else 'GAGAL'
    print(f"  Uji 1: {status1}")
    for reason in threshold_result['reasons']:
        print(f"    - {reason}")

    print("\n  [Uji 2] Membandingkan dengan model produksi...")
    production_lookup = _get_production_metric(workflow_config, model_config)
    production_result = _check_vs_production(
        metrics, production_lookup, validation_cfg, data, model_config
    )
    status2 = 'LULUS' if production_result['passed'] else 'GAGAL'

    if production_result['bootstrap_mode']:
        print(f"  [MODE BOOTSTRAP] {production_lookup['reason']}")

    prod_value = production_result['details'].get('prod_value')
    prod_label = f"{prod_value:.4f}" if prod_value is not None else "tidak ada"
    print(f"  Uji 2: {status2} (produksi: {prod_label}, "
          f"baru: {metrics.get(validation_cfg['comparison_metric'], 0):.4f})")
    for reason in production_result['reasons']:
        print(f"    - {reason}")

    print("\n  [Uji 3] Membandingkan dengan model Staging saat ini (re-evaluasi)...")
    staging_result = _check_vs_staging_reeval(metrics, data, workflow_config, model_config)
    status3 = 'LULUS' if staging_result['passed'] else 'GAGAL'

    if staging_result['bootstrap_mode']:
        print(f"  [MODE BOOTSTRAP] {staging_result['details'].get('note', '')}")
    print(f"  Uji 3: {status3}")
    for reason in staging_result['reasons']:
        print(f"    - {reason}")

    all_failure_reasons = (
        threshold_result['reasons'] + production_result['reasons'] + staging_result['reasons']
    )
    overall_passed = (
        threshold_result['passed'] and production_result['passed'] and staging_result['passed']
    )

    if overall_passed:
        print("\n  Model LOLOS validasi dan siap didaftarkan.")
    else:
        print(f"\n  Model TIDAK LOLOS validasi ({len(all_failure_reasons)} alasan).")

    return {
        'passed'               : overall_passed,
        'threshold_check'      : threshold_result,
        'production_check'     : production_result,
        'staging_reeval_check' : staging_result,
        'failure_reasons'      : all_failure_reasons,
    }
