"""
Unit Test Komponen Automated ML Workflow Pipeline
==================================================
Tahap CI pertama (lihat .github/workflows/ci-cd.yaml): setiap komponen
pipeline — mulai dari ekstraksi data hingga registrasi model — diuji dengan
definisi masukan dan keluaran sesuai target penanganan masing-masing komponen.

Komponen berat (training, MLflow server, monitoring API) di-mock sehingga
unit test hanya menguji kontrak input/output tiap komponen, cepat, dan
tidak butuh layanan eksternal.

Jalankan: pytest tests/
"""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from preprocessing.preprocessing_functions import FINAL_ASPECTS


# ── Helper: dataset dummy dengan skema kolom valid ─────────────────────────────

def _make_dataset(n_rows: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({'Komentar': [f'komentar uji {i}' for i in range(n_rows)]})
    for asp in FINAL_ASPECTS:
        df[asp] = [(-1, 0, 1)[i % 3] for i in range(n_rows)]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Ekstraksi Data (workflow/stages/extract_data.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractData:

    def test_fetch_annotated_data_since_returns_dataframe(self):
        """Input: timestamp + URL API → Output: DataFrame berisi records dari API."""
        from workflow.stages.extract_data import fetch_annotated_data_since

        fake_response = MagicMock()
        fake_response.json.return_value = {
            'data': [{'Komentar': 'bagus', 'Content Quality': 1}], 'count': 1,
        }
        with patch('workflow.stages.extract_data.requests.post',
                   return_value=fake_response) as mock_post:
            df = fetch_annotated_data_since('2025-01-01T00:00:00Z', 'http://monitoring:8001')

        mock_post.assert_called_once_with(
            'http://monitoring:8001/annotations/export',
            json={'since': '2025-01-01T00:00:00Z'}, timeout=30,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'Komentar' in df.columns

    def test_timestamp_state_roundtrip(self, tmp_path):
        """save_current_timestamp menulis ISO 8601 yang bisa dibaca kembali load_last_timestamp."""
        from workflow.stages.extract_data import load_last_timestamp, save_current_timestamp

        state_file = str(tmp_path / '.state' / 'last_ts')
        saved = save_current_timestamp(state_file)
        assert load_last_timestamp(state_file) == saved
        assert saved.endswith('Z')

    def test_check_data_sufficiency(self, tmp_path):
        """Kriteria OR: cukup via jumlah absolut ATAU rasio; selain itu belum cukup."""
        from workflow.stages.extract_data import check_data_sufficiency

        existing = tmp_path / 'existing.csv'
        _make_dataset(100).to_csv(existing, index=False)

        by_count = check_data_sufficiency(_make_dataset(100), str(existing),
                                          min_new_samples=100, min_ratio=10.0)
        assert by_count['sufficient'] and by_count['n_new'] == 100

        by_ratio = check_data_sufficiency(_make_dataset(10), str(existing),
                                          min_new_samples=1000, min_ratio=0.05)
        assert by_ratio['sufficient'] and by_ratio['ratio'] == pytest.approx(0.1)

        neither = check_data_sufficiency(_make_dataset(2), str(existing),
                                         min_new_samples=100, min_ratio=0.05)
        assert not neither['sufficient']

    def test_run_extract_data_disabled_returns_existing_dataset(self):
        """enabled=false → dataset yang ada dipakai apa adanya, tanpa panggilan API."""
        from workflow.stages.extract_data import run_extract_data

        workflow_config = {
            'data_extraction' : {'enabled': False},
            'data_sufficiency': {'min_new_samples': 100, 'min_ratio_of_existing': 0.05},
            'notification'    : {'enable_notifications': False},
        }
        model_config = {'data': {'path': 'data/raw/dataset.csv'}}

        result = run_extract_data(workflow_config, model_config)

        assert result == {
            'dataset_path'   : 'data/raw/dataset.csv',
            'n_new_samples'  : 0,
            'data_sufficient': False,
            'extraction_ts'  : None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. Validasi Data (workflow/stages/validate_data.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateData:

    def test_valid_dataset_passes(self, tmp_path):
        """Input: config dengan dataset valid → Output: laporan passed=True."""
        from workflow.stages.validate_data import run_validate_data

        path = tmp_path / 'valid.csv'
        _make_dataset(10).to_csv(path, index=False)

        report = run_validate_data({'data': {'path': str(path), 'text_column': 'Komentar'}})
        assert report['passed'] is True
        assert report['total_rows'] == 10
        assert report['issues'] == []

    def test_missing_label_column_raises(self, tmp_path):
        """Kolom label hilang → RuntimeError menghentikan pipeline sebelum training."""
        from workflow.stages.validate_data import run_validate_data

        df = _make_dataset(10).drop(columns=[FINAL_ASPECTS[0]])
        path = tmp_path / 'invalid.csv'
        df.to_csv(path, index=False)

        with pytest.raises(RuntimeError, match='Validasi data gagal'):
            run_validate_data({'data': {'path': str(path), 'text_column': 'Komentar'}})


# ══════════════════════════════════════════════════════════════════════════════
# 3. Persiapan Data (workflow/stages/prepare_data.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestPrepareData:

    def test_returns_three_splits_and_class_weights(self):
        """Input: model config → Output: dict berisi df_train/df_val/df_test/class_weights."""
        from workflow.stages import prepare_data as stage

        fake = {
            'df_train'     : _make_dataset(7),
            'df_val'       : _make_dataset(2),
            'df_test'      : _make_dataset(1),
            'class_weights': {asp: [1.0] for asp in FINAL_ASPECTS},
        }
        with patch.object(stage, 'prepare_data', return_value=fake) as mock_prep:
            result = stage.run_prepare_data({'data': {}})

        mock_prep.assert_called_once_with({'data': {}})
        assert set(result) == {'df_train', 'df_val', 'df_test', 'class_weights'}


# ══════════════════════════════════════════════════════════════════════════════
# 4. Pelatihan Model (workflow/stages/train_model.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainModel:

    def test_returns_serializable_metadata(self, tmp_path):
        """Input: config + data → Output: metadata ringan (run_id, save_dir, metrik val)."""
        from workflow.stages import train_model as stage

        save_dir = tmp_path / 'model_output'
        save_dir.mkdir()
        (save_dir / 'config.yaml').write_text('x: 1')

        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = 'run123'
        trained = {'best_val_f1': 0.81, 'best_val_det_f1': 0.90, 'save_dir': str(save_dir)}

        model_config = {
            'experiment'    : {'name': 'test', 'seed': 42},
            'representation': {'model_name': 'indobenchmark/indobert-base-p1'},
        }
        with patch.object(stage, 'mlflow', mock_mlflow), \
             patch.object(stage, 'train_model', return_value=trained) as mock_train, \
             patch.object(stage, 'set_seed'):
            result = stage.run_train_model(model_config, {'df_train': None})

        mock_train.assert_called_once()
        assert result == {
            'run_id'         : 'run123',
            'save_dir'       : str(save_dir),
            'best_val_f1'    : 0.81,
            'best_val_det_f1': 0.90,
        }
        # Artefak kecil (config.yaml) di-log; checkpoint .pt tidak
        mock_mlflow.log_artifact.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Evaluasi Model (workflow/stages/evaluate_model.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluateModel:

    def test_returns_test_metrics_and_logs_to_same_run(self, tmp_path):
        """Input: config + hasil train + data → Output: metrik test set, di-log ke run yang sama."""
        from workflow.stages import evaluate_model as stage

        metrics = {'test_mean_sentiment_f1': 0.75, 'test_mean_detect_f1': 0.88}
        mock_mlflow = MagicMock()
        train_result = {'run_id': 'run123', 'save_dir': str(tmp_path)}

        with patch.object(stage, 'mlflow', mock_mlflow), \
             patch.object(stage, 'load_model_from_checkpoint',
                          return_value=(MagicMock(), MagicMock(), {})), \
             patch.object(stage, 'evaluate_model', return_value=metrics):
            result = stage.run_evaluate_model({'mlflow': {}}, train_result, {'df_test': None})

        assert result == metrics
        mock_mlflow.start_run.assert_called_once_with(run_id='run123')
        mock_mlflow.log_metrics.assert_called_once_with(metrics)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Validasi Model (workflow/stages/validate_model.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidateModel:

    VALIDATION_CFG = {
        'min_sentiment_f1'   : 0.5,
        'min_detection_f1'   : 0.5,
        'comparison_metric'  : 'test_mean_sentiment_f1',
        'require_improvement': True,
        'skip_comparison_if_no_production': True,
    }

    def test_threshold_pass_and_fail(self):
        """Uji 1: metrik >= threshold lulus; di bawah threshold gagal dengan alasan."""
        from workflow.stages.validate_model import _check_threshold

        ok = _check_threshold(
            {'test_mean_sentiment_f1': 0.7, 'test_mean_detect_f1': 0.8},
            self.VALIDATION_CFG,
        )
        assert ok['passed'] and ok['reasons'] == []

        bad = _check_threshold(
            {'test_mean_sentiment_f1': 0.3, 'test_mean_detect_f1': 0.8},
            self.VALIDATION_CFG,
        )
        assert not bad['passed']
        assert any('Sentiment F1' in r for r in bad['reasons'])

    def test_comparison_bootstrap_mode(self):
        """Uji 2: belum ada model produksi → lulus otomatis jika skip=True, gagal jika False."""
        from workflow.stages.validate_model import _check_vs_production

        no_prod = {'exists': False, 'value': None, 'reason': 'belum ada model produksi'}
        metrics = {'test_mean_sentiment_f1': 0.7}

        assert _check_vs_production(metrics, no_prod, self.VALIDATION_CFG)['passed']

        strict_cfg = dict(self.VALIDATION_CFG, skip_comparison_if_no_production=False)
        assert not _check_vs_production(metrics, no_prod, strict_cfg)['passed']

    def test_comparison_vs_production_metric(self):
        """Uji 2: model baru harus lebih baik dari metrik model produksi."""
        from workflow.stages.validate_model import _check_vs_production

        prod = {'exists': True, 'value': 0.6, 'reason': 'produksi v1'}

        better = _check_vs_production({'test_mean_sentiment_f1': 0.7}, prod, self.VALIDATION_CFG)
        assert better['passed'] and better['details']['delta'] == pytest.approx(0.1)

        worse = _check_vs_production({'test_mean_sentiment_f1': 0.5}, prod, self.VALIDATION_CFG)
        assert not worse['passed']


# ══════════════════════════════════════════════════════════════════════════════
# 7. Registrasi Model (workflow/stages/register_model.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestRegisterModel:

    WORKFLOW_CFG = {'mlflow': {'tracking_uri' : 'sqlite:///test.db',
                               'registry_name': 'absa_test',
                               'register_stage': 'Staging'}}

    def test_failed_validation_skips_registration(self):
        """Model tidak lolos validasi → tidak didaftarkan, registry tidak disentuh."""
        from workflow.stages.register_model import run_register_model

        result = run_register_model(
            model_validation = {'passed': False, 'failure_reasons': ['F1 di bawah threshold']},
            train_result     = {'run_id': 'run123', 'save_dir': '/tmp/x'},
            metrics          = {},
            workflow_config  = self.WORKFLOW_CFG,
            model_config     = {},
        )
        assert result['registered'] is False
        assert result['model_version'] is None
        assert 'tidak lolos validasi' in result['reason']

    def test_passed_validation_registers_to_staging(self, tmp_path):
        """Model lolos validasi → didaftarkan ke registry dan ditransisikan ke Staging."""
        from workflow.stages import register_model as stage

        mock_mlflow  = MagicMock()
        mock_mlflow.register_model.return_value.version = 3
        mock_client  = MagicMock()
        metrics      = {'test_mean_sentiment_f1': 0.7, 'test_mean_detect_f1': 0.8}

        with patch.object(stage, 'mlflow', mock_mlflow), \
             patch.object(stage, 'MlflowClient', return_value=mock_client):
            result = stage.run_register_model(
                model_validation = {'passed': True, 'failure_reasons': []},
                train_result     = {'run_id': 'run123', 'save_dir': str(tmp_path)},
                metrics          = metrics,
                workflow_config  = self.WORKFLOW_CFG,
                model_config     = {},
            )

        assert result['registered'] is True
        assert result['model_version'] == 3
        assert result['model_stage'] == 'Staging'
        # Metrik test set ikut tersimpan dalam bundle artifact
        assert (tmp_path / 'metrics.json').is_file()
        mock_mlflow.register_model.assert_called_once_with(
            model_uri='runs:/run123/model', name='absa_test',
        )
        mock_client.transition_model_version_stage.assert_called_once_with(
            name='absa_test', version=3, stage='Staging',
            archive_existing_versions=True,
        )
