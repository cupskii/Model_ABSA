"""
Mekanisme Trigger
-----------------
  1. Terjadwal (scheduled)  → Metaflow Scheduler berdasarkan @schedule
  2. Manual / ad-hoc        → python workflow/flow.py run [--param ...]
  3. Eksternal (monitoring) → POST http://localhost:8002/trigger
                              (lihat workflow/trigger.py)
"""

import os
import sys
import yaml

# Pastikan root repo ada di sys.path sehingga import pipeline.* dan model.* berjalan
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from metaflow import FlowSpec, step, Parameter, current

# @schedule mengaktifkan penjadwalan otomatis via Metaflow Scheduler.
# Untuk penggunaan lokal tanpa Metaflow Service, decorator ini tidak berpengaruh.
try:
    from metaflow import schedule
    @schedule(cron='0 0 1 * *', timezone='UTC')
    class ABSARetrainingFlow(FlowSpec):
        pass
    # Reset class; definisi nyata di bawah
    del ABSARetrainingFlow
    _schedule_decorator = schedule(cron='0 0 1 * *', timezone='UTC')
except ImportError:
    _schedule_decorator = lambda cls: cls  # noqa: E731


@_schedule_decorator
class ABSARetrainingFlow(FlowSpec):
    config_path = Parameter(
        'config_path',
        help='Path ke file konfigurasi model YAML (relatif terhadap root repo)',
        default='configs/experiment_indobert_baseline.yaml',
    )
    workflow_config_path = Parameter(
        'workflow_config_path',
        help='Path ke file konfigurasi workflow YAML',
        default='workflow/pipeline_config.yaml',
    )
    trigger_reason = Parameter(
        'trigger_reason',
        help='Alasan pemicu eksekusi: scheduled | manual | monitoring_alert',
        default='manual',
    )

    # ── Step 1: Start ─────────────────────────────────────────────────────────
    @step
    def start(self):
        """Muat kedua file konfigurasi dan catat metadata eksekusi pipeline."""
        config_path = os.path.join(_REPO_ROOT, self.config_path)
        wf_config_path = os.path.join(_REPO_ROOT, self.workflow_config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)
        with open(wf_config_path, 'r', encoding='utf-8') as f:
            self.workflow_config = yaml.safe_load(f)

        # Ekspor alasan trigger ke env var agar dapat dicatat oleh tahap train
        os.environ['ABSA_TRIGGER_REASON'] = self.trigger_reason

        print(f"{'='*60}")
        print(f"ABSA RETRAINING PIPELINE DIMULAI")
        print(f"  Flow run ID    : {current.run_id}")
        print(f"  Config model   : {self.config_path}")
        print(f"  Config workflow: {self.workflow_config_path}")
        print(f"  Dipicu oleh    : {self.trigger_reason}")
        print(f"{'='*60}")

        self.next(self.extract_data)

    # ── Step 2: Extract Data ──────────────────────────────────────────────────
    @step
    def extract_data(self):
        """
        Ambil data produksi ter-anotasi dari komponen monitoring,
        periksa kecukupan volume, dan bentuk dataset pelatihan terbaru.
        """
        from workflow.stages.extract_data import run_extract_data

        print(f"\n[1/7] Ekstraksi data produksi...")
        self.extraction_result = run_extract_data(self.workflow_config, self.model_config)

        # Perbarui path dataset di model config jika ada versi baru
        self.model_config['data']['path'] = self.extraction_result['dataset_path']
        # self.model_config['data']['path'] = self.config_path["data"]["path"]

        self.next(self.validate_data)

  
    @step
    def validate_data(self):
        """Validasi integritas dataset sebelum proses pelatihan."""
        from workflow.stages.validate_data import run_validate_data

        print(f"\n[2/7] Validasi data...")
        self.validation_report = run_validate_data(self.model_config)

        self.next(self.prepare_data)


    @step
    def prepare_data(self):
        """
        Terapkan prapemrosesan teks, bagi dataset menjadi split
        train/val/test, dan hitung class weights.
        """
        from workflow.stages.prepare_data import run_prepare_data

        print(f"\n[3/7] Persiapan data...")
        self.data = run_prepare_data(self.model_config)

        self.next(self.train_step)

    # ── Step 5: Train Model ───────────────────────────────────────────────────
    # @conda mengisolasi dependensi heavy ML dalam virtual environment terpisah.
    # Metaflow membuat environment ini secara otomatis tanpa Dockerfile manual.
    #
    # Aktifkan dengan: python workflow/flow.py --environment=conda run
    #
    # @conda(
    #     libraries={
    #         'pytorch'              : '>=2.0.0',
    #         'transformers'         : '5.12.0',
    #         'numpy'                : '2.5.0',
    #         'pandas'               : '2.3.3',
    #         'scikit-learn'         : '1.9.0',
    #         'matplotlib'           : '3.11.0',
    #         'mlflow'               : '3.14.0',
    #         'iterative-stratification': '0.1.9',
    #     },
    # )
    @step
    def train_step(self):
        """
        Latih model IndoBERT dengan konfigurasi terkini dan catat
        eksperimen ke MLflow. Menyimpan checkpoint ke disk dan mengembalikan
        metadata ringan (tanpa objek model) sebagai Metaflow artifact.
        """
        from workflow.stages.train_model import run_train_model

        print(f"\n[4/7] Pelatihan model...")
        self.train_result = run_train_model(self.model_config, self.data)

        self.next(self.evaluate_step)

    # ── Step 6: Evaluate Model ────────────────────────────────────────────────
    @step
    def evaluate_step(self):
        """
        Muat ulang model dari checkpoint disk, evaluasi pada test set,
        dan tambahkan metrik ke MLflow run yang dibuka oleh train_step.
        """
        from workflow.stages.evaluate_model import run_evaluate_model

        print(f"\n[5/7] Evaluasi model pada test set...")
        self.metrics = run_evaluate_model(
            self.model_config, self.train_result, self.data
        )

        self.next(self.validate_model)

    # ── Step 7: Validate Model ────────────────────────────────────────────────
    @step
    def validate_model(self):
        """
        Validasi apakah model baru memenuhi threshold minimum dan lebih
        baik dari model yang sedang berjalan di produksi.
        """
        from workflow.stages.validate_model import run_validate_model

        print(f"\n[6/7] Validasi model...")
        self.model_validation = run_validate_model(
            self.metrics, self.data, self.workflow_config, self.model_config
        )

        self.next(self.register_model)

    # ── Step 8: Register Model ────────────────────────────────────────────────
    @step
    def register_model(self):
        """
        Daftarkan model yang sudah divalidasi ke MLflow Model Registry
        dengan stage 'Staging'. Dilewati (tidak mendaftarkan) jika model
        tidak lolos validasi.
        """
        from workflow.stages.register_model import run_register_model

        print(f"\n[7/7] Registrasi model...")
        self.registry_result = run_register_model(
            model_validation = self.model_validation,
            train_result     = self.train_result,
            metrics          = self.metrics,
            workflow_config  = self.workflow_config,
            model_config     = self.model_config,
        )

        self.next(self.end)

    # ── Step 9: End ───────────────────────────────────────────────────────────
    @step
    def end(self):
        """Cetak ringkasan eksekusi pipeline."""
        ext = self.extraction_result
        met = self.metrics
        val = self.model_validation
        reg = self.registry_result

        print(f"\n{'='*60}")
        print(f"ABSA RETRAINING PIPELINE SELESAI")
        print(f"{'='*60}")
        print(f"  Flow run ID         : {current.run_id}")
        print(f"  Dipicu oleh         : {self.trigger_reason}")
        print(f"  Data baru           : {ext['n_new_samples']} sampel")
        print(f"  Data cukup          : {ext['data_sufficient']}")
        print(f"  Test Sentiment F1   : {met.get('test_mean_sentiment_f1', 0):.4f}")
        print(f"  Test Detection F1   : {met.get('test_mean_detect_f1', 0):.4f}")
        print(f"  Lolos validasi      : {val['passed']}")
        if not val['passed']:
            for reason in val['failure_reasons']:
                print(f"    ✗ {reason}")
        print(f"  Model terdaftar     : {reg['registered']}")
        if reg['registered']:
            print(f"    → {reg['reason']}")
        print(f"  MLflow run ID       : {self.train_result['run_id'][:12]}")
        print(f"{'='*60}")


if __name__ == '__main__':
    ABSARetrainingFlow()
