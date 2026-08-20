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

        self.next(self.package_model)

    # ── Step 7: Package Model ─────────────────────────────────────────────────
    @step
    def package_model(self):
        """
        Bungkus checkpoint sebagai artifact pyfunc di MLflow run — TANPA
        registrasi/alias. Selalu dijalankan (tidak ada gerbang validasi);
        registrasi ke Model Registry + promosi alias champion/archived
        sekarang murni keputusan manual lewat tombol "Promote" di backend.
        """
        from workflow.stages.package_model import run_package_model

        print(f"\n[6/7] Packaging model ke MLflow...")
        self.package_result = run_package_model(self.train_result, self.model_config)

        self.next(self.compare_champion)

    # ── Step 8: Compare with Champion ─────────────────────────────────────────
    @step
    def compare_champion(self):
        """
        Muat model yang sedang memegang alias MLflow `champion` (kalau ada)
        dan evaluasi ulang pada test split yang sama, supaya kedua metrik
        utamanya (Sentiment F1, Detection F1) bisa ditampilkan berdampingan
        dengan model baru di UI. Murni informasional — tidak ada pass/fail.
        """
        from workflow.stages.compare_champion import run_compare_champion

        print(f"\n[7/7] Membandingkan dengan model champion saat ini...")
        self.champion_result = run_compare_champion(
            self.model_config, self.workflow_config, self.data,
            run_id=self.train_result['run_id'],
        )

        self.next(self.end)

    # ── Step 9: End ───────────────────────────────────────────────────────────
    @step
    def end(self):
        """Cetak ringkasan eksekusi pipeline."""
        ext   = self.extraction_result
        met   = self.metrics
        champ = self.champion_result
        run_id = self.train_result['run_id']

        mlflow_cfg   = self.model_config.get('mlflow', {})
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI') or mlflow_cfg.get(
            'tracking_uri', 'http://localhost:5000',
        )
        # MLFLOW_TRACKING_URI di dalam container metaflow-trigger memakai hostname
        # docker-internal (mis. http://mlflow:5000) — cocok untuk panggilan API
        # antar-container, tapi tidak bisa dibuka browser user. MLFLOW_PUBLIC_URL
        # (opsional) dipakai khusus untuk link yang ditampilkan ke user.
        public_uri = os.environ.get('MLFLOW_PUBLIC_URL') or tracking_uri
        experiment_id = self.train_result.get('experiment_id')
        mlflow_run_url = (
            f"{public_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
            if experiment_id else None
        )

        print(f"\n{'='*60}")
        print(f"ABSA RETRAINING PIPELINE SELESAI")
        print(f"{'='*60}")
        print(f"  Flow run ID         : {current.run_id}")
        print(f"  Dipicu oleh         : {self.trigger_reason}")
        print(f"  Data baru           : {ext['n_new_samples']} sampel")
        print(f"  Data cukup          : {ext['data_sufficient']}")
        print(f"  Test Sentiment F1   : {met.get('test_mean_sentiment_f1', 0):.4f}")
        print(f"  Test Detection F1   : {met.get('test_mean_detect_f1', 0):.4f}")
        if champ['champion_exists'] and champ['metrics']:
            print(f"  Champion ada        : True (versi {champ['champion_version']})")
            print(f"  Champion Sentiment F1: {champ['metrics']['test_mean_sentiment_f1']:.4f}")
            print(f"  Champion Detection F1: {champ['metrics']['test_mean_detect_f1']:.4f}")
        else:
            print(f"  Champion ada        : False ({champ['reason']})")
        print(f"  MLflow Run ID       : {run_id}")
        print(f"  MLflow Run URL      : {mlflow_run_url}")
        print(f"{'='*60}")


if __name__ == '__main__':
    ABSARetrainingFlow()
