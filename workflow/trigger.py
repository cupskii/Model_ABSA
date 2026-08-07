"""
Entry Point Trigger Eksternal — ABSA Retraining Pipeline
==========================================================
Menyediakan dua mekanisme pemicu selain penjadwalan otomatis:

  1. Manual / ad-hoc via CLI:
       python workflow/trigger.py
       python workflow/trigger.py --config_path configs/experiment_indobert_baseline.yaml
       python workflow/trigger.py --reason "manual_retrain"

  2. Eksternal via HTTP (dari backend BE_ABSA saat user meng-upload dataset
     baru dan klik "retraining", atau dari komponen monitoring saat
     mendeteksi drift/degradasi):
       python workflow/trigger.py --serve
       # Lalu POST http://localhost:8002/trigger (multipart/form-data)

     Form fields POST /trigger:
       file                  : dataset CSV baru (wajib)
       reason                : 'manual' | 'monitoring_alert' | 'scheduled' (default 'manual')
       config_path           : path konfigurasi model YAML dasar yang akan
                                di-override data.path-nya (opsional)
       workflow_config_path  : path konfigurasi workflow YAML (opsional)

     Karena tidak ada Metaflow UI/metadata-service yang di-deploy, progres
     tiap step dan log mentah pipeline diekspos lewat dua endpoint tambahan
     yang membaca file log per-run (bukan lewat Metaflow Client API):
       GET /status/{run_id}  → step mana yang sedang berjalan + metrik akhir
       GET /logs/{run_id}    → tail baris log mentah proses flow.py

Dengan adanya satu titik masuk ini, satu pipeline yang sama dapat dipicu
oleh tiga mekanisme (terjadwal, manual, monitoring) tanpa duplikasi
implementasi pipeline itu sendiri.
"""

import os
import re
import sys
import time
import uuid
import subprocess
import argparse

import yaml

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_FLOW_PATH = os.path.join(os.path.dirname(__file__), 'flow.py')

_UPLOAD_DIR = os.path.join(_REPO_ROOT, 'data', 'retrain_uploads')
_GENERATED_CONFIG_DIR = os.path.join(_REPO_ROOT, 'configs', '_generated')
_LOG_DIR = os.path.join(os.path.dirname(__file__), '.logs')

# Urutan step persis meniru marker "[k/7] ..." yang sudah dicetak flow.py —
# perubahan di sini HARUS disinkronkan manual dengan print() di flow.py,
# karena parsing status di bawah murni berbasis pencocokan teks log (tidak
# ada Metaflow Client API/metadata-service yang bisa dipakai untuk sumber
# kebenaran step yang sedang berjalan).
_STEP_DEFS = [
    ('extract_data',   'Ekstraksi Data',     '[1/7]'),
    ('validate_data',  'Validasi Data',      '[2/7]'),
    ('prepare_data',      'Persiapan Data',            '[3/7]'),
    ('train_step',        'Pelatihan Model',           '[4/7]'),
    ('evaluate_step',     'Evaluasi Model',            '[5/7]'),
    ('package_model',     'Packaging Model',           '[6/7]'),
    ('compare_champion',  'Perbandingan Champion',     '[7/7]'),
]

_METRIC_PATTERNS = {
    'test_sentiment_f1': re.compile(r'Test Sentiment F1\s*:\s*([\d.]+)'),
    'test_detection_f1': re.compile(r'Test Detection F1\s*:\s*([\d.]+)'),
    'champion_sentiment_f1': re.compile(r'Champion Sentiment F1\s*:\s*([\d.]+)'),
    'champion_detection_f1': re.compile(r'Champion Detection F1\s*:\s*([\d.]+)'),
}
_CHAMPION_EXISTS_PATTERN = re.compile(r'Champion ada\s*:\s*(True|False)')
_CHAMPION_VERSION_PATTERN = re.compile(r'Champion ada\s*:\s*True \(versi (\S+)\)')
_MLFLOW_RUN_ID_PATTERN = re.compile(r'MLflow Run ID\s*:\s*(\S+)')
_MLFLOW_RUN_URL_PATTERN = re.compile(r'MLflow Run URL\s*:\s*(\S+)')

# Registry proses aktif/selesai — in-memory saja. Jika service ini restart,
# run yang sedang berjalan kehilangan tracking-nya (log & config file di
# disk tetap ada); dapat diterima untuk alur manual/low-frequency ini.
_RUNS: dict[str, dict] = {}


# ── Fungsi Pemicu Inti ────────────────────────────────────────────────────────

def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def trigger_pipeline(
    config_path: str           = 'configs/experiment_indobert_baseline.yaml',
    workflow_config_path: str  = 'workflow/pipeline_config.yaml',
    reason: str                = 'manual',
    wait: bool                 = False,
) -> dict:
    """
    Picu eksekusi ABSA Retraining Pipeline (jalur CLI — tanpa dataset baru,
    memakai dataset yang sudah tertulis di config_path apa adanya).
    """
    cmd = [
        sys.executable, _FLOW_PATH, 'run',
        '--config_path',          config_path,
        '--workflow_config_path', workflow_config_path,
        '--trigger_reason',       reason,
    ]

    print(f"Memicu ABSA Retraining Pipeline...")
    print(f"  Alasan  : {reason}")
    print(f"  Config  : {config_path}")
    print(f"  Perintah: {' '.join(cmd)}")

    if wait:
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        return {
            'status'     : 'completed' if result.returncode == 0 else 'failed',
            'returncode' : result.returncode,
            'reason'     : reason,
        }
    else:
        proc = subprocess.Popen(cmd, cwd=_REPO_ROOT)
        print(f"  Pipeline berjalan di background (PID: {proc.pid})")
        return {
            'status': 'triggered',
            'pid'   : proc.pid,
            'reason': reason,
        }


def trigger_pipeline_with_dataset(
    dataset_bytes: bytes,
    dataset_filename: str,
    reason: str                = 'manual',
    config_path: str           = 'configs/experiment_indobert_baseline.yaml',
    workflow_config_path: str  = 'workflow/pipeline_config.yaml',
) -> dict:
    """
    Picu eksekusi pipeline dengan dataset baru yang dikirim langsung sebagai
    bytes (dari backend BE_ABSA lewat HTTP, lihat create_app()::/trigger).

    Dataset ditulis ke disk milik service ini sendiri, lalu config model
    dasar (config_path) di-copy dengan field data.path di-override ke file
    tersebut dan ditulis sebagai config baru — flow.py sama sekali tidak
    diubah, cukup diberi --config_path yang berbeda.
    """
    run_id = uuid.uuid4().hex[:12]

    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    os.makedirs(_GENERATED_CONFIG_DIR, exist_ok=True)
    os.makedirs(_LOG_DIR, exist_ok=True)

    safe_filename = os.path.basename(dataset_filename) or 'dataset.csv'
    dataset_path = os.path.join(_UPLOAD_DIR, f'{run_id}_{safe_filename}')
    with open(dataset_path, 'wb') as f:
        f.write(dataset_bytes)

    model_config = _load_yaml(_resolve_path(config_path))
    model_config['data']['path'] = dataset_path
    generated_config_path = os.path.join(_GENERATED_CONFIG_DIR, f'{run_id}.yaml')
    with open(generated_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(model_config, f, allow_unicode=True)

    log_path = os.path.join(_LOG_DIR, f'{run_id}.log')
    cmd = [
        sys.executable, _FLOW_PATH, 'run',
        '--config_path',          generated_config_path,
        '--workflow_config_path', _resolve_path(workflow_config_path),
        '--trigger_reason',       reason,
    ]

    log_fh = open(log_path, 'w', encoding='utf-8')
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT)

    _RUNS[run_id] = {
        'pid':         proc.pid,
        'process':     proc,
        'log_path':    log_path,
        'log_fh':      log_fh,
        'reason':      reason,
        'started_at':  time.time(),
        'dataset_path': dataset_path,
        'config_path': generated_config_path,
    }

    print(f"Memicu ABSA Retraining Pipeline (run_id={run_id})...")
    print(f"  Alasan  : {reason}")
    print(f"  Dataset : {dataset_path}")
    print(f"  Config  : {generated_config_path}")
    print(f"  Pipeline berjalan di background (PID: {proc.pid})")

    return {
        'run_id': run_id,
        'status': 'triggered',
        'pid':    proc.pid,
        'reason': reason,
    }


def _read_log_lines(log_path: str) -> list[str]:
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()
    except FileNotFoundError:
        return []


def get_run_status(run_id: str) -> dict | None:
    """
    Bangun status step-by-step murni dari (a) apakah proses masih hidup dan
    (b) marker teks "[k/7] ..." yang sudah dicetak flow.py di log — tanpa
    Metaflow Client API/metadata-service (lihat catatan modul di atas).
    """
    rec = _RUNS.get(run_id)
    if rec is None:
        return None

    proc = rec['process']
    returncode = proc.poll()
    lines = _read_log_lines(rec['log_path'])
    log_text = ''.join(lines)

    reached_idx = -1
    for i, (_key, _label, marker) in enumerate(_STEP_DEFS):
        if marker in log_text:
            reached_idx = i

    if returncode is None:
        overall_status = 'running'
    elif returncode == 0:
        overall_status = 'completed'
    else:
        overall_status = 'failed'

    steps = []
    for i, (key, label, _marker) in enumerate(_STEP_DEFS):
        if overall_status == 'completed':
            status = 'completed'
        elif i < reached_idx:
            status = 'completed'
        elif i == reached_idx:
            status = 'failed' if overall_status == 'failed' else 'processing'
        else:
            status = 'pending'
        steps.append({'step': key, 'label': label, 'status': status})

    current_step = _STEP_DEFS[reached_idx][0] if reached_idx >= 0 else None

    metrics = None
    champion = None
    mlflow_run_id = None
    mlflow_run_url = None
    if overall_status == 'completed':
        metrics = {}
        for name in ('test_sentiment_f1', 'test_detection_f1'):
            m = _METRIC_PATTERNS[name].search(log_text)
            if m:
                metrics[name] = float(m.group(1))

        exists_match = _CHAMPION_EXISTS_PATTERN.search(log_text)
        champion_exists = exists_match.group(1) == 'True' if exists_match else False
        champion = {'exists': champion_exists, 'version': None, 'sentiment_f1': None, 'detection_f1': None}
        if champion_exists:
            version_match = _CHAMPION_VERSION_PATTERN.search(log_text)
            if version_match:
                champion['version'] = version_match.group(1)
            sent_match = _METRIC_PATTERNS['champion_sentiment_f1'].search(log_text)
            if sent_match:
                champion['sentiment_f1'] = float(sent_match.group(1))
            det_match = _METRIC_PATTERNS['champion_detection_f1'].search(log_text)
            if det_match:
                champion['detection_f1'] = float(det_match.group(1))

        run_id_match = _MLFLOW_RUN_ID_PATTERN.search(log_text)
        if run_id_match:
            mlflow_run_id = run_id_match.group(1)
        url_match = _MLFLOW_RUN_URL_PATTERN.search(log_text)
        if url_match and url_match.group(1) != 'None':
            mlflow_run_url = url_match.group(1)

    if returncode is not None:
        try:
            rec['log_fh'].close()
        except Exception:
            pass

    return {
        'run_id':         run_id,
        'overall_status': overall_status,
        'current_step':   current_step,
        'steps':          steps,
        'metrics':        metrics,
        'champion':       champion,
        'mlflow_run_id':  mlflow_run_id,
        'mlflow_run_url': mlflow_run_url,
        'exit_code':      returncode,
    }


def get_run_logs(run_id: str, tail: int = 300) -> dict | None:
    rec = _RUNS.get(run_id)
    if rec is None:
        return None
    lines = [l.rstrip('\n') for l in _read_log_lines(rec['log_path'])]
    finished = rec['process'].poll() is not None
    return {
        'run_id':   run_id,
        'lines':    lines[-tail:] if tail else lines,
        'finished': finished,
    }


# ── HTTP Server (untuk trigger eksternal dari BE_ABSA / komponen monitoring) ───

def create_app():
    """Buat FastAPI app untuk menerima trigger dari komponen eksternal."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from pydantic import BaseModel

    app = FastAPI(
        title       = 'ABSA Pipeline Trigger API',
        description = 'Endpoint untuk memicu ABSA Retraining Pipeline secara eksternal',
        version     = '2.0.0',
    )

    class TriggerResponse(BaseModel):
        run_id : str
        status : str
        pid    : int | None = None
        reason : str
        message: str

    class StepStatus(BaseModel):
        step   : str
        label  : str
        status : str

    class ChampionStatus(BaseModel):
        exists       : bool
        version      : str | None = None
        sentiment_f1 : float | None = None
        detection_f1 : float | None = None

    class RunStatusResponse(BaseModel):
        run_id         : str
        overall_status : str
        current_step   : str | None = None
        steps          : list[StepStatus]
        metrics        : dict | None = None
        champion       : ChampionStatus | None = None
        mlflow_run_id  : str | None = None
        mlflow_run_url : str | None = None
        exit_code      : int | None = None

    class RunLogsResponse(BaseModel):
        run_id   : str
        lines    : list[str]
        finished : bool

    @app.get('/health')
    def health_check():
        """Periksa apakah trigger service berjalan."""
        return {'status': 'ok', 'service': 'ABSA Pipeline Trigger'}

    @app.post('/trigger', response_model=TriggerResponse)
    async def trigger(
        file: UploadFile                = File(...),
        reason: str                     = Form('manual'),
        config_path: str                = Form('configs/experiment_indobert_baseline.yaml'),
        workflow_config_path: str        = Form('workflow/pipeline_config.yaml'),
    ):
        dataset_bytes = await file.read()
        result = trigger_pipeline_with_dataset(
            dataset_bytes         = dataset_bytes,
            dataset_filename       = file.filename or 'dataset.csv',
            reason                 = reason,
            config_path            = config_path,
            workflow_config_path   = workflow_config_path,
        )
        return TriggerResponse(
            run_id  = result['run_id'],
            status  = result['status'],
            pid     = result.get('pid'),
            reason  = result['reason'],
            message = f"Pipeline dipicu dengan alasan: {result['reason']} (run_id: {result['run_id']})",
        )

    @app.get('/status/{run_id}', response_model=RunStatusResponse)
    def run_status(run_id: str):
        result = get_run_status(run_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' tidak ditemukan")
        return result

    @app.get('/logs/{run_id}', response_model=RunLogsResponse)
    def run_logs(run_id: str, tail: int = 300):
        result = get_run_logs(run_id, tail=tail)
        if result is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' tidak ditemukan")
        return result

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pemicu ABSA Retraining Pipeline (manual atau HTTP server)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Trigger manual langsung (tunggu selesai):
  python workflow/trigger.py --wait

  # Trigger manual di background:
  python workflow/trigger.py --reason "manual_retrain"

  # Jalankan HTTP server untuk trigger eksternal:
  python workflow/trigger.py --serve --port 8002

  # Trigger dari backend (kirim dataset baru):
  curl -X POST http://localhost:8002/trigger \\
       -F "file=@dataset_baru.csv" \\
       -F "reason=manual"
        """,
    )
    parser.add_argument(
        '--config_path',
        default='configs/experiment_indobert_baseline.yaml',
        help='Path konfigurasi model YAML (relatif terhadap root repo)',
    )
    parser.add_argument(
        '--workflow_config_path',
        default='workflow/pipeline_config.yaml',
        help='Path konfigurasi workflow YAML',
    )
    parser.add_argument(
        '--reason',
        default='manual',
        choices=['manual', 'monitoring_alert', 'scheduled'],
        help='Alasan pemicu eksekusi pipeline',
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Tunggu pipeline selesai (blocking). Default: fire-and-forget.',
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Jalankan HTTP server untuk menerima trigger eksternal',
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host HTTP server (default: 0.0.0.0)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8002,
        help='Port HTTP server (default: 8002)',
    )
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        app = create_app()
        print(f"Memulai ABSA Pipeline Trigger API di http://{args.host}:{args.port}")
        print(f"  Dokumentasi API : http://localhost:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        result = trigger_pipeline(
            config_path          = args.config_path,
            workflow_config_path = args.workflow_config_path,
            reason               = args.reason,
            wait                 = args.wait,
        )
        print(f"\nHasil: {result}")
