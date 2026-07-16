# Container image Automated ML Workflow Pipeline (ABSA Retraining)
# ==================================================================
# Berisi seluruh berkas implementasi pipeline beserta instalasi seluruh
# dependensinya. Dibangun pada tahap CI setelah unit test lolos, diuji
# integration test end-to-end, lalu disimpan ke container registry
# (Docker Hub). Lihat .github/workflows/ci-cd.yaml.
#
# Build : docker build -t absa-pipeline .
# Run   : docker run --rm absa-pipeline run
# Smoke : docker run --rm -e MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow_ci.db \
#           absa-pipeline run --config_path configs/integration_test.yaml

FROM python:3.12-slim

# git dibutuhkan DVC (versioning dataset pada tahap extract_data)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch build CPU dipasang lebih dulu dari index PyTorch — build default PyPI
# menyertakan CUDA (±2,5 GB) yang tidak terpakai di lingkungan pipeline CPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer dependensi terpisah dari kode agar cache build tetap dipakai
# selama requirements tidak berubah.
COPY requirements.txt ./requirements-train.txt
COPY workflow/requirements.txt ./requirements-workflow.txt
RUN pip install --no-cache-dir \
      -r requirements-train.txt \
      -r requirements-workflow.txt \
    && rm -rf /root/.cache

# Seluruh berkas implementasi pipeline (pengecualian di .dockerignore)
COPY . .

# Identitas eksekusi untuk metadata Metaflow
ENV USERNAME=absa-pipeline \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "workflow/flow.py"]
CMD ["run"]
