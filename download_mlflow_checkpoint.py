"""Download an existing checkpoint artifact directly from an MLflow run."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mlflow


REQUIRED_FILES = {"best_model.pt", "config.yaml"}


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the current environment."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def configure_r2_environment() -> None:
    """Map the repository's R2 variable names to boto3/MLflow names."""
    aliases = {
        "AWS_ACCESS_KEY_ID": "R2_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY": "R2_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL": "R2_ENDPOINT_URL",
    }
    for target, source in aliases.items():
        if source in os.environ:
            os.environ.setdefault(target, os.environ[source])
    os.environ.setdefault("AWS_DEFAULT_REGION", "auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download artifact checkpoint dari MLflow tanpa rerun Metaflow."
    )
    parser.add_argument("run_id", help="MLflow run ID yang memiliki artifact checkpoint")
    parser.add_argument(
        "-o",
        "--output",
        default="model_output_ci",
        help="Direktori tujuan (default: model_output_ci)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Override MLFLOW_TRACKING_URI dari environment/.env",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Timpa file checkpoint yang sudah ada di direktori tujuan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    configure_r2_environment()

    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit(
            "MLFLOW_TRACKING_URI belum tersedia. Isi .env/environment atau gunakan "
            "--tracking-uri."
        )

    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.MlflowClient().get_run(args.run_id)
    checkpoint_uri = f"{run.info.artifact_uri.rstrip('/')}/checkpoint"
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tracking URI   : {tracking_uri}")
    print(f"Checkpoint URI : {checkpoint_uri}")
    print(f"Output         : {output_dir}")

    with tempfile.TemporaryDirectory(
        prefix="mlflow-checkpoint-", dir=output_dir.parent
    ) as temp_dir:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=checkpoint_uri,
                dst_path=temp_dir,
            )
        )

        files = [path for path in downloaded.rglob("*") if path.is_file()]
        downloaded_names = {path.name for path in files}
        missing = REQUIRED_FILES - downloaded_names
        if missing:
            raise RuntimeError(
                "Checkpoint tidak lengkap; file wajib tidak ditemukan: "
                + ", ".join(sorted(missing))
            )

        for source in files:
            relative = source.relative_to(downloaded)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{destination} sudah ada. Gunakan --overwrite untuk menimpanya."
                )
            if destination.exists():
                destination.unlink()
            shutil.move(str(source), str(destination))

    print("\nCheckpoint berhasil diunduh:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  - {path.relative_to(output_dir)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
