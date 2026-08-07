"""Promosikan versi model MLflow Registry untuk dipakai di production.

Cara yang direkomendasikan MLflow 3.x adalah alias, bukan model stage:

    python promote_mlflow_model.py

Perintah di atas memasang alias ``champion`` dan tag
``deployment_status=production`` pada versi terbaru ``absa_indobert``.

Jika integrasi lama masih membaca stage ``Production``, gunakan:

    python promote_mlflow_model.py --version 1 --set-legacy-stage
"""

import argparse
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promosikan sebuah model version di MLflow Model Registry."
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="Alamat MLflow Tracking Server (default: %(default)s).",
    )
    parser.add_argument(
        "--model-name",
        default="absa_indobert",
        help="Nama registered model (default: %(default)s).",
    )
    parser.add_argument(
        "--version",
        help="Versi yang dipromosikan. Jika kosong, pilih versi bernomor terbesar.",
    )
    parser.add_argument(
        "--alias",
        default="champion",
        help="Alias model production (default: %(default)s).",
    )
    parser.add_argument(
        "--set-legacy-stage",
        action="store_true",
        help="Juga set stage deprecated menjadi Production.",
    )
    return parser.parse_args()


def latest_version(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise RuntimeError(
            f"Registered model '{model_name}' tidak memiliki model version."
        )
    return str(max(versions, key=lambda item: int(item.version)).version)


def promote(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    client = MlflowClient()

    version = str(args.version) if args.version else latest_version(
        client, args.model_name
    )

    # Pastikan target ada sebelum alias/tag diubah.
    target = client.get_model_version(args.model_name, version)
    if str(target.status).upper() == "FAILED_REGISTRATION":
        raise RuntimeError(
            f"{args.model_name} v{version} berstatus FAILED_REGISTRATION."
        )

    client.set_registered_model_alias(args.model_name, args.alias, version)
    client.set_model_version_tag(
        args.model_name, version, "deployment_status", "production"
    )

    if args.set_legacy_stage:
        client.transition_model_version_stage(
            name=args.model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )

    print(f"Berhasil: {args.model_name} v{version}")
    print(f"Alias : {args.alias}")
    print("Tag   : deployment_status=production")
    print(f"URI   : models:/{args.model_name}@{args.alias}")
    if args.set_legacy_stage:
        print("Stage : Production (legacy/deprecated)")


def main() -> int:
    args = parse_args()
    try:
        promote(args)
        return 0
    except (MlflowException, RuntimeError) as exc:
        print(f"Gagal mempromosikan model: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
