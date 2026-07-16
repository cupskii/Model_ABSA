import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.validate_data import validate_data


def run_validate_data(model_config: dict) -> dict:
    report = validate_data(model_config)

    print(f"  Total baris dataset: {report['total_rows']}")
    if report['issues']:
        for issue in report['issues']:
            print(f"  [!] {issue}")

    if not report['passed']:
        raise RuntimeError(
            f"Validasi data gagal dengan {len(report['issues'])} masalah kritis: "
            f"{report['issues']}"
        )

    print("  Validasi lulus — semua pemeriksaan berhasil.")
    return report
