import pandas as pd

from preprocessing.load_data import load_data
from preprocessing.preprocessing_functions import FINAL_ASPECTS

VALID_LABEL_VALUES = {-1, 0, 1}


def validate_data(config: dict) -> dict:
    data_path = config['data']['path']
    text_col  = config['data']['text_column']

    df = load_data(data_path)

    report = {
        'total_rows': len(df),
        'issues'    : [],
        'passed'    : True,
    }

    # 1. Periksa kolom teks
    if text_col not in df.columns:
        report['issues'].append(f"Kolom teks '{text_col}' tidak ditemukan")
        report['passed'] = False
        return report

    empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == '')
    n_empty = int(empty_mask.sum())
    if n_empty > 0:
        report['issues'].append(f"{n_empty} baris teks kosong ditemukan")
        report['n_empty_text'] = n_empty

    # 2. Periksa kolom label dan nilai yang valid
    for asp in FINAL_ASPECTS:
        if asp not in df.columns:
            report['issues'].append(f"Kolom label '{asp}' tidak ditemukan")
            report['passed'] = False
        else:
            non_nan = df[asp].dropna()
            invalid = non_nan[~non_nan.isin(VALID_LABEL_VALUES)]
            if len(invalid) > 0:
                report['issues'].append(
                    f"Kolom '{asp}' memiliki nilai tidak valid: "
                    f"{sorted(invalid.unique().tolist())}"
                )
    if 'Technical & Access' in df.columns:
        n_technical_neutral = int((df['Technical & Access'] == 0).sum())
        if n_technical_neutral:
            report['issues'].append(
                f"{n_technical_neutral} label Netral pada 'Technical & Access' "
                "akan dipetakan ke None oleh skema model 3 kelas saat ini"
            )

    return report


# ── PENGUJIAN MODUL ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    SAMPLE_CONFIG = {
        'data': {
            'path'       : 'data/raw/ABSA_dataset_final_CLEAN.csv',
            'text_column': 'Komentar',
        }
    }

    print("=" * 60)
    print("PENGUJIAN validate_data")
    print("=" * 60)

    report = validate_data(SAMPLE_CONFIG)

    print(f"Total baris : {report['total_rows']}")
    print(f"Status      : {'LULUS' if report['passed'] else 'GAGAL'}")
    if report['issues']:
        print(f"Masalah ({len(report['issues'])}):")
        for issue in report['issues']:
            print(f"  - {issue}")
    else:
        print("Tidak ada masalah ditemukan.")
    print("=" * 60)
