"""
Ekspor gabungan val+test split ke file dataset.

Membaca dataset mentah, mereplikasi stratified split 70/15/15 yang sama
persis dengan yang dipakai prepare_data() (rasio & random_state dari config
eksperimen), lalu menggabungkan split val dan test menjadi satu file.

Preprocessing teks (apply_preprocessing) TIDAK direplikasi di sini karena
tidak memengaruhi hasil split — stratified_split hanya bergantung pada
kolom label yang dikonversi convert_labels(), bukan pada teks yang sudah
dibersihkan.

Penggunaan:
    python pipeline/export_val_test.py
    python pipeline/export_val_test.py --config configs/experiment_indobert_baseline.yaml
"""

import argparse
import os
import shutil
import sys

import pandas as pd
import yaml

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from preprocessing.load_data import load_data
from preprocessing.preprocessing_functions import convert_labels, stratified_split, FINAL_ASPECTS


def export_val_test(config_path: str) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg  = config['data']
    split_cfg = data_cfg['split']
    data_path = data_cfg['path']

    df_raw = load_data(data_path)
    original_columns = list(df_raw.columns)

    df = convert_labels(df_raw.copy())
    df_train, df_val, df_test = stratified_split(
        df,
        train_ratio  = split_cfg['train_ratio'],
        val_ratio    = split_cfg['val_ratio'],
        random_state = split_cfg['random_state'],
    )

    # Gabungkan val+test, urut ulang sesuai posisi asal di file mentah,
    # dan kembalikan hanya ke kolom asli (buang kolom lbl_* turunan split).
    df_combined = pd.concat([df_val, df_test]).sort_index()[original_columns]

    backup_path = data_path + '.bak'
    shutil.copy2(data_path, backup_path)

    df_combined.to_csv(data_path, index=False, encoding='utf-8')

    print(f"Total baris dataset mentah : {len(df_raw)}")
    print(f"Train (dibuang)            : {len(df_train)}")
    print(f"Val                        : {len(df_val)}")
    print(f"Test                       : {len(df_test)}")
    print(f"Val+Test digabung          : {len(df_combined)}")
    print(f"Backup dataset asli        : {backup_path}")
    print(f"Ditulis ke                 : {data_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gabungkan split val+test dan tulis ulang ke path dataset di config.'
    )
    parser.add_argument(
        '--config',
        default='configs/experiment_indobert_baseline.yaml',
        help='Path ke file konfigurasi eksperimen YAML (menyediakan data.path dan data.split).',
    )
    args = parser.parse_args()

    export_val_test(args.config)
