import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.prepare_data import prepare_data


def run_prepare_data(model_config: dict) -> dict:
    data = prepare_data(model_config)

    if 'cv_folds' in data:
        print(
            f"  Split dataset -> Development: {len(data['df_train'])} | "
            f"CV folds: {len(data['cv_folds'])} | Test: {len(data['df_test'])}"
        )
        return data

    n_train = len(data['df_train'])
    n_val   = len(data['df_val'])
    n_test  = len(data['df_test'])
    print(f"  Split dataset → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return data
