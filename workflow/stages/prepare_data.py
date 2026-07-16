import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.prepare_data import prepare_data


def run_prepare_data(model_config: dict) -> dict:
    data = prepare_data(model_config)

    n_train = len(data['df_train'])
    n_val   = len(data['df_val'])
    n_test  = len(data['df_test'])
    print(f"  Split dataset → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return data
