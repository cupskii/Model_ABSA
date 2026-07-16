import pandas as pd

from preprocessing.load_data import load_data
from preprocessing.preprocessing_functions import (
    remove_emoji, lowercase, remove_url_mention,
    compress_repeated_chars, remove_special_chars,
    normalize_whitespace, normalize_slang, remove_stopwords,
    convert_labels, stratified_split, stratified_split_train_test,
    multilabel_stratified_folds, compute_class_weights, FINAL_ASPECTS, LABEL_NAMES,
)


def apply_preprocessing(text: str, flags: dict) -> str:
    """
    Terapkan setiap tahap prapemrosesan sesuai flag boolean di konfigurasi.
    Urutan tahapan tetap, hanya keaktifannya yang dikendalikan flag.
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text)
    if flags.get('remove_emoji', True):
        text = remove_emoji(text)
    if flags.get('lowercase', True):
        text = lowercase(text)
    if flags.get('remove_url_mention', True):
        text = remove_url_mention(text)
    if flags.get('compress_repeated_chars', True):
        text = compress_repeated_chars(text)
    if flags.get('remove_special_chars', True):
        text = remove_special_chars(text)
    text = normalize_whitespace(text)
    if flags.get('normalize_slang', True):
        text = normalize_slang(text)
    if flags.get('remove_stopwords', True):
        text = remove_stopwords(text)
    return normalize_whitespace(text)


def print_class_distribution(**splits: pd.DataFrame) -> None:
    """
    Tampilkan jumlah & persentase tiap kelas, per aspek, dibandingkan
    antar split yang diberikan -- untuk memverifikasi hasil stratified split.
    Terima jumlah split berapa pun, mis. train/val/test atau train/test saja.
    """
    print("\n" + "=" * 60)
    print("DISTRIBUSI KELAS PER ASPEK PER SPLIT")
    print("=" * 60)

    for asp in FINAL_ASPECTS:
        col = f'lbl_{asp}'
        print(f"\n{asp}:")
        rows = []
        for idx, label_name in enumerate(LABEL_NAMES[asp]):
            row = {'kelas': label_name}
            for split_name, df_split in splits.items():
                cnt = int((df_split[col] == idx).sum())
                pct = 100 * cnt / len(df_split) if len(df_split) else 0.0
                row[f'{split_name}_n'] = cnt
                row[f'{split_name}_%'] = round(pct, 1)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))


def _load_and_label(config: dict) -> pd.DataFrame:
    """Muat dataset, terapkan prapemrosesan, dan konversi label -- tahap yang
    sama dipakai baik oleh split train/val/test maupun train/test (CV)."""
    data_cfg   = config['data']
    prep_flags = config['preprocessing']
    text_col   = data_cfg['text_column']

    df = load_data(data_cfg['path'])

    # Prapemrosesan teks dengan flag dari konfigurasi
    df['komentar_clean'] = df[text_col].apply(
        lambda t: apply_preprocessing(t, prep_flags)
    )

    # Fallback: teks yang menjadi kosong setelah preprocessing diisi versi lowercase aslinya
    empty_mask = df['komentar_clean'].str.strip() == ''
    if empty_mask.any():
        df.loc[empty_mask, 'komentar_clean'] = (
            df.loc[empty_mask, text_col].astype(str).str.lower().str.strip()
        )

    # Konversi label anotasi ke indeks kelas
    return convert_labels(df)


def prepare_data(config: dict) -> dict:
    """
    Muat dataset, terapkan prapemrosesan sesuai konfigurasi, bagi dataset
    menjadi train/val/test, dan hitung class weights dari training set.

    Returns
    -------
    dict dengan kunci:
      df_train      : DataFrame split pelatihan
      df_val        : DataFrame split validasi
      df_test       : DataFrame split pengujian
      class_weights : dict bobot kelas per aspek (dihitung dari train saja)
    """
    if config['data'].get('cv', {}).get('enabled', False):
        return prepare_data_cv(config)

    split_cfg = config['data']['split']
    df = _load_and_label(config)

    # Stratified split
    df_train, df_val, df_test = stratified_split(
        df,
        train_ratio  = split_cfg['train_ratio'],
        val_ratio    = split_cfg['val_ratio'],
        random_state = split_cfg['random_state'],
    )

    print_class_distribution(train=df_train, val=df_val, test=df_test)

    # Class weights dihitung dari training set saja
    class_weights = compute_class_weights(df_train)

    return {
        'df_train'     : df_train,
        'df_val'       : df_val,
        'df_test'      : df_test,
        'class_weights': class_weights,
    }


def prepare_data_cv(config: dict) -> dict:
    """
    Versi tanpa val split -- dipakai saat validasi dilakukan lewat
    cross-validation pada training set (bukan lewat val split terpisah).

    Konfigurasi 'data.split' cukup berisi 'train_ratio' dan 'random_state'
    (test mengambil sisa rasionya, mis. train_ratio=0.80 -> test=0.20).

    Returns
    -------
    dict dengan kunci:
      df_train      : DataFrame development (akan di-fold lagi saat CV)
      df_test       : DataFrame split pengujian (holdout, tidak disentuh saat CV)
      cv_folds      : indeks train/validation untuk setiap fold
      class_weights : bobot kelas development untuk final retraining
    """
    split_cfg = config['data']['split']
    cv_cfg    = config['data'].get('cv', {})
    df = _load_and_label(config)

    # Stratified split, hanya train/test
    df_train, df_test = stratified_split_train_test(
        df,
        train_ratio  = split_cfg['train_ratio'],
        random_state = split_cfg['random_state'],
    )

    print_class_distribution(development=df_train, test=df_test)

    cv_folds = multilabel_stratified_folds(
        df_train,
        n_splits=int(cv_cfg.get('n_splits', 5)),
        shuffle=bool(cv_cfg.get('shuffle', True)),
        random_state=int(cv_cfg.get('random_state', split_cfg['random_state'])),
    )
    print("\nCROSS-VALIDATION FOLDS")
    for fold_no, fold in enumerate(cv_folds, start=1):
        print(
            f"  Fold {fold_no}: train={len(fold['train_idx'])} | "
            f"val={len(fold['val_idx'])}"
        )

    # Class weights dihitung dari training set saja
    class_weights = compute_class_weights(df_train)

    return {
        'df_train'     : df_train,
        'df_test'      : df_test,
        'cv_folds'     : cv_folds,
        'class_weights': class_weights,
    }


# ── PENGUJIAN MODUL ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    SAMPLE_CONFIG = {
        'data': {
            'path'       : 'data/raw/dataset_final.csv',
            'text_column': 'Komentar',
            'split': {
                'train_ratio' : 0.80,
                'random_state': 42,
            },
            'cv': {
                'enabled'     : True,
                'n_splits'    : 5,
                'shuffle'     : True,
                'random_state': 42,
            },
        },
        'preprocessing': {
            'remove_emoji'           : True,
            'lowercase'              : True,
            'remove_url_mention'     : True,
            'compress_repeated_chars': True,
            'remove_special_chars'   : True,
            'normalize_slang'        : True,
            'remove_stopwords'       : False,
        },
    }

    print("=" * 60)
    print("PENGUJIAN prepare_data")
    print("=" * 60)

    data = prepare_data(SAMPLE_CONFIG)

    print(f"Development : {len(data['df_train'])} baris")
    print(f"CV folds    : {len(data['cv_folds'])}")
    print(f"Test        : {len(data['df_test'])} baris")

    print("\nSampel teks setelah preprocessing:")
    for _, row in data['df_train'].head(3).iterrows():
        print(f"  {row['komentar_clean'][:80]}")

    print("\nClass weights:")
    for asp, w in data['class_weights'].items():
        print(f"  {asp}: {[round(x, 3) for x in w]}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("PENGUJIAN prepare_data_cv (train/test saja, untuk cross-validation)")
    print("=" * 60)

    # CV_CONFIG = {
    #     'data': {
    #         'path'       : SAMPLE_CONFIG['data']['path'],
    #         'text_column': 'Komentar',
    #         'split': {
    #             'train_ratio' : 0.80,
    #             'random_state': 42,
    #         },
    #     },
    #     'preprocessing': SAMPLE_CONFIG['preprocessing'],
    # }

    # data_cv = prepare_data_cv(CV_CONFIG)

    # print(f"Train : {len(data_cv['df_train'])} baris")
    # print(f"Test  : {len(data_cv['df_test'])} baris")
    # print("=" * 60)
