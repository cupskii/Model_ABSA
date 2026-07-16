import pandas as pd

from preprocessing.load_data import load_data
from preprocessing.preprocessing_functions import (
    remove_emoji, lowercase, remove_url_mention,
    compress_repeated_chars, remove_special_chars,
    normalize_whitespace, normalize_slang, remove_stopwords,
    convert_labels, stratified_split, compute_class_weights, FINAL_ASPECTS,
)

SENTIMENT_LABELS = {
    -1: 'Negatif',
    0: 'Netral',
    1: 'Positif',
}


def compute_sentiment_proportions(df: pd.DataFrame) -> dict:
    """Hitung jumlah dan proporsi sentimen berlabel pada setiap aspek.

    Nilai NaN tidak dimasukkan ke penyebut proporsi karena menandakan aspek
    tersebut tidak dibahas. Jumlah NaN tetap disertakan sebagai ``n_missing``.
    """
    proportions = {}

    for aspect in FINAL_ASPECTS:
        if aspect not in df.columns:
            continue

        labeled = df[aspect].dropna()
        n_labeled = int(len(labeled))
        sentiments = {}

        for value, name in SENTIMENT_LABELS.items():
            count = int((labeled == value).sum())
            percentage = (count / n_labeled * 100) if n_labeled else 0.0
            sentiments[name] = {
                'count': count,
                'percentage': round(percentage, 2),
            }

        proportions[aspect] = {
            'n_labeled': n_labeled,
            'n_missing': int(df[aspect].isna().sum()),
            'sentiments': sentiments,
        }

    return proportions


def compute_split_sentiment_proportions(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict:
    return {
        'train': compute_sentiment_proportions(df_train),
        'val': compute_sentiment_proportions(df_val),
        'test': compute_sentiment_proportions(df_test),
    }


def print_sentiment_proportions(
    proportions: dict,
    split_name: str | None = None,
) -> None:
    """Tampilkan proporsi sentimen per aspek dalam bentuk tabel teks."""
    split_label = f" - {split_name.upper()}" if split_name else ''
    print(f"\nProporsi sentimen per aspek{split_label} "
          "(berdasarkan data berlabel):")
    print(f"{'Aspek':<25} {'Berlabel':>8} {'Negatif':>17} "
          f"{'Netral':>17} {'Positif':>17}")
    print('-' * 88)

    for aspect, stats in proportions.items():
        cells = []
        for name in SENTIMENT_LABELS.values():
            sentiment = stats['sentiments'][name]
            cells.append(
                f"{sentiment['count']} ({sentiment['percentage']:.2f}%)"
            )
        print(f"{aspect:<25} {stats['n_labeled']:>8} "
              f"{cells[0]:>17} {cells[1]:>17} {cells[2]:>17}")


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
      sentiment_proportions : jumlah dan persentase sentimen per aspek untuk
                              split train, val, dan test
    """
    data_cfg   = config['data']
    prep_flags = config['preprocessing']
    split_cfg  = data_cfg['split']
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
    df = convert_labels(df)

    # Stratified split
    df_train, df_val, df_test = stratified_split(
        df,
        train_ratio  = split_cfg['train_ratio'],
        val_ratio    = split_cfg['val_ratio'],
        random_state = split_cfg['random_state'],
    )

    # Distribusi dihitung per split dari label anotasi asli. Nilai label aspek
    # masih tersedia meskipun kolom indeks kelas sudah ditambahkan.
    sentiment_proportions = compute_split_sentiment_proportions(
        df_train, df_val, df_test
    )

    # Class weights dihitung dari training set saja
    class_weights = compute_class_weights(df_train)

    return {
        'df_train'     : df_train,
        'df_val'       : df_val,
        'df_test'      : df_test,
        'class_weights': class_weights,
        'sentiment_proportions': sentiment_proportions,
    }


# ── PENGUJIAN MODUL ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    SAMPLE_CONFIG = {
        'data': {
            'path'       : 'data/raw/dataset_sisa.csv',
            'text_column': 'Komentar',
            'split': {
                'train_ratio' : 0.70,
                'val_ratio'   : 0.15,
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

    print(f"Train : {len(data['df_train'])} baris")
    print(f"Val   : {len(data['df_val'])} baris")
    print(f"Test  : {len(data['df_test'])} baris")

    for split_name, proportions in data['sentiment_proportions'].items():
        print_sentiment_proportions(proportions, split_name)

    print("\nSampel teks setelah preprocessing:")
    for _, row in data['df_train'].head(3).iterrows():
        print(f"  {row['komentar_clean'][:80]}")

    print("\nClass weights:")
    for asp, w in data['class_weights'].items():
        print(f"  {asp}: {[round(x, 3) for x in w]}")
    print("=" * 60)
