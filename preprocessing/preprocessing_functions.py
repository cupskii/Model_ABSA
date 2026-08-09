import re
import pandas as pd


# ── KONFIGURASI ──────────────────────────────────────────────────────────────
FINAL_ASPECTS = [
    'Content Quality',
    'Subscription & Pricing',
    'UI/UX',
    'Functionality',
    'Technical & Access',
]

NUM_CLASSES = {
    'Content Quality'       : 4,   # Neg / Neu / Pos / None
    'Subscription & Pricing': 4,
    'UI/UX'                 : 4,
    'Functionality'         : 4,
    'Technical & Access'    : 3,   # Neg / Pos / None
}

LABEL_NAMES = {
    'Content Quality'       : ['Negatif', 'Netral', 'Positif', 'None'],
    'Subscription & Pricing': ['Negatif', 'Netral', 'Positif', 'None'],
    'UI/UX'                 : ['Negatif', 'Netral', 'Positif', 'None'],
    'Functionality'         : ['Negatif', 'Netral', 'Positif', 'None'],
    'Technical & Access'    : ['Negatif', 'Positif', 'None'],
}

SLANG_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
    'gk' : 'tidak', 'tdk': 'tidak', 'yg'  : 'yang' , 'dgn' : 'dengan',
    'dg' : 'dengan','utk': 'untuk', 'krn' : 'karena','karna': 'karena',
    'klo': 'kalau', 'kl' : 'kalau', 'udah': 'sudah', 'udh' : 'sudah',
    'sdh': 'sudah', 'blm': 'belum', 'lg'  : 'lagi'  ,'lgs' : 'langsung',
    'sy' : 'saya' , 'bgt': 'banget','bs'  : 'bisa'  ,'bsa' : 'bisa',
    'tp' : 'tapi' , 'ttp': 'tetap', 'jd'  : 'jadi'  ,
    'app': 'aplikasi', 'apps': 'aplikasi', 'apk': 'aplikasi',
    'ok' : 'oke'  , 'subs': 'berlangganan', 'langganan': 'berlangganan',
    'epaper': 'e-paper', 'e paper': 'e-paper',
    'independent' : 'independen',  
    'credible'    : 'kredibel',    
    'content'     : 'konten',     
}

# Kata-kata yang SENGAJA tidak dimasukkan ke STOPWORDS karena bermakna sentimen:
#   Negasi       : tidak, bukan, belum, jangan, tanpa, tak, tiada
#   Kontrastif   : tapi, tetapi, namun, melainkan, sedangkan, padahal,
#                  meskipun, walaupun, kendati, meski, biarpun, sekalipun
#   Intensifier  : sangat, sekali, banget, kurang, cukup, agak, lumayan,
#                  paling, lebih, terlalu, amat, begitu, sungguh, hampir
#   Kondisional  : karena, sebab, sehingga, jika, kalau, apabila, bila
STOPWORDS = {
    # konjungsi koordinatif netral
    'dan', 'atau', 'serta',
    # preposisi
    'di', 'ke', 'dari', 'pada', 'dalam', 'oleh', 'dengan', 'untuk',
    'bagi', 'tentang', 'terhadap', 'mengenai', 'antara',
    'sejak', 'hingga', 'sampai', 'sebelum', 'setelah', 'selama',
    # kata ganti orang
    'saya', 'aku', 'kamu', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita',
    # determiner
    'ini', 'itu', 'tersebut', 'yang',
    # kopula / eksistensial
    'adalah', 'ialah', 'merupakan', 'ada',
    # partikel informal (tidak bermakna sentimen)
    'sih', 'deh', 'nih', 'kah', 'lah', 'pun', 'kok', 'kan',
    'ya', 'dong', 'lho', 'lo', 'si',
    # aspektual netral
    'sudah', 'telah', 'akan', 'sedang',
    # aditif
    'juga', 'pula',
    # keterangan waktu generik
    'saat', 'ketika', 'waktu',
    # lainnya
    'para', 'sang',
}


# ── FUNGSI PEMBERSIHAN TEKS (STEP 1) ─────────────────────────────────────────

def remove_emoji(text: str) -> str:
    """Hapus emoji dan karakter non-ASCII."""
    return text.encode('ascii', 'ignore').decode('ascii')


def lowercase(text: str) -> str:
    """Konversi seluruh teks ke huruf kecil."""
    return text.lower()


def remove_url_mention(text: str) -> str:
    """Hapus URL (http/www) dan mention (@username)."""
    return re.sub(r'http\S+|www\S+|@\w+', '', text)


def compress_repeated_chars(text: str) -> str:
    """Compress karakter berulang lebih dari 2 kali (mahaaal → mahaal)."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def remove_special_chars(text: str) -> str:
    """Hapus karakter tidak perlu; pertahankan huruf, angka, dan ?.!.,-'"""
    return re.sub(r"[^a-z0-9\s\?\!\.\,\-']", ' ', text)


def normalize_whitespace(text: str) -> str:
    """Hilangkan spasi ganda dan spasi di awal/akhir teks."""
    return re.sub(r'\s+', ' ', text).strip()


def normalize_slang(text: str) -> str:
    """Ganti kata tidak baku dengan padanan baku berdasarkan SLANG_DICT."""
    words = [SLANG_DICT.get(w, w) for w in text.split()]
    return ' '.join(words)


def remove_stopwords(text: str) -> str:
    """
    Hapus stopword netral dari teks.

    Kata-kata yang TIDAK dihapus (dipertahankan secara eksplisit):
      - Negasi     : tidak, bukan, belum, jangan, tanpa, tak, tiada
      - Kontrastif : tapi, tetapi, namun, melainkan, sedangkan, padahal,
                     meskipun, walaupun, kendati, meski, biarpun, sekalipun
      - Intensifier: sangat, sekali, banget, kurang, cukup, agak, lumayan,
                     paling, lebih, terlalu, amat, begitu, sungguh, hampir
      - Kondisional: karena, sebab, sehingga, jika, kalau, apabila, bila

    Catatan: fungsi ini TIDAK dipanggil oleh clean_text() karena IndoBERT
    bersifat kontekstual dan tidak memerlukan penghapusan stopword.
    Gunakan fungsi ini hanya pada pipeline non-BERT (TF-IDF, dll.).
    """
    words = [w for w in text.split() if w not in STOPWORDS]
    return normalize_whitespace(' '.join(words))


def clean_text(text: str) -> str:
    """
    Pembersihan teks lengkap untuk IndoBERT — memanggil semua fungsi di atas
    secara berurutan. Stopword tidak dihapus karena BERT sudah kontekstual.

    Urutan:
      1. Hapus emoji (non-ASCII)
      2. Lowercase
      3. Hapus URL dan mention
      4. Compress karakter berulang >2
      5. Hapus karakter tidak perlu, pertahankan ?.!.,
      6. Normalisasi spasi
      7. Normalisasi kata slang (word-level)
      8. Normalisasi spasi (pasca-slang)
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text)
    text = remove_emoji(text)
    text = lowercase(text)
    text = remove_url_mention(text)
    text = compress_repeated_chars(text)
    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    text = normalize_slang(text)
    text = normalize_whitespace(text)
    return text


# ── KONVERSI LABEL (STEP 2) ───────────────────────────────────────────────────

def convert_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konversi label anotasi → indeks kelas PyTorch.

    Skema:
      4-kelas: -1 → 0 (Neg) | 0 → 1 (Neu) | 1 → 2 (Pos) | NaN → 3 (None)
      3-kelas: -1 → 0 (Neg) | 1 → 1 (Pos)               | NaN → 2 (None)
    """
    def to_3class(v):
        if pd.isna(v): return 3
        return {-1: 0, 0: 1, 1: 2}.get(int(v), 3)

    def to_binary(v):
        # Skema biner tidak punya kelas Netral: anotasi 0 (Netral) yang masih
        # tersisa di data mentah dipetakan eksplisit ke None (bukan fallback
        # diam-diam) — sebaiknya baris seperti ini dibersihkan di sumber data.
        if pd.isna(v): return 2
        return {-1: 0, 0: 2, 1: 1}.get(int(v), 2)

    for asp in FINAL_ASPECTS:
        col = asp if asp in df.columns else None
        if col:
            fn = to_binary if asp == 'Technical & Access' else to_3class
            df[f'lbl_{asp}'] = df[asp].apply(fn)
        else:
            none_idx = 2 if asp == 'Technical & Access' else 3
            df[f'lbl_{asp}'] = none_idx

    return df


# ── STRATIFIED SPLIT 70/15/15 (STEP 3) ───────────────────────────────────────

def build_multilabel_matrix(df: pd.DataFrame):
    """Bangun matriks biner multi-label untuk keperluan stratified split."""
    Y = pd.DataFrame(index=df.index)

    Y['CQ_neg']  = (df['lbl_Content Quality'] == 0).astype(int)
    Y['CQ_neu']  = (df['lbl_Content Quality'] == 1).astype(int)
    Y['CQ_pos']  = (df['lbl_Content Quality'] == 2).astype(int)
    Y['CQ_none'] = (df['lbl_Content Quality'] == 3).astype(int)

    Y['SP_neg']  = (df['lbl_Subscription & Pricing'] == 0).astype(int)
    Y['SP_neu']  = (df['lbl_Subscription & Pricing'] == 1).astype(int)
    Y['SP_pos']  = (df['lbl_Subscription & Pricing'] == 2).astype(int)
    Y['SP_none'] = (df['lbl_Subscription & Pricing'] == 3).astype(int)

    Y['UI_neg']  = (df['lbl_UI/UX'] == 0).astype(int)
    Y['UI_neu']  = (df['lbl_UI/UX'] == 1).astype(int)
    Y['UI_pos']  = (df['lbl_UI/UX'] == 2).astype(int)
    Y['UI_none'] = (df['lbl_UI/UX'] == 3).astype(int)

    Y['FN_neg']  = (df['lbl_Functionality'] == 0).astype(int)
    Y['FN_neu']  = (df['lbl_Functionality'] == 1).astype(int)
    Y['FN_pos']  = (df['lbl_Functionality'] == 2).astype(int)
    Y['FN_none'] = (df['lbl_Functionality'] == 3).astype(int)

    Y['TA_neg']  = (df['lbl_Technical & Access'] == 0).astype(int)
    Y['TA_pos']  = (df['lbl_Technical & Access'] == 1).astype(int)
    Y['TA_none'] = (df['lbl_Technical & Access'] == 2).astype(int)

    return Y.values


def stratified_split(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15, random_state=42):
    """Bagi dataset menjadi train/val/test dengan multilabel stratified split."""
    # Import lokal (bukan di top-level modul): iterstrat cuma dipakai fungsi
    # ini (training-time), supaya modul ini tetap bisa diimpor tanpa iterstrat
    # ter-install saat load_context() model serving (absa_pyfunc.py) — lihat
    # register_model.py untuk daftar pip_requirements model serving.
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    df = df.copy()
    Y = build_multilabel_matrix(df)

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - train_ratio), random_state=random_state
    )
    train_idx, temp_idx = next(msss.split(df, Y))
    df_train = df.iloc[train_idx].copy()
    df_temp  = df.iloc[temp_idx].copy()

    temp_ratio   = 1 - train_ratio
    val_fraction = val_ratio / temp_ratio
    Y_temp = Y[temp_idx]

    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_fraction), random_state=random_state
    )
    val_idx_local, test_idx_local = next(msss2.split(df_temp, Y_temp))
    df_val  = df_temp.iloc[val_idx_local].copy()
    df_test = df_temp.iloc[test_idx_local].copy()

    return df_train, df_val, df_test


def check_split_size(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """Cetak ringkasan ukuran dan proporsi setiap split."""
    total = len(df_train) + len(df_val) + len(df_test)
    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train : {len(df_train):4d} ({len(df_train)/total:.2%})")
    print(f"Val   : {len(df_val):4d} ({len(df_val)/total:.2%})")
    print(f"Test  : {len(df_test):4d} ({len(df_test)/total:.2%})")
    print("=" * 60)


def compare_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame,
                         df_test: pd.DataFrame, aspect_col: str):
    """Bandingkan distribusi kelas pada setiap split untuk satu aspek."""
    def get_dist(df):
        labels  = df[aspect_col]
        counts  = labels.value_counts().sort_index()
        pct     = labels.value_counts(normalize=True).sort_index().mul(100).round(2)
        return counts, pct

    train_count, train_pct = get_dist(df_train)
    val_count,   val_pct   = get_dist(df_val)
    test_count,  test_pct  = get_dist(df_test)

    classes = sorted(
        set(train_count.index) | set(val_count.index) | set(test_count.index)
    )
    rows = [
        {
            "Class"      : c,
            "Train Count": train_count.get(c, 0), "Train %": train_pct.get(c, 0),
            "Val Count"  : val_count.get(c, 0),   "Val %"  : val_pct.get(c, 0),
            "Test Count" : test_count.get(c, 0),  "Test %" : test_pct.get(c, 0),
        }
        for c in classes
    ]
    print("\n" + "=" * 80)
    print(aspect_col)
    print("=" * 80)
    print(pd.DataFrame(rows).to_string(index=False))


# ── CLASS WEIGHTS (STEP 4) ────────────────────────────────────────────────────

def compute_class_weights(df_train: pd.DataFrame) -> dict:
    """
    Hitung class weights dari training set.
    Formula: w[c] = N / (n_classes × count[c])
    """
    weights = {}
    for asp in FINAL_ASPECTS:
        col   = f'lbl_{asp}'
        n_cls = NUM_CLASSES[asp]
        N     = len(df_train)
        w_list = []
        for c in range(n_cls):
            cnt = int((df_train[col] == c).sum())
            w   = float(N) / (n_cls * cnt) if cnt > 0 else 1.0
            w_list.append(w)
        weights[asp] = w_list
    return weights
