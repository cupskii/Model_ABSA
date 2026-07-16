"""
Skrip Perhitungan Interrater Agreement untuk Anotasi ABSA Multilabel
(Penulis vs AI) -- Aspect-Based Sentiment Analysis

FORMAT DATA YANG DIHARAPKAN
----------------------------
Satu baris = satu komentar. Satu kolom per aspek, berisi:
    -1  -> sentimen negatif untuk aspek tsb
     0  -> sentimen netral untuk aspek tsb
     1  -> sentimen positif untuk aspek tsb
    (kosong / NaN) -> aspek tsb TIDAK dibahas di komentar tsb

Contoh initial_penulis.csv:
    komentar,UI/UX,Functionality,Subscription & Pricing,Content Quality,Technical & Access
    "Tampilan bagus tapi sering error",1,-1,,,
    "Harga langganan kemahalan",,,-1,,
    "Loading lambat terus",,,,,-1

Contoh initial_ai.csv: struktur kolom sama persis.

CARA PAKAI
----------
1. Sesuaikan bagian KONFIGURASI di bawah (terutama KEY_COLUMN dan
   ASPECT_COLUMNS) dengan data kamu.
2. Jalankan: python hitung_kappa_absa.py

KENAPA DIPISAH JADI DUA JENIS AGREEMENT PER ASPEK
----------------------------------------------------
- "Agreement Deteksi Aspek": mengukur apakah kedua pelabel SAMA-SAMA
  sepakat suatu aspek dibahas atau tidak di komentar tsb (non-null vs
  null), dihitung sebagai Cohen's Kappa biner.
- "Agreement Sentimen": HANYA dihitung pada subset komentar yang sudah
  disepakati kedua pelabel sebagai membahas aspek tsb (keduanya
  non-null) -- lalu dicek apakah nilai sentimen (-1/0/1) juga sama.
  Ini menghindari kappa sentimen "tercemar" oleh disagreement soal
  relevansi aspek, yang sebetulnya persoalan berbeda.

OUTPUT
------
- hasil_agreement_aspek.csv    : kappa deteksi aspek per aspek + overall
- hasil_agreement_sentimen.csv : kappa sentimen per aspek + overall
- confusion_matrix_aspek_<nama_aspek>.csv     : 2x2 (dibahas / tidak)
- confusion_matrix_sentimen_<nama_aspek>.csv  : 3x3 (-1/0/1)
"""

import sys
import warnings
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Beberapa kombinasi data yang valid (mis. subset kecil dengan satu kelas
# saja, atau agreement sempurna) memicu warning teknis dari numpy/sklearn
# meski hasil perhitungannya tetap benar (ditangani sebagai "n/a" secara
# eksplisit di kode). Warning ini disembunyikan agar output tidak berisik.
warnings.filterwarnings("ignore")

# ============================== KONFIGURASI ==============================
FILE_PENULIS = "initial_penulis.csv"
FILE_AI = "initial_ai.csv"

# Kolom kunci untuk mencocokkan baris antar file.
# Default pakai teks komentar itu sendiri. Kalau datamu punya kolom ID unik
# (lebih aman daripada mencocokkan lewat teks komentar), ganti ke nama
# kolom ID tsb -- jauh lebih disarankan kalau ada kemungkinan komentar
# duplikat/identik.
KEY_COLUMN = "Komentar"

ASPECT_COLUMNS = [
    "UI/UX",
    "Functionality",
    "Subscription & Pricing",
    "Content Quality",
    "Technical & Access",
]

# Jika True, spasi berlebih pada teks komentar dirapikan dulu sebelum
# dicocokkan (menghindari gagal match hanya karena spasi tersisa).
NORMALIZE_KEY_WHITESPACE = True

OUTPUT_ASPEK = "hasil_agreement_aspek.csv"
OUTPUT_SENTIMEN = "hasil_agreement_sentimen.csv"
CM_PREFIX_ASPEK = "confusion_matrix_aspek"
CM_PREFIX_SENTIMEN = "confusion_matrix_sentimen"
# ===========================================================================


def interpretasi_kappa(k):
    """Interpretasi nilai kappa berdasarkan skala Landis & Koch (1977)."""
    if k != k:  # NaN check
        return "n/a"
    if k < 0:
        return "Poor"
    elif k <= 0.20:
        return "Slight"
    elif k <= 0.40:
        return "Fair"
    elif k <= 0.60:
        return "Moderate"
    elif k <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def hitung_kappa_aman(y1, y2):
    """Wrapper cohen_kappa_score yang tidak crash saat semua label sama
    persis di kedua sisi (kasus 0/0 pada rumus kappa -- terjadi kalau
    Po = Pe = 1, kappa jadi tak terdefinisi meski agreement sempurna)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            k = cohen_kappa_score(y1, y2)
        except Exception:
            k = float("nan")
    return k


def hitung_pe(p_observed, kappa):
    """Menurunkan Pe (chance agreement) dari Po dan kappa untuk pelaporan
    terpisah -- membantu mendiagnosis kappa rendah akibat distribusi
    kelas yang timpang, bukan karena annotator benar-benar sering beda."""
    if kappa != kappa:  # kappa NaN (mis. hanya 1 kelas yang muncul)
        return float("nan")
    if kappa >= 1.0:  # agreement sempurna -> Pe tak terdefinisi, bukan error
        return float("nan")
    return (p_observed - kappa) / (1 - kappa)


def muat_dan_gabungkan(file_a, file_b, key_col):
    try:
        df_a = pd.read_csv(file_a)
        df_b = pd.read_csv(file_b)
    except FileNotFoundError as e:
        sys.exit(f"[ERROR] File tidak ditemukan: {e.filename}")

    if key_col not in df_a.columns or key_col not in df_b.columns:
        sys.exit(
            f"[ERROR] Kolom kunci '{key_col}' tidak ditemukan di salah satu file.\n"
            f"  Kolom di {file_a}: {list(df_a.columns)}\n"
            f"  Kolom di {file_b}: {list(df_b.columns)}"
        )

    if NORMALIZE_KEY_WHITESPACE:
        df_a[key_col] = df_a[key_col].astype(str).str.strip()
        df_b[key_col] = df_b[key_col].astype(str).str.strip()

    dup_a = df_a[key_col].duplicated().sum()
    dup_b = df_b[key_col].duplicated().sum()
    if dup_a or dup_b:
        print(
            f"[PERINGATAN] Ditemukan nilai '{key_col}' duplikat "
            f"({file_a}: {dup_a}, {file_b}: {dup_b}). Ini berisiko membuat "
            f"penggabungan data tidak akurat (satu baris bisa cocok ke lebih "
            f"dari satu baris pasangannya). Sebaiknya gunakan kolom ID unik "
            f"sebagai KEY_COLUMN jika memungkinkan.\n"
        )

    merged = pd.merge(df_a, df_b, on=key_col, how="inner", suffixes=("_penulis", "_ai"))

    print(f"Jumlah komentar pada '{file_a}' : {len(df_a)}")
    print(f"Jumlah komentar pada '{file_b}' : {len(df_b)}")
    print(f"Jumlah komentar overlap (irisan): {len(merged)}\n")

    if merged.empty:
        sys.exit("[ERROR] Tidak ada baris yang cocok di kedua file. Periksa kembali KEY_COLUMN.")

    # Pastikan kolom sentimen bertipe numerik nullable (-1/0/1/NA), sekaligus
    # deteksi jika ada nilai non-numerik nyasar (misal typo teks di kolom aspek).
    for aspek in ASPECT_COLUMNS:
        for suf in ("_penulis", "_ai"):
            col = f"{aspek}{suf}"
            if col not in merged.columns:
                continue
            asli_non_null = merged[col].notna()
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
            rusak = asli_non_null & merged[col].isna()
            if rusak.any():
                print(
                    f"[PERINGATAN] {rusak.sum()} nilai pada kolom '{col}' bukan angka "
                    f"-1/0/1 yang valid dan diubah jadi kosong (NaN). Periksa data mentahnya."
                )

    return merged


def proses_satu_aspek(merged, aspek):
    col_p = f"{aspek}_penulis"
    col_a = f"{aspek}_ai"

    if col_p not in merged.columns or col_a not in merged.columns:
        print(f"[DILEWATI] Kolom aspek '{aspek}' tidak ditemukan di kedua file.")
        return None, None, None, None

    hadir_p = merged[col_p].notna()
    hadir_a = merged[col_a].notna()

    # ---- 1) Agreement deteksi aspek (biner: dibahas / tidak) ----
    n_total = len(merged)
    p_obs_aspek = (hadir_p == hadir_a).mean()
    k_aspek = hitung_kappa_aman(hadir_p, hadir_a)
    pe_aspek = hitung_pe(p_obs_aspek, k_aspek)

    hasil_aspek = {
        "aspek": aspek,
        "n_komentar": n_total,
        "percent_agreement": round(p_obs_aspek, 3),
        "chance_agreement": round(pe_aspek, 3) if pe_aspek == pe_aspek else "n/a",
        "cohen_kappa": round(k_aspek, 3) if k_aspek == k_aspek else "n/a",
        "interpretasi": interpretasi_kappa(k_aspek),
    }

    cm_aspek = pd.DataFrame(
        confusion_matrix(hadir_p, hadir_a, labels=[False, True]),
        index=["Penulis=tidak_dibahas", "Penulis=dibahas"],
        columns=["AI=tidak_dibahas", "AI=dibahas"],
    )

    # ---- 2) Agreement sentimen, hanya pada subset yang sama-sama "dibahas" ----
    subset = merged.loc[hadir_p & hadir_a, [col_p, col_a]]
    n_overlap_sentimen = len(subset)

    if n_overlap_sentimen == 0:
        hasil_sentimen = {
            "aspek": aspek,
            "n_komentar_disepakati_dibahas": 0,
            "percent_agreement": "n/a",
            "chance_agreement": "n/a",
            "cohen_kappa": "n/a",
            "interpretasi": "n/a (tidak ada komentar yang disepakati membahas aspek ini)",
        }
        cm_sentimen = None
    else:
        y_p = subset[col_p]
        y_a = subset[col_a]
        p_obs_sent = (y_p == y_a).mean()
        k_sent = hitung_kappa_aman(y_p, y_a)
        pe_sent = hitung_pe(p_obs_sent, k_sent)

        hasil_sentimen = {
            "aspek": aspek,
            "n_komentar_disepakati_dibahas": n_overlap_sentimen,
            "percent_agreement": round(p_obs_sent, 3),
            "chance_agreement": round(pe_sent, 3) if pe_sent == pe_sent else "n/a",
            "cohen_kappa": round(k_sent, 3) if k_sent == k_sent else "n/a",
            "interpretasi": interpretasi_kappa(k_sent),
        }

        label_sentimen = sorted(set(y_p.dropna()) | set(y_a.dropna()))
        cm_sentimen = pd.DataFrame(
            confusion_matrix(y_p, y_a, labels=label_sentimen),
            index=[f"Penulis={l}" for l in label_sentimen],
            columns=[f"AI={l}" for l in label_sentimen],
        )

    return hasil_aspek, cm_aspek, hasil_sentimen, cm_sentimen


def main():
    print("=" * 66)
    print(" PERHITUNGAN INTERRATER AGREEMENT ABSA MULTILABEL: PENULIS vs AI")
    print("=" * 66, "\n")

    merged = muat_dan_gabungkan(FILE_PENULIS, FILE_AI, KEY_COLUMN)

    hasil_aspek_list, hasil_sentimen_list = [], []
    hadir_p_all, hadir_a_all = [], []       # untuk kappa aspek overall (pooled)
    sent_p_all, sent_a_all = [], []         # untuk kappa sentimen overall (pooled)

    for aspek in ASPECT_COLUMNS:
        hasil_aspek, cm_aspek, hasil_sentimen, cm_sentimen = proses_satu_aspek(merged, aspek)
        if hasil_aspek is None:
            continue

        hasil_aspek_list.append(hasil_aspek)
        hasil_sentimen_list.append(hasil_sentimen)

        if cm_aspek is not None:
            path = f"{CM_PREFIX_ASPEK}_{aspek}.csv".replace("/", "-")
            cm_aspek.to_csv(path)
        if cm_sentimen is not None:
            path = f"{CM_PREFIX_SENTIMEN}_{aspek}.csv".replace("/", "-")
            cm_sentimen.to_csv(path)

        col_p, col_a = f"{aspek}_penulis", f"{aspek}_ai"
        hadir_p_all.append(merged[col_p].notna())
        hadir_a_all.append(merged[col_a].notna())

        both = merged[col_p].notna() & merged[col_a].notna()
        sent_p_all.append(merged.loc[both, col_p])
        sent_a_all.append(merged.loc[both, col_a])

    if not hasil_aspek_list:
        sys.exit("[ERROR] Tidak ada aspek yang berhasil diproses. Periksa ASPECT_COLUMNS.")

    # ---- Baris OVERALL: gabungkan seluruh aspek jadi satu perhitungan pooled ----
    hadir_p_pool = pd.concat(hadir_p_all, ignore_index=True)
    hadir_a_pool = pd.concat(hadir_a_all, ignore_index=True)
    p_obs_pool = (hadir_p_pool == hadir_a_pool).mean()
    k_pool = hitung_kappa_aman(hadir_p_pool, hadir_a_pool)
    pe_pool = hitung_pe(p_obs_pool, k_pool)
    hasil_aspek_list.append({
        "aspek": "OVERALL (pooled 5 aspek)",
        "n_komentar": len(hadir_p_pool),
        "percent_agreement": round(p_obs_pool, 3),
        "chance_agreement": round(pe_pool, 3) if pe_pool == pe_pool else "n/a",
        "cohen_kappa": round(k_pool, 3) if k_pool == k_pool else "n/a",
        "interpretasi": interpretasi_kappa(k_pool),
    })

    sent_p_pool = pd.concat(sent_p_all, ignore_index=True) if sent_p_all else pd.Series(dtype="Int64")
    sent_a_pool = pd.concat(sent_a_all, ignore_index=True) if sent_a_all else pd.Series(dtype="Int64")
    if len(sent_p_pool) > 0:
        p_obs_sp = (sent_p_pool == sent_a_pool).mean()
        k_sp = hitung_kappa_aman(sent_p_pool, sent_a_pool)
        pe_sp = hitung_pe(p_obs_sp, k_sp)
        hasil_sentimen_list.append({
            "aspek": "OVERALL (pooled 5 aspek)",
            "n_komentar_disepakati_dibahas": len(sent_p_pool),
            "percent_agreement": round(p_obs_sp, 3),
            "chance_agreement": round(pe_sp, 3) if pe_sp == pe_sp else "n/a",
            "cohen_kappa": round(k_sp, 3) if k_sp == k_sp else "n/a",
            "interpretasi": interpretasi_kappa(k_sp),
        })

    df_aspek = pd.DataFrame(hasil_aspek_list)
    df_sentimen = pd.DataFrame(hasil_sentimen_list)

    print("=" * 66)
    print(" AGREEMENT DETEKSI ASPEK (dibahas / tidak dibahas)")
    print("=" * 66)
    print(df_aspek.to_string(index=False), "\n")

    print("=" * 66)
    print(" AGREEMENT SENTIMEN (hanya pada aspek yang disepakati dibahas)")
    print("=" * 66)
    print(df_sentimen.to_string(index=False), "\n")

    df_aspek.to_csv(OUTPUT_ASPEK, index=False)
    df_sentimen.to_csv(OUTPUT_SENTIMEN, index=False)
    print(f"Hasil deteksi aspek disimpan ke : {OUTPUT_ASPEK}")
    print(f"Hasil sentimen disimpan ke      : {OUTPUT_SENTIMEN}")
    print(f"Confusion matrix per aspek disimpan sebagai file terpisah "
          f"('{CM_PREFIX_ASPEK}_*.csv' dan '{CM_PREFIX_SENTIMEN}_*.csv').")


if __name__ == "__main__":
    main()