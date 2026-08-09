# Catatan: Kenapa Sentiment/Detection F1 Cuma ~58, dan Apa yang Diperbaiki

Branch: `fix/metric-sentiment-f1`

Ringkasan singkat: **angka ~58 itu bukan karena modelmu jelek** — ada satu bug
di perhitungan Sentiment F1 yang menekan angkanya ke bawah secara sistematis,
ditambah karakteristik dataset (kecil + sangat imbalanced) yang membuat
macro-F1 memang sulit tinggi. Branch ini memperbaiki yang bisa diperbaiki di
kode. **Model perlu di-train ulang setelah fix ini** (alasannya di bawah).

---

## 1. Bug utama: Sentiment F1 menghitung kelas None sebagai "kelas hantu"

Lokasi: `pipeline/evaluate_model.py`, fungsi `_compute_split_metrics()`.

Kode lama:

```python
mask = y_true != none_i
sentiment_f1[asp] = f1_score(y_true[mask], y_pred[mask],
                             average='macro', zero_division=0)
```

Mask hanya memfilter **label asli** (gold) yang bukan None — tapi **prediksi**
model masih bisa berupa None. Kalau parameter `labels=` tidak diberikan,
sklearn otomatis memakai gabungan kelas yang muncul di `y_true` **dan**
`y_pred`. Akibatnya, begitu model memprediksi None pada satu saja sampel yang
sebenarnya aktif (dan ini pasti sering, karena None adalah kelas mayoritas
62–93%), kelas None ikut dihitung sebagai kelas ke-4 dengan F1 = 0 selamanya
(tidak pernah ada gold None di dalam mask, jadi precision-nya pasti 0).

Konsekuensinya: macro-F1 dibagi 4, bukan 3 — **nilai maksimum efektif metrik
ini langsung turun ke 0.75**. Ilustrasi dengan angka nyata (bisa dicoba
sendiri):

```python
from sklearn.metrics import f1_score
y_true = [0, 1, 2, 0, 1, 2]        # semua sampel aktif
y_pred = [0, 1, 2, 3, 3, 3]        # separuh diprediksi None

f1_score(y_true, y_pred, average='macro')                   # 0.50  <- lama
f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro') # 0.667 <- baru
```

Jadi kalau performa sentimen riil model sekitar 70-an, yang terlapor bisa
jadi cuma 50-an. Angka 58 kemarin kemungkinan besar merepresentasikan model
yang sebenarnya jauh lebih baik dari itu.

Fix: tambahkan `labels=list(range(none_i))` sehingga macro hanya dihitung
atas kelas sentimen (Neg/Neu/Pos, atau Neg/Pos untuk Technical & Access).
Kesalahan model memprediksi None pada aspek yang disebut **tetap terhukum** —
lewat recall kelas asli, dan lewat Pair-based Micro-F1 yang dihitung terpisah.

### ⚠️ Kenapa harus train ulang

Metrik ini juga dipakai untuk **early stopping dan pemilihan best checkpoint**
di `pipeline/train_model.py` (`avg_sent > best_sent`). Artinya model
"terbaik" yang tersimpan sekarang dipilih berdasarkan metrik yang terdistorsi.
Setelah fix, jalankan ulang training supaya checkpoint yang terpilih memang
optimal menurut metrik yang benar.

### ⚠️ Angka lama vs baru tidak sebanding

Untuk laporan TA: angka Sentiment F1 lama (0.54–0.58) **tidak bisa
dibandingkan langsung** dengan angka setelah fix — definisi metriknya
berubah. Kalau butuh perbandingan antar-eksperimen lama vs baru, pakai
**Pair-based Micro-F1** dan **Aspect Detection F1** (sudah dihitung di
`compute_pooled_metrics()`) — keduanya tidak berubah, dan keduanya juga
metrik standar di literatur ABSA (Schmitt et al., Li et al.), jadi cocok
dijadikan metrik headline di laporan.

---

## 2. Class weighting dimatikan padahal data sangat imbalanced

Distribusi label `data/raw/dataset_final.csv` (1.389 komentar):

| Aspek                  | Aktif | Neg | Neu | Pos | None (%) |
|------------------------|------:|----:|----:|----:|---------:|
| Content Quality        |   525 |  86 |  63 | 376 |      62% |
| Subscription & Pricing |   175 |  90 |  43 |  42 |      87% |
| UI/UX                  |   137 |  47 |  42 |  48 |      90% |
| Functionality          |    96 |  32 |  30 |  34 |      93% |
| Technical & Access     |   229 | 185 |  5* |  39 |      84% |

Config baseline (`configs/experiment_indobert_baseline.yaml`) memakai
`use_class_weights: false`. Dengan None 62–93%, model tanpa class weighting
cenderung "main aman" memprediksi None — recall kelas sentimen jatuh, dan
(dikombinasikan dengan bug di atas) angkanya makin tertekan.

Fix: `use_class_weights: true`. Bobotnya sudah dihitung otomatis dari
training set oleh `compute_class_weights()`.

## 3. (*) Anotasi Netral pada aspek biner dibuang diam-diam

`Technical & Access` skemanya biner (Neg/Pos), tapi di data mentah masih ada
5 baris beranotasi Netral (nilai 0). `to_binary()` lama memetakannya ke None
lewat fallback `.get(..., 2)` tanpa ada yang sadar. Sekarang pemetaannya
eksplisit (`0 → None`) plus komentar. Idealnya 5 baris ini dibersihkan di
sumber data: dianotasi ulang jadi Neg/Pos, atau dihapus anotasinya.

---

## Apa saja yang berubah di branch ini

| File | Perubahan |
|---|---|
| `pipeline/evaluate_model.py` | Sentiment F1: `labels=` eksplisit, macro hanya atas kelas sentimen |
| `configs/experiment_indobert_baseline.yaml` | `use_class_weights: true` + deskripsi diperbarui |
| `preprocessing/preprocessing_functions.py` | `to_binary()`: Netral → None eksplisit, bukan fallback |
| `tests/test_evaluate_metrics.py` | Baru — regression test metrik (4 test, semua lulus) |

Yang TIDAK berubah: Detection F1 (memang by design macro atas semua kelas
termasuk None), Pair-based Micro-F1, Aspect Detection F1, arsitektur model,
dan alur training.

## Langkah selanjutnya (saran)

1. **Train ulang** dengan config yang sudah diperbaiki, bandingkan angkanya.
2. Perlakukan macro Sentiment F1 dengan hati-hati di laporan: test set 15%
   dari 1.389 komentar berarti per aspek cuma ada **±4–9 sampel per kelas
   sentimen** (Functionality: ~5 Neg, ~4 Neu, ~5 Pos). Satu sampel salah
   klasifikasi bisa menggeser F1 aspek itu 10+ poin. Pertimbangkan lapor
   interval ketidakpastian (bootstrap pada test set).
3. Kelas Netral cuma 30–63 sampel di hampir semua aspek — ini penyeret
   terbesar macro-F1. Opsi: tambah data/augmentasi khusus Netral (sudah
   mulai di `ABSA_gt3_augmented.csv`), atau gabungkan/hapus kelas Netral
   untuk aspek yang sampelnya belasan.
4. Bersihkan 5 baris Netral di `Technical & Access` di sumber data.
