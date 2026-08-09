"""
Unit Test Metrik Evaluasi (pipeline/evaluate_model.py)
======================================================
Regression test untuk perhitungan Sentiment F1: macro harus dihitung hanya
atas kelas sentimen (Neg/Neu/Pos). Sebelumnya, prediksi None pada sampel
aktif ikut masuk sebagai kelas ekstra ber-F1 0 (karena sklearn memakai
gabungan kelas y_true dan y_pred saat labels= tidak diberikan), sehingga
macro dibagi 4 dan nilai maksimum efektif turun ke 0.75.

Jalankan: pytest tests/
"""

import pytest

from pipeline.evaluate_model import NONE_IDX, _compute_split_metrics
from preprocessing.preprocessing_functions import FINAL_ASPECTS, convert_labels


def _single_aspect_case(aspect, y_true, y_pred):
    """Bangun dict prediksi/label lengkap semua aspek; aspek selain `aspect`
    diisi satu sampel None sehingga tidak berkontribusi pada rata-rata
    tertimbang sentiment F1 (bobot n_sent = 0)."""
    all_labels = {a: [NONE_IDX[a]] for a in FINAL_ASPECTS}
    all_preds  = {a: [NONE_IDX[a]] for a in FINAL_ASPECTS}
    all_labels[aspect] = y_true
    all_preds[aspect]  = y_pred
    return all_preds, all_labels


def test_sentiment_f1_mengabaikan_kelas_none_pada_prediksi():
    # 6 sampel aktif; separuh diprediksi None (index 3).
    # Per kelas sentimen: P=1.0, R=0.5 → F1=2/3; macro atas 3 kelas = 2/3.
    # Dengan bug lama, None ikut jadi kelas ke-4 → (3 × 2/3 + 0) / 4 = 0.5.
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 3, 3, 3]
    all_preds, all_labels = _single_aspect_case('Content Quality', y_true, y_pred)

    _, sentiment_f1, _, avg_sentiment = _compute_split_metrics(all_preds, all_labels)

    assert sentiment_f1['Content Quality'] == pytest.approx(2 / 3)
    assert avg_sentiment == pytest.approx(2 / 3)


def test_sentiment_f1_sempurna_bernilai_satu():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 2, 2]
    all_preds, all_labels = _single_aspect_case('Content Quality', y_true, y_pred)

    _, sentiment_f1, _, avg_sentiment = _compute_split_metrics(all_preds, all_labels)

    assert sentiment_f1['Content Quality'] == pytest.approx(1.0)
    assert avg_sentiment == pytest.approx(1.0)


def test_detection_f1_tetap_menyertakan_kelas_none():
    # Detection F1 memang macro atas semua kelas termasuk None — pastikan
    # perbaikan sentiment F1 tidak mengubah perilaku ini.
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    all_preds, all_labels = _single_aspect_case('Content Quality', y_true, y_pred)

    detect_f1, _, _, _ = _compute_split_metrics(all_preds, all_labels)

    assert detect_f1['Content Quality'] == pytest.approx(1.0)


def test_convert_labels_netral_pada_aspek_biner_dipetakan_ke_none():
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'Technical & Access': [-1, 0, 1, None]})

    result = convert_labels(df)

    # Skema biner: -1 → 0 (Neg), 1 → 1 (Pos), NaN → 2 (None);
    # anotasi Netral (0) yang tersisa di data mentah juga → 2 (None).
    assert result['lbl_Technical & Access'].tolist() == [0, 2, 1, 2]
