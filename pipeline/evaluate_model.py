import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from preprocessing.preprocessing_functions import FINAL_ASPECTS, NUM_CLASSES, LABEL_NAMES

NONE_IDX = {asp: NUM_CLASSES[asp] - 1 for asp in FINAL_ASPECTS}


def _asp_key(asp: str) -> str:
    """Konversi nama aspek ke format aman untuk MLflow metric key."""
    return (
        asp.replace(' ', '_')
           .replace('&', 'and')
           .replace('/', '_')
           .lower()
    )


def _eval_loop(model, loader, device):
    """
    Loop evaluasi inti — dapat digunakan saat validasi per-epoch maupun evaluasi
    final pada test set.

    Returns
    -------
    detect_f1    : dict  — macro F1 semua kelas (termasuk None) per aspek
    sentiment_f1 : dict  — macro F1 kelas sentimen saja (tidak termasuk None) per aspek
    avg_detect   : float — rata-rata tertimbang detect_f1
    avg_sentiment: float — rata-rata tertimbang sentiment_f1
    all_preds    : dict  — prediksi per aspek
    all_labels   : dict  — label asli per aspek
    """
    model.eval()
    all_preds  = {a: [] for a in FINAL_ASPECTS}
    all_labels = {a: [] for a in FINAL_ASPECTS}

    with torch.no_grad():
        for batch in loader:
            iids  = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            logits = model(iids, amask)
            for asp in FINAL_ASPECTS:
                preds  = logits[asp].argmax(dim=-1).cpu().numpy()
                labels = batch['labels'][asp].numpy()
                all_preds[asp].extend(preds.tolist())
                all_labels[asp].extend(labels.tolist())

    detect_f1    = {}
    sentiment_f1 = {}

    for asp in FINAL_ASPECTS:
        y_true = np.array(all_labels[asp])
        y_pred = np.array(all_preds[asp])
        none_i = NONE_IDX[asp]

        # Detection F1: semua kelas termasuk None
        detect_f1[asp] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Sentiment F1: hanya sampel dengan label bukan None
        mask = y_true != none_i
        sentiment_f1[asp] = (
            f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
            if mask.sum() > 0 else 0.0
        )

    n_det  = [len(all_labels[a]) for a in FINAL_ASPECTS]
    n_sent = [sum(1 for l in all_labels[a] if l != NONE_IDX[a]) for a in FINAL_ASPECTS]

    avg_detect    = float(np.average(list(detect_f1.values()), weights=n_det))
    avg_sentiment = (
        float(np.average(list(sentiment_f1.values()), weights=n_sent))
        if sum(n_sent) > 0 else 0.0
    )

    return detect_f1, sentiment_f1, avg_detect, avg_sentiment, all_preds, all_labels


def compute_test_metrics(model, tokenizer, config: dict, data: dict) -> dict:
    """
    Hitung metrik test set (flat, format sama seperti evaluate_model()) tanpa
    menulis classification_report.txt / confusion_matrix.png ke disk.

    Dipakai untuk re-evaluasi model baseline yang diunduh dari MLflow Model
    Registry pada test set yang sama dengan model baru (lihat Uji 3 di
    workflow/stages/validate_model.py) — perbandingan lewat metrik tersimpan
    saja tidak valid begitu dataset training berubah antar siklus retraining,
    karena test split ikut berbeda.
    """
    from torch.utils.data import DataLoader
    from model.absa_model import ABSADataset, ABSACollator

    rep_cfg = config['representation']
    params  = config['model']['params']
    device  = next(model.parameters()).device

    test_ds     = ABSADataset(data['df_test'], tokenizer, rep_cfg['max_length'], FINAL_ASPECTS)
    collator    = ABSACollator(tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=params['batch_size'],
        shuffle=False, num_workers=0, collate_fn=collator,
    )

    det_f1, sent_f1, avg_det, avg_sent, _, _ = _eval_loop(model, test_loader, device)

    metrics = {
        'test_mean_detect_f1'   : avg_det,
        'test_mean_sentiment_f1': avg_sent,
    }
    for asp in FINAL_ASPECTS:
        k = _asp_key(asp)
        metrics[f'test_{k}_detect_f1']    = det_f1[asp]
        metrics[f'test_{k}_sentiment_f1'] = sent_f1[asp]
    return metrics


def evaluate_model(config: dict, trained: dict, data: dict) -> dict:
    """
    Evaluasi model pada test set dan kembalikan metrik dalam format flat
    yang siap di-log ke MLflow.

    Metrik yang dikembalikan:
      test_mean_detect_f1        — rata-rata Detection F1 semua aspek
      test_mean_sentiment_f1     — rata-rata Sentiment F1 semua aspek (metrik utama)
      test_{asp}_detect_f1       — Detection F1 per aspek
      test_{asp}_sentiment_f1    — Sentiment F1 per aspek
    """
    model_type = config['model']['type']
    if model_type == 'indobert_multitask':
        return _evaluate_indobert(config, trained, data)
    raise ValueError(f"Model type tidak didukung: {model_type}")


def _evaluate_indobert(config: dict, trained: dict, data: dict) -> dict:
    from torch.utils.data import DataLoader
    from model.absa_model import ABSADataset, ABSACollator

    rep_cfg = config['representation']
    params  = config['model']['params']
    model     = trained['model']      # best model state (sudah dimuat di train_model)
    tokenizer = trained['tokenizer']
    device    = trained['device']

    test_ds     = ABSADataset(data['df_test'], tokenizer, rep_cfg['max_length'], FINAL_ASPECTS)
    collator    = ABSACollator(tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=params['batch_size'],
        shuffle=False, num_workers=0, collate_fn=collator,
    )

    det_f1, sent_f1, avg_det, avg_sent, all_preds, all_labels = _eval_loop(
        model, test_loader, device
    )

    # ── Cetak laporan ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HASIL EVALUASI TEST SET")
    print(f"{'='*60}")
    print(f"{'Aspek':<30} {'Detection':>10} {'Sentimen':>10}")
    print(f"{'─'*52}")
    for asp in FINAL_ASPECTS:
        print(f"{asp:<30} {det_f1[asp]:>10.4f} {sent_f1[asp]:>10.4f}")
    print(f"{'─'*52}")
    print(f"{'Rata-rata (tertimbang)':<30} {avg_det:>10.4f} {avg_sent:>10.4f}")
    print(f"\n  Sentiment F1 adalah metrik utama keberhasilan model.")

    print(f"\n── Classification Report per Aspek ──")
    for asp in FINAL_ASPECTS:
        y_true  = np.array(all_labels[asp])
        y_pred  = np.array(all_preds[asp])
        n_aktif = (y_true != NONE_IDX[asp]).sum()
        print(f"\n{asp}  (n_total={len(y_true)}, n_aktif={n_aktif})")
        # labels= eksplisit agar tetap konsisten dengan target_names meskipun
        # tidak semua kelas muncul di test set (mis. test set kecil)
        print(classification_report(
            y_true, y_pred, labels=list(range(len(LABEL_NAMES[asp]))),
            target_names=LABEL_NAMES[asp], zero_division=0,
        ))

    # ── Simpan classification report ke file ──────────────────────────
    save_dir = trained['save_dir']
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w', encoding='utf-8') as fout:
        fout.write("Classification Report per Aspek (Test Set)\n")
        fout.write("=" * 60 + "\n\n")
        for asp in FINAL_ASPECTS:
            y_true  = np.array(all_labels[asp])
            y_pred  = np.array(all_preds[asp])
            n_aktif = (y_true != NONE_IDX[asp]).sum()
            fout.write(f"{asp}  (n_total={len(y_true)}, n_aktif={n_aktif})\n")
            fout.write(classification_report(
                y_true, y_pred, labels=list(range(len(LABEL_NAMES[asp]))),
                target_names=LABEL_NAMES[asp], zero_division=0,
            ))
            fout.write("\n")

    # ── Confusion matrix per aspek ─────────────────────────────────────
    n_cols = 3
    n_rows = (len(FINAL_ASPECTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for i, asp in enumerate(FINAL_ASPECTS):
        y_true = np.array(all_labels[asp])
        y_pred = np.array(all_preds[asp])
        n_cls  = len(LABEL_NAMES[asp])
        cm     = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))

        ax = axes[i]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(asp, fontsize=9, pad=4)
        ticks = list(range(n_cls))
        ax.set_xticks(ticks)
        ax.set_xticklabels(LABEL_NAMES[asp], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(LABEL_NAMES[asp], fontsize=8)
        ax.set_xlabel('Prediksi', fontsize=8)
        ax.set_ylabel('Aktual',   fontsize=8)
        thresh = cm.max() / 2.
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center', fontsize=9,
                        color='white' if cm[r, c] > thresh else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Confusion Matrix per Aspek — Test Set', fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Metrik flat untuk MLflow ────────────────────────────────────────
    metrics = {
        'test_mean_detect_f1'   : avg_det,
        'test_mean_sentiment_f1': avg_sent,
    }
    for asp in FINAL_ASPECTS:
        k = _asp_key(asp)
        metrics[f'test_{k}_detect_f1']    = det_f1[asp]
        metrics[f'test_{k}_sentiment_f1'] = sent_f1[asp]

    return metrics
