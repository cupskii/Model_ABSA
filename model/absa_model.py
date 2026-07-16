import os, json, pickle, re, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


#  SEED 
def set_seed(seed: int) -> None:
    """Seed semua sumber randomness untuk reprodusibilitas eksperimen.

    Mencakup: Python random, NumPy, PyTorch CPU & CUDA, cuDNN determinism,
    dan PYTHONHASHSEED. Panggil sebelum inisialisasi model dan DataLoader.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── KONFIGURASI ───────────────────────────────────────────────────────────────
CONFIG = {
    'model_name'   : 'indobenchmark/indobert-base-p1',
    'max_length'   : 128,
    'dropout_rate' : 0.17366466880936898,
    'batch_size'   : 8,
    'learning_rate': 4.801650729207848e-05,
    'num_epochs'   : 25,
    'warmup_ratio' : 0.05366045312634687,
    'weight_decay' : 0.039008177788135416,
    'max_grad_norm': 1.0,
    'patience'     : 5,
    'save_dir'     : './model_output_v2',
}

ASPECTS = [
    'Content Quality',
    'Subscription & Pricing',
    'UI/UX',
    'Functionality',
    'Technical & Access',
]

# Skema: 0=Neg | 1=Neu | 2=Pos | 3=None
NUM_CLASSES = {
    'Content Quality'       : 4,
    'Subscription & Pricing': 4,
    'UI/UX'                 : 4,
    'Functionality'         : 4,
    'Technical & Access'    : 3,
}

LABEL_NAMES = {
    'Content Quality'       : ['Negatif', 'Netral', 'Positif', 'None'],
    'Subscription & Pricing': ['Negatif', 'Netral', 'Positif', 'None'],
    'UI/UX'                 : ['Negatif', 'Netral', 'Positif', 'None'],
    'Functionality'         : ['Negatif', 'Netral', 'Positif', 'None'],
    'Technical & Access'    : ['Negatif', 'Positif', 'None'],
}

# None index per aspek (indeks kelas None)
NONE_IDX = {asp: NUM_CLASSES[asp] - 1 for asp in ASPECTS}

# Class weights dari training set — urutan: Neg / Neu / Pos / None
# (untuk Technical & Access: Neg / Pos / None)
# Perbarui nilai ini dari class_weights.json hasil data_preperation.py
# Class weights — None dikecilkan ke 0.10 agar model lebih fokus ke sentimen
CLASS_WEIGHTS = {
      "Content Quality": [
    7.165,
    12.35344827586207,
    1.1482371794871795,
    0.34380998080614206
  ],
  "Subscription & Pricing": [
    5.045774647887324,
    18.855263157894736,
    21.073529411764707,
    0.2701734539969834
  ],
  "UI/UX": [
    8.331395348837209,
    35.825,
    16.28409090909091,
    0.26380706921944036
  ],
  "Functionality": [
    6.513636363636364,
    17.05952380952381,
    32.56818181818182,
    0.2661589895988113
  ],
  "Technical & Access": [
    3.9153005464480874,
    25.140350877192983,
    0.3697110423116615
  ]
}

SLANG_DICT = {
    'gak':'tidak','ga':'tidak','nggak':'tidak','ngga':'tidak','gk':'tidak',
    'tdk':'tidak','yg':'yang','dgn':'dengan','dg':'dengan','utk':'untuk',
    'krn':'karena','karna':'karena','klo':'kalau','kl':'kalau',
    'udah':'sudah','udh':'sudah','sdh':'sudah','blm':'belum',
    'lg':'lagi','lgs':'langsung','sy':'saya','bgt':'banget',
    'bs':'bisa','bsa':'bisa','tp':'tapi','ttp':'tetap','jd':'jadi',
    'app':'aplikasi','apps':'aplikasi','apk':'aplikasi',
    'ok':'oke','subs':'berlangganan',
    'langgan':'berlangganan','langgannya':'berlangganannya',
    'langganan':'berlangganan','berlangganannya':'berlangganannya',
    'harga':'harga','mahal':'mahal',
    'epaper':'e-paper','e paper':'e-paper',
}


# ── DATASET ───────────────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_length, aspects):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length= max_length
        self.aspects   = aspects
        self.label_cols= [f'lbl_{a}' for a in aspects]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row['komentar_clean'])

        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True)

        labels = {}
        for col, aspect in zip(self.label_cols, self.aspects):
            labels[aspect] = torch.tensor(int(row[col]), dtype=torch.long)

        return {
            'input_ids'     : encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels'        : labels,
        }


class ABSACollator:
    def __init__(self, tokenizer):
        self._base = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    def __call__(self, features):
        text_feats  = [{'input_ids': f['input_ids'],
                        'attention_mask': f['attention_mask']}
                       for f in features]
        labels_list = [f['labels'] for f in features]
        batch = self._base(text_feats)
        batch['labels'] = {
            asp: torch.stack([l[asp] for l in labels_list])
            for asp in labels_list[0]
        }
        return batch


# ── MODEL ─────────────────────────────────────────────────────────────────────
class ABSAModel(nn.Module):
    def __init__(self, model_name, aspects, num_classes, dropout_rate):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden        = self.encoder.config.hidden_size   # 768
        self.dropout  = nn.Dropout(dropout_rate)
        self.heads    = nn.ModuleDict({
            asp.replace(' ','_').replace('&','and'):
                nn.Linear(hidden, num_classes[asp])
            for asp in aspects
        })
        self.aspects = aspects

    def _key(self, asp):
        return asp.replace(' ','_').replace('&','and')

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {asp: self.heads[self._key(asp)](cls) for asp in self.aspects}


# ── LOSS ──────────────────────────────────────────────────────────────────────
def compute_loss(logits_dict, labels_dict, class_weights_dict, device):
    total = torch.tensor(0.0, device=device)
    for asp, logits in logits_dict.items():
        labels  = labels_dict[asp].to(device)
        weights = torch.tensor(class_weights_dict[asp],
                               dtype=torch.float32, device=device)
        total  += F.cross_entropy(logits, labels, weight=weights)
    return total / len(logits_dict)
