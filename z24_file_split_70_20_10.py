# z24_file_split_70_20_10.py
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Z24Cfg:
    root: str = "./DatasetPDT"
    subdir: str = "avt/Processed"
    key: str = "data_last5"

    channels: int = 5
    end_t: int = 58000
    window: int = 8000
    gap: int = 2000
    seed: int = 42

    normalize: bool = True
    eps: float = 1e-8
    return_ct: bool = True


def _starts(cfg: Z24Cfg) -> List[int]:
    step = cfg.window + cfg.gap
    return list(range(0, cfg.end_t - cfg.window + 1, step))


def _list_mat_files(cfg: Z24Cfg, cls: str) -> List[str]:
    d = os.path.join(cfg.root, cls, cfg.subdir)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing dir: {d}")
    files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".mat")]
    if not files:
        raise FileNotFoundError(f"No .mat in: {d}")
    return files


def _load_window(fp: str, start: int, cfg: Z24Cfg) -> np.ndarray:
    mat = scipy.io.loadmat(fp)
    if cfg.key not in mat:
        raise KeyError(f"Missing key '{cfg.key}' in {fp}. keys={list(mat.keys())}")

    x = np.asarray(mat[cfg.key])  # (T,C)
    if x.ndim != 2:
        raise ValueError(f"{fp}: expected 2D, got {x.shape}")
    if x.shape[1] != cfg.channels:
        raise ValueError(f"{fp}: expected C={cfg.channels}, got {x.shape[1]}")
    if x.shape[0] < cfg.end_t:
        raise ValueError(f"{fp}: T={x.shape[0]} < end_t={cfg.end_t}")

    end = start + cfg.window
    w = x[start:end, :]
    return w.astype(np.float32)


def _split_counts_70_20_10(n: int) -> Tuple[int, int, int]:
    """
    Deterministic count logic, non-empty splits.
    - n=9 -> 6,2,1
    - n=8 -> 6,1,1
    """
    if n < 3:
        raise ValueError(f"Need >=3 files for 70/20/10, got n={n}")

    n_train = int(round(0.70 * n))
    n_train = max(1, min(n_train, n - 2))  # must leave at least 2 for val+test
    remain = n - n_train

    n_val = int(round(0.20 * n))
    n_val = max(1, min(n_val, remain - 1))  # must leave >=1 for test
    n_test = remain - n_val

    if n_test < 1:
        # safety fallback
        n_test = 1
        n_val = remain - 1

    return n_train, n_val, n_test


def _split_files_70_20_10(files: List[str], rng: random.Random):
    files = files[:]
    rng.shuffle(files)
    n = len(files)
    n_tr, n_va, n_te = _split_counts_70_20_10(n)

    tr = files[:n_tr]
    va = files[n_tr:n_tr + n_va]
    te = files[n_tr + n_va:n_tr + n_va + n_te]
    return tr, va, te


def _build_index(files_with_label: List[Tuple[str, int]], cfg: Z24Cfg):
    starts = _starts(cfg)
    return [(fp, y, s) for fp, y in files_with_label for s in starts]


def _mean_std_from_train(train_index, cfg: Z24Cfg):
    sum_c = np.zeros((cfg.channels,), dtype=np.float64)
    sumsq_c = np.zeros((cfg.channels,), dtype=np.float64)
    count = 0

    for fp, _, s in train_index:
        w = _load_window(fp, s, cfg).astype(np.float64)
        sum_c += w.sum(axis=0)
        sumsq_c += (w * w).sum(axis=0)
        count += w.shape[0]

    mean = sum_c / max(count, 1)
    var = sumsq_c / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


class Z24Windows(Dataset):
    def __init__(self, index, cfg: Z24Cfg, mean=None, std=None):
        self.index = index
        self.cfg = cfg
        self.mean = mean
        self.std = std
        if cfg.normalize and (mean is None or std is None):
            raise ValueError("normalize=True requires mean/std computed from TRAIN")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        fp, y, s = self.index[i]
        w = _load_window(fp, s, self.cfg)
        if self.cfg.normalize:
            w = (w - self.mean[None, :]) / (self.std[None, :] + self.cfg.eps)
        if self.cfg.return_ct:
            w = w.T
        return torch.from_numpy(w), torch.tensor(y, dtype=torch.long)


def make_loaders(classes: List[str], cfg: Z24Cfg, batch_size: int = 32, num_workers: int = 0):
    rng = random.Random(cfg.seed)

    tr_files = []
    va_files = []
    te_files = []
    per_class = {}

    for y, cls in enumerate(classes):
        files = _list_mat_files(cfg, cls)
        tr, va, te = _split_files_70_20_10(files, rng)

        tr_files += [(fp, y) for fp in tr]
        va_files += [(fp, y) for fp in va]
        te_files += [(fp, y) for fp in te]

        per_class[cls] = {
            "files": len(files),
            "train_files": len(tr),
            "val_files": len(va),
            "test_files": len(te),
        }

    tr_index = _build_index(tr_files, cfg)
    va_index = _build_index(va_files, cfg)
    te_index = _build_index(te_files, cfg)

    mean = std = None
    if cfg.normalize:
        mean, std = _mean_std_from_train(tr_index, cfg)

    tr_ds = Z24Windows(tr_index, cfg, mean, std) if cfg.normalize else Z24Windows(tr_index, cfg)
    va_ds = Z24Windows(va_index, cfg, mean, std) if cfg.normalize else Z24Windows(va_index, cfg)
    te_ds = Z24Windows(te_index, cfg, mean, std) if cfg.normalize else Z24Windows(te_index, cfg)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "windows_per_file": len(_starts(cfg)),
        "train_files": len(tr_files),
        "val_files": len(va_files),
        "test_files": len(te_files),
        "train_samples": len(tr_index),
        "val_samples": len(va_index),
        "test_samples": len(te_index),
        "per_class": per_class,
        "mean": mean,
        "std": std,
    }
    return tr_loader, va_loader, te_loader, meta
