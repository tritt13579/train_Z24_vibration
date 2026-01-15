# z24_class_split_70_30.py
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
    end_t: int = 64000
    window: int = 2048
    gap: int = 256
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


def _all_samples_for_class(files: List[str], y: int, cfg: Z24Cfg) -> List[Tuple[str, int, int]]:
    starts = _starts(cfg)
    return [(fp, y, s) for fp in files for s in starts]


def _split_samples_70_30(samples: List[Tuple[str, int, int]], rng: random.Random):
    n = len(samples)
    if n < 2:
        raise ValueError(f"Need >=2 samples for 70/30 split, got n={n}")
    samples = samples[:]
    rng.shuffle(samples)

    n_train = int(round(0.70 * n))
    n_train = max(1, min(n_train, n - 1))

    tr = samples[:n_train]
    va = samples[n_train:]
    return tr, va


def _mean_std_from_train(train_index: List[Tuple[str, int, int]], cfg: Z24Cfg):
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
    def __init__(self, index: List[Tuple[str, int, int]], cfg: Z24Cfg,
                 mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
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

    tr_idx = []
    va_idx = []
    per_class = {}

    for y, cls in enumerate(classes):
        files = _list_mat_files(cfg, cls)
        all_samples = _all_samples_for_class(files, y, cfg)
        tr, va = _split_samples_70_30(all_samples, rng)
        tr_idx += tr
        va_idx += va
        per_class[cls] = {"files": len(files), "samples": len(all_samples), "train": len(tr), "val": len(va)}

    mean = std = None
    if cfg.normalize:
        mean, std = _mean_std_from_train(tr_idx, cfg)

    tr_ds = Z24Windows(tr_idx, cfg, mean, std) if cfg.normalize else Z24Windows(tr_idx, cfg)
    va_ds = Z24Windows(va_idx, cfg, mean, std) if cfg.normalize else Z24Windows(va_idx, cfg)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "windows_per_file": len(_starts(cfg)),
        "train_samples": len(tr_idx),
        "val_samples": len(va_idx),
        "per_class": per_class,
        "mean": mean,
        "std": std,
    }
    return tr_loader, va_loader, meta
