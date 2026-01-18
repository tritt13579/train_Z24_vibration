# z24_file_split_70_30.py
import os
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.io
from scipy.signal import resample
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# CONFIG
# =========================
@dataclass
class Z24Cfg:
    root: str = "./DatasetPDT"
    subdir: str = "avt/Processed"
    key: str = "data_last5"

    channels: int = 5
    end_t: int = 64000
    window: int = 2048
    seed: int = 42

    normalize: bool = True
    eps: float = 1e-8
    return_ct: bool = True

    # augmentation (paper-style)
    # Create 3 noisy variants per window with k in [0.03, 0.05, 0.07]
    noise_ks: tuple = (0.03, 0.05, 0.07)

    crop_ratio_min: float = 0.85
    crop_ratio_max: float = 0.95


# =========================
# UTIL
# =========================
def _starts(cfg: Z24Cfg) -> List[int]:
    # non-overlapping windows, drop last if incomplete
    return list(range(0, cfg.end_t - cfg.window + 1, cfg.window))


def _class_dir(cfg: Z24Cfg, cls: str) -> str:
    return os.path.join(cfg.root, cls, cfg.subdir)


def _list_mat_files(cfg: Z24Cfg, cls: str) -> List[str]:
    d = _class_dir(cfg, cls)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing dir: {d}")
    files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".mat")]
    if not files:
        raise FileNotFoundError(f"No .mat in: {d}")
    return files


def _load_window(fp: str, start: int, cfg: Z24Cfg) -> np.ndarray:
    mat = scipy.io.loadmat(fp)
    if cfg.key not in mat:
        raise KeyError(f"Missing key '{cfg.key}' in {fp}")
    x = np.asarray(mat[cfg.key])  # (T,C)

    end = start + cfg.window
    w = x[start:end, :]
    return w.astype(np.float32)


def _split_files_70_30(files: List[str], rng: random.Random):
    files = files[:]
    rng.shuffle(files)
    n = len(files)
    n_train = max(1, min(int(round(0.7 * n)), n - 1))
    return files[:n_train], files[n_train:]


def _mean_std_from_train(train_index, cfg: Z24Cfg):
    """
    train_index contains (fp, y, start, aug_type)
    We compute mean/std from TRAIN windows (clean only) for stability.
    (Augmented variants are derived from the same windows.)
    """
    sum_c = np.zeros((cfg.channels,), dtype=np.float64)
    sumsq_c = np.zeros((cfg.channels,), dtype=np.float64)
    count = 0

    for fp, _, s, aug_type in train_index:
        if aug_type != "clean":
            continue
        w = _load_window(fp, s, cfg).astype(np.float64)
        sum_c += w.sum(axis=0)
        sumsq_c += (w * w).sum(axis=0)
        count += w.shape[0]

    mean = sum_c / max(count, 1)
    var = sumsq_c / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


# =========================
# DATASET
# =========================
class Z24Windows(Dataset):
    def __init__(self, index, cfg: Z24Cfg, mean=None, std=None, is_train=False):
        self.index = index  # (fp, y, start, aug_type)
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.is_train = is_train

        if cfg.normalize and (mean is None or std is None):
            raise ValueError("normalize=True requires mean/std computed from TRAIN")

    def __len__(self):
        return len(self.index)

    def _crop_and_resample(self, w: np.ndarray):
        T = w.shape[0]
        ratio = random.uniform(self.cfg.crop_ratio_min, self.cfg.crop_ratio_max)
        crop_len = max(2, int(T * ratio))  # ensure >=2

        start = random.randint(0, T - crop_len)
        cropped = w[start:start + crop_len, :]

        resized = resample(cropped, T, axis=0)
        return resized.astype(np.float32)

    def __getitem__(self, i):
        fp, y, s, aug_type = self.index[i]
        w = _load_window(fp, s, self.cfg)  # (T,C)

        # ---- augment (TRAIN index only; VAL uses clean only) ----
        if aug_type.startswith("noise_"):
            # aug_type = "noise_0.03" / "noise_0.05" / "noise_0.07"
            try:
                k = float(aug_type.split("_", 1)[1])
            except Exception as e:
                raise ValueError(f"Bad aug_type for noise: {aug_type}") from e

            noise = np.random.normal(
                0.0,
                k * self.std[None, :],
                size=w.shape
            ).astype(np.float32)
            w = w + noise

        elif aug_type == "reverse":
            w = w[::-1]

        elif aug_type == "crop":
            w = self._crop_and_resample(w)

        elif aug_type == "clean":
            pass
        else:
            raise ValueError(f"Unknown aug_type: {aug_type}")

        # normalize
        if self.cfg.normalize:
            w = (w - self.mean[None, :]) / (self.std[None, :] + self.cfg.eps)

        if self.cfg.return_ct:
            w = w.T  # (C,T)

        return torch.from_numpy(w), torch.tensor(y, dtype=torch.long)


# =========================
# BUILD LOADERS
# =========================
def _build_index(files_with_label, cfg: Z24Cfg, augment: bool):
    starts = _starts(cfg)
    index = []

    for fp, y in files_with_label:
        for s in starts:
            if augment:
                index.append((fp, y, s, "clean"))

                # 3 noisy variants
                for k in cfg.noise_ks:
                    index.append((fp, y, s, f"noise_{k:.2f}"))

                # reverse + crop
                index.append((fp, y, s, "reverse"))
                index.append((fp, y, s, "crop"))
            else:
                index.append((fp, y, s, "clean"))

    return index


def make_loaders(classes, cfg: Z24Cfg, batch_size=32, num_workers=0):
    rng = random.Random(cfg.seed)

    tr_files = []
    va_files = []

    for y, cls in enumerate(classes):
        files = _list_mat_files(cfg, cls)
        tr, va = _split_files_70_30(files, rng)
        tr_files += [(fp, y) for fp in tr]
        va_files += [(fp, y) for fp in va]

    # build indices
    tr_index = _build_index(tr_files, cfg, augment=True)
    va_index = _build_index(va_files, cfg, augment=False)

    # compute mean/std from TRAIN clean windows only
    mean, std = _mean_std_from_train(tr_index, cfg)

    tr_ds = Z24Windows(tr_index, cfg, mean, std, is_train=True)
    va_ds = Z24Windows(va_index, cfg, mean, std, is_train=False)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "windows_per_file": len(_starts(cfg)),
        "train_files": len(tr_files),
        "val_files": len(va_files),
        "train_samples": len(tr_index),
        "val_samples": len(va_index),
        "mean": mean,
        "std": std,
        "noise_ks": list(cfg.noise_ks),
    }

    return tr_loader, va_loader, meta
