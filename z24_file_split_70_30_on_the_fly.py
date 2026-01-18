# z24_file_split_70_30.py
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.io
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
    return_ct: bool = True  # True: (C,T), False: (T,C)

    # =========================
    # ON-THE-FLY AUGMENT (TRAIN ONLY)
    # =========================
    aug_enable: bool = True

    # time shift (roll) on time axis
    p_shift: float = 0.7
    shift_max: int = 128  # shift in [-128, +128]

    # reverse in time axis
    p_reverse: float = 0.2

    # amplitude scaling (global)
    p_scale: float = 0.5
    scale_min: float = 0.9
    scale_max: float = 1.1

    # gaussian noise
    p_noise: float = 0.8
    noise_k_min: float = 0.03
    noise_k_max: float = 0.07
    noise_bias_power: float = 2.0  # u**2 -> bias toward small noise


# =========================
# UTIL
# =========================
def _starts(cfg: Z24Cfg) -> List[int]:
    """
    Non-overlapping windows:
      sample1: 0..2047
      sample2: 2048..4095
      ...
    Drop last if incomplete automatically by range stop.
    """
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
    """
    Return window shape (window, C) = (2048, 5)
    """
    mat = scipy.io.loadmat(fp)
    if cfg.key not in mat:
        raise KeyError(f"Missing key '{cfg.key}' in {fp}. keys={list(mat.keys())}")

    x = np.asarray(mat[cfg.key])  # expected (T,C)
    if x.ndim != 2:
        raise ValueError(f"{fp}: expected 2D array (T,C), got {x.shape}")

    if x.shape[1] != cfg.channels:
        raise ValueError(f"{fp}: expected C={cfg.channels}, got {x.shape[1]}")

    if x.shape[0] < cfg.end_t:
        raise ValueError(f"{fp}: T={x.shape[0]} < end_t={cfg.end_t}")

    end = start + cfg.window
    if end > cfg.end_t:
        raise ValueError(f"{fp}: end={end} > end_t={cfg.end_t}")

    w = x[start:end, :]  # (window,C)
    if w.shape[0] != cfg.window:
        raise ValueError(f"{fp}: window length mismatch {w.shape[0]} != {cfg.window}")

    return w.astype(np.float32)


def _split_files_70_30(files: List[str], rng: random.Random) -> Tuple[List[str], List[str]]:
    """
    File-level split to avoid leakage.
    """
    n = len(files)
    if n < 2:
        raise ValueError(f"Need >=2 files for 70/30 split, got n={n}")

    files = files[:]
    rng.shuffle(files)

    n_train = int(round(0.70 * n))
    n_train = max(1, min(n_train, n - 1))
    tr = files[:n_train]
    va = files[n_train:]
    return tr, va


def _build_index(files_with_label: List[Tuple[str, int]], cfg: Z24Cfg) -> List[Tuple[str, int, int]]:
    """
    Index stores ONLY clean windows:
      (fp, label, start)
    Augmentation happens on-the-fly in Dataset (train only).
    """
    starts = _starts(cfg)
    return [(fp, y, s) for fp, y in files_with_label for s in starts]


def _mean_std_from_train(train_index: List[Tuple[str, int, int]], cfg: Z24Cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std using TRAIN CLEAN windows only.
    Over all time steps and all windows.
    """
    sum_c = np.zeros((cfg.channels,), dtype=np.float64)
    sumsq_c = np.zeros((cfg.channels,), dtype=np.float64)
    count = 0

    for fp, _, s in train_index:
        w = _load_window(fp, s, cfg).astype(np.float64)  # (T,C)
        sum_c += w.sum(axis=0)
        sumsq_c += (w * w).sum(axis=0)
        count += w.shape[0]

    mean = sum_c / max(count, 1)
    var = sumsq_c / max(count, 1) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))

    return mean.astype(np.float32), std.astype(np.float32)


def _seed_worker(worker_id: int):
    """
    Make random/np random different per dataloader worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================
# DATASET
# =========================
class Z24Windows(Dataset):
    def __init__(
        self,
        index: List[Tuple[str, int, int]],
        cfg: Z24Cfg,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        is_train: bool = False,
    ):
        self.index = index  # (fp, y, start)
        self.cfg = cfg
        self.mean = mean
        self.std = std
        self.is_train = is_train

        if cfg.normalize and (mean is None or std is None):
            raise ValueError("normalize=True requires mean/std computed from TRAIN")

    def __len__(self) -> int:
        return len(self.index)

    def _sample_noise_k(self) -> float:
        """
        Sample k in [k_min, k_max], biased toward small.
        k = k_min + (k_max-k_min) * (u ** noise_bias_power)
        """
        u = random.random()
        u = u ** float(self.cfg.noise_bias_power)
        return float(self.cfg.noise_k_min + (self.cfg.noise_k_max - self.cfg.noise_k_min) * u)

    def _augment_train(self, w: np.ndarray) -> np.ndarray:
        """
        On-the-fly augmentation (TRAIN ONLY).
        Order:
          1) time shift (roll)
          2) reverse
          3) amplitude scaling
          4) gaussian noise
        """
        # 1) time shift
        if self.cfg.p_shift > 0 and random.random() < self.cfg.p_shift:
            shift = random.randint(-int(self.cfg.shift_max), int(self.cfg.shift_max))
            if shift != 0:
                w = np.roll(w, shift=shift, axis=0)

        # 2) reverse
        if self.cfg.p_reverse > 0 and random.random() < self.cfg.p_reverse:
            w = w[::-1].copy()

        # 3) amplitude scaling (global)
        if self.cfg.p_scale > 0 and random.random() < self.cfg.p_scale:
            a = random.uniform(float(self.cfg.scale_min), float(self.cfg.scale_max))
            w = (w * a).astype(np.float32)

        # 4) gaussian noise (per-channel std from TRAIN)
        if self.cfg.p_noise > 0 and random.random() < self.cfg.p_noise:
            k = self._sample_noise_k()
            noise = np.random.normal(
                loc=0.0,
                scale=k * self.std[None, :],   # (1,C)
                size=w.shape
            ).astype(np.float32)
            w = w + noise

        return w.astype(np.float32)

    def __getitem__(self, i: int):
        fp, y, s = self.index[i]
        w = _load_window(fp, s, self.cfg)  # (T,C)

        # augment only for TRAIN
        if self.is_train and self.cfg.aug_enable:
            w = self._augment_train(w)

        # normalize
        if self.cfg.normalize:
            w = (w - self.mean[None, :]) / (self.std[None, :] + self.cfg.eps)

        # output format
        if self.cfg.return_ct:
            w = w.T  # (C,T)

        return torch.from_numpy(w), torch.tensor(y, dtype=torch.long)


# =========================
# BUILD LOADERS
# =========================
def make_loaders(classes: List[str], cfg: Z24Cfg, batch_size: int = 32, num_workers: int = 0):
    rng = random.Random(cfg.seed)

    tr_files: List[Tuple[str, int]] = []
    va_files: List[Tuple[str, int]] = []
    per_class: Dict[str, Dict[str, int]] = {}

    for y, cls in enumerate(classes):
        files = _list_mat_files(cfg, cls)
        tr, va = _split_files_70_30(files, rng)

        tr_files += [(fp, y) for fp in tr]
        va_files += [(fp, y) for fp in va]

        per_class[cls] = {
            "files": len(files),
            "train_files": len(tr),
            "val_files": len(va),
        }

    tr_index = _build_index(tr_files, cfg)
    va_index = _build_index(va_files, cfg)

    mean = std = None
    if cfg.normalize:
        mean, std = _mean_std_from_train(tr_index, cfg)

    tr_ds = Z24Windows(tr_index, cfg, mean, std, is_train=True)
    va_ds = Z24Windows(va_index, cfg, mean, std, is_train=False)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    meta = {
        "windows_per_file": len(_starts(cfg)),
        "train_files": len(tr_files),
        "val_files": len(va_files),
        "train_samples": len(tr_index),
        "val_samples": len(va_index),
        "per_class": per_class,
        "mean": mean,
        "std": std,
        "augment": {
            "aug_enable": cfg.aug_enable,
            "p_shift": cfg.p_shift,
            "shift_max": cfg.shift_max,
            "p_reverse": cfg.p_reverse,
            "p_scale": cfg.p_scale,
            "scale_min": cfg.scale_min,
            "scale_max": cfg.scale_max,
            "p_noise": cfg.p_noise,
            "noise_k_min": cfg.noise_k_min,
            "noise_k_max": cfg.noise_k_max,
            "noise_bias_power": cfg.noise_bias_power,
        },
    }

    return tr_loader, va_loader, meta
