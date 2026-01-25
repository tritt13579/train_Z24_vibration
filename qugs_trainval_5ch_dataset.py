# QUGS train/val data processing (first 5 channels only, no time column)
import os
import random
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =========================
# CONFIG
# =========================
@dataclass
class QUGS5Cfg:
    root: str = "./QUGS"
    train_subdir: str = "Dataset A"
    val_subdir: str = "Dataset B"

    window: int = 2048
    step: int = 2048  # non-overlap by default

    channels: int = 5  # use first 5 channels (no time column)
    normalize: bool = True
    eps: float = 1e-8
    return_ct: bool = True  # return shape (C, T) instead of (T, C)

    cache_files: int = 2  # max files cached in memory
    # augmentation (Z24-style)
    noise_ks: tuple = (0.03, 0.05, 0.07)
    # val subset (use only a fraction of Dataset B)
    val_fraction: float = 0.5
    val_seed: int = 42


# =========================
# UTIL
# =========================
_DATA_LINE_RE = re.compile(r"^\s*[+-]?\d")


def _is_data_line(line: str) -> bool:
    return bool(_DATA_LINE_RE.match(line))


def _detect_header(fp: str) -> Tuple[int, int]:
    header_lines = 0
    first_data_cols = 0
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if _is_data_line(line):
                first_data_cols = len(line.strip().split())
                break
            header_lines += 1

    if first_data_cols == 0:
        raise ValueError(f"No numeric data found in: {fp}")

    return header_lines, first_data_cols


def _load_qugs_file(fp: str, cfg: QUGS5Cfg, meta_cache: Dict) -> np.ndarray:
    if fp in meta_cache:
        data = meta_cache[fp]["data"]
        return data

    header_lines, ncols = _detect_header(fp)
    raw = np.loadtxt(fp, skiprows=header_lines)
    if raw.ndim != 2 or raw.shape[1] != ncols:
        raise ValueError(f"Unexpected data shape in {fp}: {raw.shape}")

    if raw.shape[1] < 1 + cfg.channels:
        raise ValueError(f"Not enough channels in {fp}: have {raw.shape[1]-1}, need {cfg.channels}")

    # drop time column, keep first N channels
    data = raw[:, 1 : 1 + cfg.channels].astype(np.float32)

    meta_cache[fp] = {
        "data": data,
        "n_samples": data.shape[0],
        "n_channels": data.shape[1],
    }
    return data


def _list_txt_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing dir: {root}")
    files = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"No .txt in: {root}")
    return files


def _extract_class_name(fp: str) -> str:
    name = os.path.splitext(os.path.basename(fp))[0]
    # expected patterns: zzzAD1, zzzAU, zzzBD2, zzzBU...
    m = re.search(r"(D\d+|U)$", name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse class from filename: {fp}")
    return m.group(1).upper()


def _class_sort_key(cls: str):
    if cls == "U":
        return (0, 0)
    if cls.startswith("D"):
        try:
            return (1, int(cls[1:]))
        except ValueError:
            return (1, 9999)
    return (2, cls)


def _build_label_map(train_files: List[str], val_files: List[str]) -> Dict[str, int]:
    classes = sorted({*_extract_class_name(fp) for fp in train_files + val_files}, key=_class_sort_key)
    return {cls: i for i, cls in enumerate(classes)}


def _subset_starts(starts: List[int], fraction: float, rng: random.Random) -> List[int]:
    if fraction >= 1.0:
        return starts
    if fraction <= 0.0:
        return []
    n_keep = max(1, int(np.ceil(len(starts) * fraction)))
    starts = starts[:]
    rng.shuffle(starts)
    return starts[:n_keep]


def _build_index(
    files: List[str],
    cfg: QUGS5Cfg,
    meta_cache: Dict,
    label_map: Dict[str, int],
    augment: bool,
    val_fraction: float = 1.0,
    rng: random.Random = None,
):
    index = []
    for fp in files:
        data = _load_qugs_file(fp, cfg, meta_cache)
        n_samples = data.shape[0]
        cls = _extract_class_name(fp)
        y = label_map[cls]
        if n_samples < cfg.window:
            continue
        starts = list(range(0, n_samples - cfg.window + 1, cfg.step))
        if not augment and val_fraction < 1.0:
            if rng is None:
                rng = random.Random(0)
            starts = _subset_starts(starts, val_fraction, rng)
        for s in starts:
            if augment:
                index.append((fp, y, s, "clean"))
                for k in cfg.noise_ks:
                    index.append((fp, y, s, f"noise_{k:.2f}"))
                index.append((fp, y, s, "reverse"))
            else:
                index.append((fp, y, s, "clean"))
    return index


def _mean_std_from_train(index, cfg: QUGS5Cfg, meta_cache: Dict) -> Tuple[np.ndarray, np.ndarray]:
    sum_c = None
    sumsq_c = None
    count = 0

    starts_by_file = defaultdict(list)
    for fp, _, s, aug_type in index:
        if aug_type != "clean":
            continue
        starts_by_file[fp].append(s)

    for fp, starts in starts_by_file.items():
        data = _load_qugs_file(fp, cfg, meta_cache).astype(np.float64)
        if sum_c is None:
            sum_c = np.zeros((data.shape[1],), dtype=np.float64)
            sumsq_c = np.zeros((data.shape[1],), dtype=np.float64)

        for s in starts:
            w = data[s : s + cfg.window]
            sum_c += w.sum(axis=0)
            sumsq_c += (w * w).sum(axis=0)
            count += w.shape[0]

    if count == 0:
        raise ValueError("No training samples found to compute mean/std")

    mean = sum_c / count
    var = sumsq_c / count - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


# =========================
# DATASET
# =========================
class QUGS5Windows(Dataset):
    def __init__(self, index, cfg: QUGS5Cfg, mean=None, std=None):
        self.index = index
        self.cfg = cfg
        self.mean = mean
        self.std = std

        self._meta_cache: Dict[str, Dict] = OrderedDict()

        if cfg.normalize and (mean is None or std is None):
            raise ValueError("normalize=True requires mean/std computed from TRAIN")

    def __len__(self):
        return len(self.index)

    def _get_data(self, fp: str) -> np.ndarray:
        if fp in self._meta_cache:
            data = self._meta_cache.pop(fp)
            self._meta_cache[fp] = data
            return data["data"]

        data = _load_qugs_file(fp, self.cfg, self._meta_cache)
        # enforce LRU size
        while len(self._meta_cache) > self.cfg.cache_files:
            self._meta_cache.popitem(last=False)
        return data

    def __getitem__(self, i):
        fp, y, s, aug_type = self.index[i]
        x = self._get_data(fp)
        w = x[s : s + self.cfg.window]

        if aug_type.startswith("noise_"):
            if self.std is None:
                raise ValueError("Noise augmentation requires std from training data")
            k = float(aug_type.split("_", 1)[1])
            noise = np.random.normal(0.0, k * self.std[None, :], size=w.shape).astype(np.float32)
            w = w + noise
        elif aug_type == "reverse":
            w = w[::-1]
        elif aug_type != "clean":
            raise ValueError(f"Unknown aug_type: {aug_type}")

        if self.cfg.normalize:
            w = (w - self.mean[None, :]) / (self.std[None, :] + self.cfg.eps)

        if self.cfg.return_ct:
            w = w.T  # (C,T)

        return torch.from_numpy(w), torch.tensor(y, dtype=torch.long)


# =========================
# BUILD LOADERS
# =========================
def make_loaders(cfg: QUGS5Cfg, batch_size=32, num_workers=0):
    train_root = os.path.join(cfg.root, cfg.train_subdir)
    val_root = os.path.join(cfg.root, cfg.val_subdir)

    train_files = _list_txt_files(train_root)
    val_files = _list_txt_files(val_root)

    label_map = _build_label_map(train_files, val_files)
    meta_cache = {}

    train_index = _build_index(train_files, cfg, meta_cache, label_map, augment=True)
    val_rng = random.Random(cfg.val_seed)
    val_index = _build_index(
        val_files,
        cfg,
        meta_cache,
        label_map,
        augment=False,
        val_fraction=cfg.val_fraction,
        rng=val_rng,
    )

    mean, std = _mean_std_from_train(train_index, cfg, meta_cache)

    train_ds = QUGS5Windows(train_index, cfg, mean, std)
    val_ds = QUGS5Windows(val_index, cfg, mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    meta = {
        "classes": {v: k for k, v in label_map.items()},
        "train_files": len(train_files),
        "val_files": len(val_files),
        "train_samples": len(train_index),
        "val_samples": len(val_index),
        "window": cfg.window,
        "step": cfg.step,
        "channels": cfg.channels,
        "mean": mean,
        "std": std,
        "noise_ks": list(cfg.noise_ks),
        "val_fraction": cfg.val_fraction,
    }

    return train_loader, val_loader, meta
