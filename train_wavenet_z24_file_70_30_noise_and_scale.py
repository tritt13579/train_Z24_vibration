# train_wavenet_z24_file_70_30_noise_and_scale.py
import os
os.environ["PYTHONHASHSEED"] = "4"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import json
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from z24_file_split_70_30 import Z24Cfg, make_loaders
# from z24_class_split_70_30 import Z24Cfg, make_loaders


# =========================
# CONFIG
# =========================
CFG_USE_AMP = False         # AMP only on CUDA
CFG_SANITY_OVERFIT = False   # overfit 1 batch check
GLOBAL_SEED = 4
SPLIT_SEED = 42

# Dataset config
DATASET_ROOT = "./DatasetPDT"
DATASET_SUBDIR = "avt/Processed"
DATASET_KEY = "data_last5"

# Output base dir
BASE_HISTORY_DIR = "./History"
RUN_GROUP = "noise_and_scale"  # results saved to ./History/noise_and_scale/v1, v2, ...


# =========================
# UTIL
# =========================
def set_global_determinism(seed: int = 4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _counts(arr: np.ndarray):
    d = {}
    for v in arr.tolist():
        d[int(v)] = d.get(int(v), 0) + 1
    return dict(sorted(d.items(), key=lambda x: x[0]))


def make_versioned_run_dir(base_dir: str, prefix: str = "v"):
    """
    Create a new versioned directory: base_dir/v1, v2, ...
    Returns (run_dir, version_str)
    """
    os.makedirs(base_dir, exist_ok=True)

    existing = []
    for name in os.listdir(base_dir):
        if name.startswith(prefix):
            num_part = name[len(prefix):]
            if num_part.isdigit():
                existing.append(int(num_part))

    next_v = (max(existing) + 1) if existing else 1
    version = f"{prefix}{next_v}"
    run_dir = os.path.join(base_dir, version)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, version


def plot_history(history, out_base_png):
    epochs = np.arange(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base_png.replace(".png", "_loss.png"), dpi=160)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_base_png.replace(".png", "_acc.png"), dpi=160)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, num_classes, out_path_png, title="Confusion Matrix"):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=160)
    plt.close()


@torch.no_grad()
def eval_loop(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)

        correct += (pred == yb).sum().item()
        total += xb.size(0)

        all_true.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    all_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    all_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, all_true, all_pred


# ============================================================
# AUGMENTATIONS (TRAIN ONLY)
# ============================================================
def apply_amplitude_scaling(
    x: torch.Tensor,
    p: float = 0.7,
    scale_min: float = 0.90,
    scale_max: float = 1.10,
) -> torch.Tensor:
    """
    x: [B, T, C]
    - scaling factor is SAME for all channels of each sample (physically reasonable for Z24)
    - applied per-sample with prob p
    """
    if p <= 0:
        return x

    B = x.size(0)
    device = x.device
    dtype = x.dtype

    mask = (torch.rand((B, 1, 1), device=device) < p)
    scales = torch.empty((B, 1, 1), device=device, dtype=dtype).uniform_(scale_min, scale_max)
    scales = torch.where(mask, scales, torch.ones_like(scales))
    return x * scales


def add_gaussian_noise(x: torch.Tensor, p: float = 0.5, std_ratio: float = 0.01) -> torch.Tensor:
    """
    x: [B, T, C]
    - apply per-sample with prob p
    - noise std per sample = std_ratio * std(x_sample over (T,C))
    """
    if p <= 0:
        return x

    B = x.size(0)
    device = x.device

    mask = (torch.rand((B, 1, 1), device=device) < p)
    s = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-12)

    noise = torch.randn_like(x) * (std_ratio * s)
    noise = torch.where(mask, noise, torch.zeros_like(noise))
    return x + noise


# ============================================================
# WaveNet in PyTorch (causal conv + gated residual + skip)
# ============================================================
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0, bias=bias
        )

    def forward(self, x):
        pad_left = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, (pad_left, 0))
        return self.conv(x)


class WaveNetResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv_tanh = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv_sig  = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv_1x1   = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tanh_out = torch.tanh(self.conv_tanh(x))
        sigm_out = torch.sigmoid(self.conv_sig(x))
        gated = tanh_out * sigm_out
        gated = self.dropout(gated)
        out = self.conv_1x1(gated)
        res = out + x
        skip = out
        return res, skip


class WaveNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 5,
        num_filters: int = 64,
        kernel_size: int = 2,
        number_of_blocks: int = 3,
        residuals_per_block: int = 8,
        downsample_factor: int = 32,
        pool_type: str = "avg",
        dropout: float = 0.0,
    ):
        super().__init__()

        if downsample_factor > 1:
            if pool_type == "avg":
                self.pool = nn.AvgPool1d(downsample_factor, downsample_factor)
            elif pool_type == "max":
                self.pool = nn.MaxPool1d(downsample_factor, downsample_factor)
            else:
                raise ValueError("pool_type must be 'avg' or 'max'")
        else:
            self.pool = None

        self.in_proj = nn.Conv1d(input_channels, num_filters, kernel_size=1)

        blocks = []
        total_layers = number_of_blocks * residuals_per_block
        for i in range(total_layers):
            k = i % residuals_per_block
            dilation = 2 ** k
            blocks.append(WaveNetResidualBlock(num_filters, kernel_size, dilation, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.head_relu = nn.ReLU()
        self.head_conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.head_conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]

        if self.pool is not None:
            x = self.pool(x)

        x = self.in_proj(x)

        skips = None
        for blk in self.blocks:
            x, skip = blk(x)
            skips = skip if skips is None else (skips + skip)

        x = self.head_relu(skips)
        x = F.relu(self.head_conv1(x))
        x = F.relu(self.head_conv2(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)


def sanity_overfit_one_batch(model, train_loader, device, steps=200, lr=5e-3):
    print("\n==============================")
    print("[SANITY] Overfit 1 batch check")
    print("==============================")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    it = iter(train_loader)
    xb, yb = next(it)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = logits.argmax(dim=1)
        acc = (pred == yb).float().mean().item()

        if step == 1 or step % 20 == 0 or step == steps:
            print(f"[SANITY] step {step:04d}/{steps} - loss:{loss.item():.4f} - acc:{acc:.4f}")

    y_true = yb.detach().cpu().numpy()
    y_pred = model(xb).argmax(dim=1).detach().cpu().numpy()
    print(f"[SANITY] batch y_true: {_counts(y_true)}")
    print(f"[SANITY] batch y_pred: {_counts(y_pred)}")
    print("[SANITY] Done.\n")


def train_wavenet_file_split_70_30(
    classes,
    batch_size=32,
    epochs=50,
    learning_rate=1e-3,

    num_filters=64,
    kernel_size=2,
    number_of_blocks=3,
    residuals_per_block=8,

    downsample_factor=32,
    pool_type="avg",

    dropout=0.0,
    weight_decay=0.0,

    num_workers=0,

    # --- amplitude scaling (TRAIN) ---
    scaling_p: float = 0.7,
    scale_min: float = 0.90,
    scale_max: float = 1.10,

    # --- gaussian noise (TRAIN) ---
    noise_p: float = 0.5,
    noise_std_ratio: float = 0.01,
):
    set_global_determinism(GLOBAL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Using device: {device}")
    if device.type == "cuda":
        print(f"[MAIN] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[MAIN] CUDA: {torch.version.cuda}")

    # -------------------------
    # Versioned output under group dir
    # -------------------------
    group_dir = os.path.join(BASE_HISTORY_DIR, RUN_GROUP)
    run_dir, version = make_versioned_run_dir(group_dir, prefix="v")

    ckpt_path = os.path.join(run_dir, f"wavenet_best_{version}.pt")
    print(f"[RUN] group_dir={group_dir}")
    print(f"[RUN] output_dir={run_dir}")
    print(f"[RUN] ckpt_path={ckpt_path}")

    run_info = {
        "run_group": RUN_GROUP,
        "version": version,
        "classes": classes,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_filters": num_filters,
        "kernel_size": kernel_size,
        "number_of_blocks": number_of_blocks,
        "residuals_per_block": residuals_per_block,
        "downsample_factor": downsample_factor,
        "pool_type": pool_type,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "scaling_p": scaling_p,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "noise_p": noise_p,
        "noise_std_ratio": noise_std_ratio,
        "split_seed": SPLIT_SEED,
        "global_seed": GLOBAL_SEED,
    }
    with open(os.path.join(run_dir, f"run_info_{version}.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    cfg = Z24Cfg(
        root=DATASET_ROOT,
        subdir=DATASET_SUBDIR,
        key=DATASET_KEY,
        channels=5,
        end_t=64000,
        window=2048,
        gap=256,
        seed=SPLIT_SEED,
        normalize=True,
        return_ct=False,
    )

    train_loader, val_loader, meta = make_loaders(
        classes=classes,
        cfg=cfg,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    num_classes = len(classes)

    use_amp = (device.type == "cuda") and CFG_USE_AMP
    print(f"[CFG] AMP enabled? {use_amp} (CFG_USE_AMP={CFG_USE_AMP})")
    print(f"[AUG] scaling: p={scaling_p}, range=[{scale_min}, {scale_max}] (TRAIN only)")
    print(f"[AUG] noise:   p={noise_p}, std_ratio={noise_std_ratio} (TRAIN only)")
    print("[AUG] order: scaling -> noise")

    model = WaveNetClassifier(
        num_classes=num_classes,
        input_channels=cfg.channels,
        num_filters=num_filters,
        kernel_size=kernel_size,
        number_of_blocks=number_of_blocks,
        residuals_per_block=residuals_per_block,
        downsample_factor=downsample_factor,
        pool_type=pool_type,
        dropout=dropout,
    ).to(device)

    print(model)
    print(f"[DATA] windows_per_file={meta.get('windows_per_file')}, train_samples={meta.get('train_samples')}, val_samples={meta.get('val_samples')}")

    if CFG_SANITY_OVERFIT:
        sanity_overfit_one_batch(model, train_loader, device, steps=200, lr=5e-3)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        train_true_all = []
        train_pred_all = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # TRAIN augmentation: BOTH (scaling -> noise)
            xb = apply_amplitude_scaling(xb, p=scaling_p, scale_min=scale_min, scale_max=scale_max)
            xb = add_gaussian_noise(xb, p=noise_p, std_ratio=noise_std_ratio)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)

            train_true_all.append(yb.detach().cpu().numpy())
            train_pred_all.append(pred.detach().cpu().numpy())

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        train_true_all = np.concatenate(train_true_all) if train_true_all else np.array([], dtype=np.int64)
        train_pred_all = np.concatenate(train_pred_all) if train_pred_all else np.array([], dtype=np.int64)

        val_loss, val_acc, val_true, val_pred = eval_loop(model, val_loader, criterion, device)

        history["loss"].append(train_loss)
        history["acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{epochs} "
            f"- loss:{train_loss:.4f} acc:{train_acc:.4f} "
            f"- val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}"
        )

        print(f"[TRAIN] y_true counts: {_counts(train_true_all)}")
        print(f"[TRAIN] y_pred counts: {_counts(train_pred_all)}")
        print(f"[VAL]   y_true counts: {_counts(val_true)}")
        print(f"[VAL]   y_pred counts: {_counts(val_pred)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "cfg": cfg.__dict__,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "wavenet_params": {
                        "num_filters": num_filters,
                        "kernel_size": kernel_size,
                        "number_of_blocks": number_of_blocks,
                        "residuals_per_block": residuals_per_block,
                        "downsample_factor": downsample_factor,
                        "pool_type": pool_type,
                        "dropout": dropout,
                        "scaling_p": scaling_p,
                        "scale_min": scale_min,
                        "scale_max": scale_max,
                        "noise_p": noise_p,
                        "noise_std_ratio": noise_std_ratio,
                    },
                },
                ckpt_path,
            )
            print(f"[CKPT] Saved best -> {ckpt_path} (epoch={best_epoch}, val_loss={best_val_loss:.4f})")

        scheduler.step(val_loss)

    with open(os.path.join(run_dir, f"history_{version}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_history(history, os.path.join(run_dir, f"curves_{version}.png"))
    plot_confusion_matrix(
        val_true,
        val_pred,
        num_classes,
        os.path.join(run_dir, f"val_cm_{version}.png"),
        title=f"VAL Confusion Matrix - {RUN_GROUP} - {version}",
    )

    print(f"[DONE] best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
    print(f"[SAVED] {run_dir} (history + plots + ckpt)")
    return model, history


if __name__ == "__main__":
    train_wavenet_file_split_70_30(
        classes=['01', '03', '04', '05', '06'],
        batch_size=64,
        epochs=150,
        learning_rate=5e-4,

        num_filters=128,
        kernel_size=3,
        number_of_blocks=4,
        residuals_per_block=10,

        downsample_factor=1,
        pool_type="avg",

        dropout=0.3,
        weight_decay=1e-4,

        num_workers=4,

        # ----------------------------
        # Z24-friendly augmentation
        # ----------------------------
        scaling_p=0.7,
        scale_min=0.90,
        scale_max=1.10,

        noise_p=0.5,
        noise_std_ratio=0.01,
    )
