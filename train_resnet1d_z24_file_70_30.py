# train_resnet1d_z24_file_70_30.py
import os
os.environ["PYTHONHASHSEED"] = "4"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import json
import random
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from z24_file_split_70_30_on_the_fly import Z24Cfg, make_loaders


# =========================
# CONFIG
# =========================
CFG_USE_AMP = False          # AMP only on CUDA
GLOBAL_SEED = 4
SPLIT_SEED = 42

# Dataset config
DATASET_ROOT = "./DatasetPDT"
DATASET_SUBDIR = "avt/Processed"
DATASET_KEY = "data_last5"

# Output base dir (each run will create ./History/v1, v2, ...)
HISTORY_DIR = "./History"


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
# ResNet1D
# ============================================================
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1DClassifier(nn.Module):
    """
    Input from dataset: [B, T, C]
    internal conv:      [B, C, T]
    Head: GAP -> Linear
    """
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 5,
        layers=(2, 2, 2, 2),
        base_channels: int = 32,
        stem_kernel_size: int = 7,
        block_kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must have 4 elements, e.g. (2,2,2,2)")

        self.in_channels = base_channels
        self.conv1 = nn.Conv1d(
            input_channels, base_channels,
            kernel_size=stem_kernel_size, stride=2,
            padding=stem_kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_channels, layers[0], block_kernel_size, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], block_kernel_size, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], block_kernel_size, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], block_kernel_size, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 8, num_classes)

    def _make_layer(self, out_channels, blocks, kernel_size, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, kernel_size, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels, kernel_size, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def train_resnet1d_file_split_70_30(
    classes,
    batch_size=32,
    epochs=60,
    learning_rate=3e-4,

    layers=(2, 2, 2, 2),
    base_channels=32,
    stem_kernel_size=7,
    block_kernel_size=3,
    dropout=0.2,

    weight_decay=1e-4,
    num_workers=4,
):
    """
    IMPORTANT:
    - Augmentation is handled INSIDE dataset z24_file_split_70_30.py (OFFLINE expanded index):
      clean + noise_{0.03,0.05,0.07} + reverse
    - VAL uses clean only
    - So DO NOT add any extra noise augmentation here.
    """
    set_global_determinism(GLOBAL_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MAIN] Using device: {device}")
    if device.type == "cuda":
        print(f"[MAIN] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[MAIN] CUDA: {torch.version.cuda}")

    # -------------------------
    # Versioned output (no overwrite)
    # -------------------------
    run_dir, version = make_versioned_run_dir(HISTORY_DIR, prefix="v")
    ckpt_path = os.path.join(run_dir, f"resnet1d_z24_file_70_30_best_{version}.pt")
    print(f"[RUN] output_dir={run_dir}")
    print(f"[RUN] ckpt_path={ckpt_path}")

    # Dataset config (match your dataset file)
    cfg = Z24Cfg(
        root=DATASET_ROOT,
        subdir=DATASET_SUBDIR,
        key=DATASET_KEY,
        channels=5,
        end_t=64000,
        window=2048,
        seed=SPLIT_SEED,
        normalize=True,
        return_ct=False,           # dataset returns [T, C] for model
        noise_ks=(0.03, 0.05, 0.07)
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
    print(f"[AUG] OFFLINE dataset aug: clean + noise{tuple(meta.get('noise_ks', []))} + reverse (VAL clean only)")
    print(f"[DATA] windows_per_file={meta.get('windows_per_file')} "
          f"train_samples={meta.get('train_samples')} val_samples={meta.get('val_samples')} "
          f"train_files={meta.get('train_files')} val_files={meta.get('val_files')}")

    # Save run_info.json
    run_info = {
        "version": version,
        "classes": classes,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "layers": list(layers),
        "base_channels": base_channels,
        "stem_kernel_size": stem_kernel_size,
        "block_kernel_size": block_kernel_size,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "num_workers": num_workers,
        "split_seed": SPLIT_SEED,
        "global_seed": GLOBAL_SEED,
        "dataset": {
            "root": DATASET_ROOT,
            "subdir": DATASET_SUBDIR,
            "key": DATASET_KEY,
            "end_t": cfg.end_t,
            "window": cfg.window,
            "no_overlap": True,
            "augmentation": {
                "offline_index_expansion": True,
                "noise_ks": list(cfg.noise_ks),
                "reverse": True,
                "val_clean_only": True,
            }
        }
    }
    with open(os.path.join(run_dir, f"run_info_{version}.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    model = ResNet1DClassifier(
        num_classes=num_classes,
        input_channels=cfg.channels,
        layers=layers,
        base_channels=base_channels,
        stem_kernel_size=stem_kernel_size,
        block_kernel_size=block_kernel_size,
        dropout=dropout,
    ).to(device)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = -1
    last_val_true = None
    last_val_pred = None

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
        last_val_true, last_val_pred = val_true, val_pred

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
                    "resnet1d_params": {
                        "layers": list(layers),
                        "base_channels": base_channels,
                        "stem_kernel_size": stem_kernel_size,
                        "block_kernel_size": block_kernel_size,
                        "dropout": dropout,
                        "weight_decay": weight_decay,
                    },
                },
                ckpt_path,
            )
            print(f"[CKPT] Saved best -> {ckpt_path} (epoch={best_epoch}, val_loss={best_val_loss:.4f})")

        scheduler.step(val_loss)

    with open(os.path.join(run_dir, f"resnet1d_z24_file_70_30_history_{version}.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    plot_history(history, os.path.join(run_dir, f"resnet1d_z24_file_70_30_curves_{version}.png"))

    if last_val_true is not None and last_val_pred is not None:
        plot_confusion_matrix(
            last_val_true,
            last_val_pred,
            num_classes,
            os.path.join(run_dir, f"resnet1d_z24_file_70_30_val_cm_{version}.png"),
            title=f"VAL Confusion Matrix (ResNet1D 70/30 file split) - {version}",
        )

    print(f"[DONE] best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f}")
    print(f"[SAVED] {run_dir} (history + plots + ckpt)")
    return model, history


if __name__ == "__main__":
    train_resnet1d_file_split_70_30(
        classes=['01', '03', '04', '05', '06'],
        batch_size=32,
        epochs=50,
        learning_rate=5e-4,

        layers=(2, 2, 2, 2),
        base_channels=32,
        stem_kernel_size=7,
        block_kernel_size=3,
        dropout=0.3,

        weight_decay=1e-4,
        num_workers=4,
    )
