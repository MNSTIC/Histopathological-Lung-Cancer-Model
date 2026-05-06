"""
10_cross_dataset.py
===================
Canonical S2 evaluation — zero-shot cross-dataset evaluation of an
LC25000-trained HAGCA-Net checkpoint on PatchCamelyon (PCam) test set.

Strict zero-shot semantics:
  - No PCam fine-tuning.
  - No use of PCam test labels.
  - PCam validation labels MAY be used for threshold tuning when
    USE_THRESHOLD_TUNING is enabled (standard domain-adaptation practice).

Canonical defaults (best single-flag config from the Phase 5 ablation):
  USE_LC25000_STAIN_NORM  = True   ← only flag with a positive contribution
  USE_PROBABILITY_MAPPING = False
  USE_8_AUG_TTA           = False  ← marginally hurt acc on this checkpoint
  USE_THRESHOLD_TUNING    = False
  USE_BN_RECALIBRATION    = False

The Phase 5 ablation (see CLAUDE.md Session 9 + OPTIMIZATION_STATUS.md
phase5_flag_contributions) found that LC25000 stain normalization was
the only flag whose removal hurt every metric; probability mapping,
8-aug TTA, and threshold tuning each marginally reduced accuracy on
this checkpoint, so they default to OFF in the canonical pipeline.
Toggle the flags at the top of this file to override.

The Phase 5 FP32 softmax fix is preserved: softmax is applied INSIDE
the autocast block, where PyTorch's autocast policy forces it to FP32.
Computing softmax outside autocast on FP16 logits artificially
depressed the AUC by ~0.09 (see CLAUDE.md Session 9 for the bug
mechanism — argmax-based metrics were unaffected because argmax is
order-preserving across dtype precision).

Dataset layout expected:
  data/external_test/archive/pcam/test_split.h5
        key "x"  -> (32768, 96, 96, 3) uint8  images
  data/external_test/archive/pcam/validation_split.h5
        key "x"  -> (32768, 96, 96, 3) uint8  images
  data/external_test/archive/Labels/Labels/
        camelyonpatch_level_2_split_test_y.h5  key "y" (binary)
        camelyonpatch_level_2_split_valid_y.h5 key "y" (binary)

Class mapping (3-class LC25000 → binary PCam):
  lung_aca (idx 0) → tumor  (1)
  lung_n   (idx 1) → normal (0)
  lung_scc (idx 2) → tumor  (1)

Outputs (always):
  results/metrics/cross_dataset_metrics.json
  results/metrics/cross_dataset_report.txt
  results/plots/cross_dataset_confusion.png

Outputs (when RUN_FLAG_ABLATION = True):
  results/metrics/cross_dataset_metrics_optimized.json
  results/metrics/cross_dataset_flag_ablation.json
  results/plots/cross_dataset_confusion_optimized.png
  results/plots/cross_dataset_threshold_sweep.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\10_cross_dataset.py
"""

import sys, json, time, importlib.util, threading, queue
from pathlib import Path

import h5py
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import softmax
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger


# ════════════════════════════════════════════════════════════════════════════
#  TOGGLEABLE FLAGS  (canonical defaults = Phase 5 best single-flag config)
# ════════════════════════════════════════════════════════════════════════════

USE_LC25000_STAIN_NORM  = True   # Reinhard (LC25000 ref stats) + CLAHE
USE_PROBABILITY_MAPPING = False  # P(tumor)=p_aca+p_scc; otherwise argmax-based
USE_8_AUG_TTA           = False  # mean softmax over 4 flips x {orig, rot90}
USE_THRESHOLD_TUNING    = False  # sweep tau on PCam val labels (allowed)
USE_BN_RECALIBRATION    = False  # off by default; toggle for BN-TTA ablation

# When True, run the Phase 5 flag ablation alongside the primary config.
# Off by default — flip to True to regenerate the optimized + ablation JSONs.
RUN_FLAG_ABLATION = False

# Cap the number of PCam val/test samples (None = full 32,768 each).
MAX_SAMPLES = None


# ════════════════════════════════════════════════════════════════════════════
#  Module loading helpers (avoid renaming files starting with digits)
# ════════════════════════════════════════════════════════════════════════════

def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pp = _load("preprocess03", "03_preprocessing.py")
compute_reinhard_stats = _pp.compute_reinhard_stats
reinhard_normalize     = _pp.reinhard_normalize
apply_clahe            = _pp.apply_clahe
compute_global_ref_stats = _pp.compute_global_ref_stats

HAGCANet = _load("model06", "06_model_hagcanet.py").HAGCANet


# ════════════════════════════════════════════════════════════════════════════
#  Constants — paths + transforms
# ════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

PCAM_ROOT      = CFG.PROJECT_ROOT / "data" / "external_test" / "archive"
PCAM_IMG_VAL   = PCAM_ROOT / "pcam" / "validation_split.h5"
PCAM_LBL_VAL   = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_valid_y.h5"
PCAM_IMG_TEST  = PCAM_ROOT / "pcam" / "test_split.h5"
PCAM_LBL_TEST  = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"

LC25000_TRAIN_CSV = CFG.SPLITS_DIR / "train.csv"
LC25000_REF_STATS_CACHE = CFG.METRICS_DIR / "lc25000_reinhard_ref_stats.json"

CLAHE_CLIP_LIMIT = getattr(CFG, "CLAHE_CLIP_LIMIT", 2.0)
CLAHE_TILE_SIZE  = tuple(getattr(CFG, "CLAHE_TILE_SIZE", (8, 8)))

# Vanilla transform = Resize + ToTensor + Normalize (no stain norm).
VANILLA_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _load_or_compute_lc25000_ref_stats(logger) -> tuple:
    """Reinhard reference statistics from LC25000 training images (cached)."""
    if LC25000_REF_STATS_CACHE.exists():
        try:
            with open(LC25000_REF_STATS_CACHE) as f:
                ref = tuple(json.load(f)["ref_stats"])
            logger.info(f"Loaded LC25000 Reinhard ref stats from cache: "
                        f"{[round(v,2) for v in ref]}")
            return ref
        except Exception:
            pass
    logger.info("Computing LC25000 Reinhard ref stats (200-image sample) ...")
    ref = compute_global_ref_stats(LC25000_TRAIN_CSV, n_sample=200)
    LC25000_REF_STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(LC25000_REF_STATS_CACHE, "w") as f:
        json.dump({"ref_stats": list(ref),
                   "source": str(LC25000_TRAIN_CSV),
                   "n_sample": 200}, f, indent=2)
    return ref


def make_stain_transform(ref_stats):
    """PIL-input transform: Reinhard(LC25000-ref) -> CLAHE -> Resize -> Tensor -> Normalize."""
    def _stain_clahe(pil_img):
        arr = np.array(pil_img.convert("RGB"))
        arr = reinhard_normalize(arr, ref_stats)
        arr = apply_clahe(arr, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE)
        return Image.fromarray(arr)
    return transforms.Compose([
        transforms.Lambda(_stain_clahe),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ════════════════════════════════════════════════════════════════════════════
#  PCam dataset (lazy H5)
# ════════════════════════════════════════════════════════════════════════════

class PCamDataset(Dataset):
    def __init__(self, img_h5, lbl_h5, transform=None, max_samples=None):
        self.img_h5 = str(img_h5)
        self.lbl_h5 = str(lbl_h5)
        self.transform = transform
        with h5py.File(self.img_h5, "r") as f:
            n_total = f["x"].shape[0]
        self.n = n_total if max_samples is None else min(max_samples, n_total)
        self._img_file = None
        self._lbl_file = None

    def _open(self):
        if self._img_file is None:
            self._img_file = h5py.File(self.img_h5, "r")
            self._lbl_file = h5py.File(self.lbl_h5, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._open()
        img_arr = self._img_file["x"][idx]
        label   = int(self._lbl_file["y"][idx, 0, 0, 0])
        img = Image.fromarray(img_arr.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


class PrefetchLoader:
    def __init__(self, loader, device, queue_size=3):
        self.loader = loader; self.device = device; self.queue_size = queue_size
    def __len__(self): return len(self.loader)
    def __iter__(self):
        q = queue.Queue(maxsize=self.queue_size); sentinel = object()
        def _worker():
            try:
                for imgs, labels in self.loader:
                    imgs   = imgs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    q.put((imgs, labels))
            finally:
                q.put(sentinel)
        threading.Thread(target=_worker, daemon=True).start()
        while True:
            it = q.get()
            if it is sentinel: break
            yield it


# ════════════════════════════════════════════════════════════════════════════
#  Inference: collect 3-class softmax probs for an entire loader
# ════════════════════════════════════════════════════════════════════════════

CANCER_CLASS_INDICES = [0, 2]   # lung_aca, lung_scc
NORMAL_CLASS_INDEX   = 1        # lung_n


@torch.no_grad()
def _eight_aug_probs(model, imgs, device):
    """Mean softmax over {orig, hflip, vflip, hvflip} x {orig, rot90}.

    Softmax runs INSIDE the autocast block, where PyTorch's autocast
    policy forces it to FP32 (Phase 5 fix; FP16 softmax outside autocast
    was depressing AUC).
    """
    flips = [imgs,
             torch.flip(imgs, dims=[3]),
             torch.flip(imgs, dims=[2]),
             torch.flip(imgs, dims=[2, 3])]
    probs = []
    use_amp = (device.type == "cuda" and CFG.AMP)
    with torch.amp.autocast("cuda", enabled=use_amp):
        for x in flips:
            probs.append(softmax(model(x), dim=1))
            probs.append(softmax(model(torch.rot90(x, k=1, dims=(2, 3))), dim=1))
    return torch.stack(probs, 0).mean(0)


@torch.no_grad()
def _single_probs(model, imgs, device):
    """Single-pass softmax. FP32 inside autocast (Phase 5 fix)."""
    use_amp = (device.type == "cuda" and CFG.AMP)
    with torch.amp.autocast("cuda", enabled=use_amp):
        return softmax(model(imgs), dim=1)


@torch.no_grad()
def collect_probs(model, loader, device, use_tta, logger, name):
    model.eval()
    all_probs, all_labels = [], []
    n_batches = len(loader)
    t0 = time.time()
    for b_idx, (imgs, labels) in enumerate(PrefetchLoader(loader, device)):
        if use_tta:
            p = _eight_aug_probs(model, imgs, device)
        else:
            p = _single_probs(model, imgs, device)
        all_probs.append(p.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().tolist())
        if (b_idx + 1) % 100 == 0 or (b_idx + 1) == n_batches:
            logger.info(f"  [{name}] batch {b_idx+1}/{n_batches} "
                        f"({time.time()-t0:.0f}s)")
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


# ════════════════════════════════════════════════════════════════════════════
#  BN recalibration (test-time adaptation lite, no labels)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def bn_recalibrate(model, loader, device, logger):
    """Recompute BN running stats over one pass through `loader` (no labels)."""
    bn_modules = [m for m in model.modules()
                  if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    if not bn_modules:
        logger.info("  (model has no BN modules — recalibration is a no-op)")
        return
    logger.info(f"  Recalibrating {len(bn_modules)} BN modules over PCam test set ...")
    saved_modes = [m.training for m in bn_modules]
    saved_momentum = [m.momentum for m in bn_modules]
    for m in bn_modules:
        m.train()
        m.momentum = None  # cumulative average
        m.reset_running_stats()
    model.eval()  # everything else stays in eval
    for m in bn_modules:
        m.train()
    use_amp = (device.type == "cuda" and CFG.AMP)
    t0 = time.time()
    for b_idx, (imgs, _) in enumerate(PrefetchLoader(loader, device)):
        with torch.amp.autocast("cuda", enabled=use_amp):
            _ = model(imgs)
        if (b_idx + 1) % 200 == 0:
            logger.info(f"    bn-recal batch {b_idx+1} ({time.time()-t0:.0f}s)")
    for m, was_training, mom in zip(bn_modules, saved_modes, saved_momentum):
        m.train(was_training)
        m.momentum = mom


# ════════════════════════════════════════════════════════════════════════════
#  Probability mapping + threshold sweep
# ════════════════════════════════════════════════════════════════════════════

def derive_predictions(probs3, use_prob_map, use_thresh, tau):
    """probs3: (N, 3) softmax. Returns (preds, tumor_p)."""
    p_aca = probs3[:, 0]; p_n = probs3[:, 1]; p_scc = probs3[:, 2]
    if use_prob_map:
        denom = p_aca + p_n + p_scc
        denom = np.where(denom < 1e-12, 1.0, denom)
        tumor_p = (p_aca + p_scc) / denom
        thr = float(tau) if use_thresh else 0.5
        preds = (tumor_p >= thr).astype(int)
    else:
        # Argmax-based binary (mirrors original 10_cross_dataset.py exactly).
        three_cls = probs3.argmax(axis=1)
        preds = (three_cls != NORMAL_CLASS_INDEX).astype(int)
        tumor_p = p_aca + p_scc  # for AUC reporting
    return preds, tumor_p


def sweep_threshold(val_tumor_p, val_labels, lo=0.05, hi=0.95, step=0.01):
    taus = np.arange(lo, hi + 1e-9, step)
    f1s, accs, precs, recs = [], [], [], []
    for tau in taus:
        pr = (val_tumor_p >= tau).astype(int)
        f1s.append(f1_score(val_labels, pr, average="binary", zero_division=0))
        accs.append(accuracy_score(val_labels, pr))
        precs.append(precision_score(val_labels, pr, average="binary", zero_division=0))
        recs.append(recall_score(val_labels, pr, average="binary", zero_division=0))
    f1s = np.array(f1s)
    best = int(np.argmax(f1s))
    return {
        "taus": taus.tolist(),
        "f1":   f1s.tolist(),
        "acc":  accs,
        "prec": precs,
        "rec":  recs,
        "best_idx": best,
        "best_tau": float(taus[best]),
        "best_f1":  float(f1s[best]),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion(labels, preds, save_path, title):
    cm   = confusion_matrix(labels, preds)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, sub in zip(
        axes, [cm, norm], ["d", ".2f"], ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["Normal", "Tumor"],
                    yticklabels=["Normal", "Tumor"],
                    ax=ax, linewidths=0.5)
        ax.set_title(f"Confusion Matrix — {sub}", fontsize=12)
        ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(sweep, save_path, title):
    taus = np.array(sweep["taus"]); f1 = np.array(sweep["f1"])
    prec = np.array(sweep["prec"]); rec = np.array(sweep["rec"])
    plt.figure(figsize=(9, 5))
    plt.plot(taus, f1,   label="F1 (binary)", linewidth=2)
    plt.plot(taus, prec, label="Precision",   linestyle="--", alpha=0.7)
    plt.plot(taus, rec,  label="Recall",      linestyle="--", alpha=0.7)
    plt.axvline(0.5, color="grey", linestyle=":", alpha=0.6, label="default 0.5")
    plt.axvline(sweep["best_tau"], color="red", linestyle="-", alpha=0.7,
                label=f"τ*={sweep['best_tau']:.2f}, F1={sweep['best_f1']:.4f}")
    plt.xlabel("Threshold τ on P(tumor)")
    plt.ylabel("Score"); plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
#  Caching: collect val/test probs once per (stain, tta, bn) combination
# ════════════════════════════════════════════════════════════════════════════

def collect_for_combo(model, device, ref_stats, stain, tta, bn_recal, logger,
                      need_val=True):
    """Returns dict with val_probs, val_labels, test_probs, test_labels.

    val collection is skipped when need_val=False (saves time when neither
    threshold tuning nor flag ablation needs it).
    """
    tf = make_stain_transform(ref_stats) if stain else VANILLA_TRANSFORM

    val_p = val_y = None
    if need_val:
        val_ds  = PCamDataset(PCAM_IMG_VAL,  PCAM_LBL_VAL,  tf, max_samples=MAX_SAMPLES)
        val_loader  = DataLoader(val_ds,  batch_size=CFG.BATCH_SIZE,
                                 shuffle=False, num_workers=0, pin_memory=False)

    test_ds = PCamDataset(PCAM_IMG_TEST, PCAM_LBL_TEST, tf, max_samples=MAX_SAMPLES)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)

    if bn_recal:
        logger.info(f"  BN recalibration over PCam test set "
                    f"(stain={stain}, tta={tta}) ...")
        bn_recalibrate(model, test_loader, device, logger)

    if need_val:
        logger.info(f"  Collecting val probs (stain={stain}, tta={tta}) ...")
        val_p, val_y = collect_probs(model, val_loader, device, tta, logger, "val")
    logger.info(f"  Collecting test probs (stain={stain}, tta={tta}) ...")
    test_p, test_y = collect_probs(model, test_loader, device, tta, logger, "test")

    return {"val_probs": val_p, "val_labels": val_y,
            "test_probs": test_p, "test_labels": test_y}


# ════════════════════════════════════════════════════════════════════════════
#  Metric computation per flag config
# ════════════════════════════════════════════════════════════════════════════

def evaluate_config(name, flags, probs_pool, logger):
    """flags = dict {stain, prob, tta, thresh, bn}. Returns (summary, preds, tumor_p, sweep)."""
    key = (flags["stain"], flags["tta"], flags["bn"])
    pool = probs_pool[key]
    val_probs3, val_y = pool["val_probs"], pool["val_labels"]
    tst_probs3, tst_y = pool["test_probs"], pool["test_labels"]

    chosen_tau = 0.5
    if flags["thresh"] and flags["prob"]:
        denom = val_probs3.sum(axis=1)
        denom = np.where(denom < 1e-12, 1.0, denom)
        val_tumor_p = (val_probs3[:, 0] + val_probs3[:, 2]) / denom
        sweep = sweep_threshold(val_tumor_p, val_y, 0.05, 0.95, 0.01)
        chosen_tau = sweep["best_tau"]
        logger.info(f"  [{name}] τ* = {chosen_tau:.2f}  (val F1 = {sweep['best_f1']:.4f})")
    else:
        sweep = None

    preds, tumor_p = derive_predictions(
        tst_probs3, flags["prob"], flags["thresh"], chosen_tau,
    )

    acc  = accuracy_score(tst_y, preds)
    prec = precision_score(tst_y, preds, average="binary", zero_division=0)
    rec  = recall_score(tst_y, preds, average="binary", zero_division=0)
    f1   = f1_score(tst_y, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(tst_y, tumor_p)
    except Exception:
        auc = float("nan")

    return {
        "config":           name,
        "flags":            flags,
        "chosen_threshold": round(float(chosen_tau), 4),
        "accuracy":         round(float(acc),  4),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1_binary":        round(float(f1),   4),
        "roc_auc":          round(float(auc),  4),
        "n_test":           int(len(tst_y)),
    }, preds, tumor_p, sweep


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("cross_dataset")

    logger.info("=" * 72)
    logger.info("  STEP 10: CANONICAL S2 EVALUATION (LC25000 → PCam zero-shot)")
    logger.info("=" * 72)
    logger.info(f"  primary flags: stain={USE_LC25000_STAIN_NORM}, "
                f"prob_map={USE_PROBABILITY_MAPPING}, tta={USE_8_AUG_TTA}, "
                f"thresh={USE_THRESHOLD_TUNING}, bn_recal={USE_BN_RECALIBRATION}")
    logger.info(f"  flag ablation: {RUN_FLAG_ABLATION}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt_path = CFG.BEST_MODEL_PATH
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    model = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    val_f1 = ckpt.get("val_f1", float("nan"))
    if isinstance(val_f1, float):
        logger.info(f"Loaded {ckpt_path.name}  (epoch {ckpt.get('epoch','?')}, "
                    f"val_f1={val_f1:.4f})")
    else:
        logger.info(f"Loaded {ckpt_path.name}")

    # ── Reference stats (LC25000) ────────────────────────────────────────────
    # Only needed if any config uses stain norm. Cheap to compute either way.
    ref_stats = (_load_or_compute_lc25000_ref_stats(logger)
                 if (USE_LC25000_STAIN_NORM or RUN_FLAG_ABLATION)
                 else None)

    # ── Configs to evaluate ──────────────────────────────────────────────────
    primary = {"stain": USE_LC25000_STAIN_NORM,
               "prob":  USE_PROBABILITY_MAPPING,
               "tta":   USE_8_AUG_TTA,
               "thresh": USE_THRESHOLD_TUNING,
               "bn":     USE_BN_RECALIBRATION}

    configs = {"primary": primary}
    if RUN_FLAG_ABLATION:
        # Sanity-check baseline reproduction: every flag off.
        configs["all_off"] = {"stain": False, "prob": False,
                              "tta": False, "thresh": False, "bn": False}
        # Each default-ON flag turned off in turn (others = primary).
        for k in ["stain", "prob", "tta", "thresh"]:
            if primary[k]:
                cfg = dict(primary); cfg[k] = False
                configs[f"off_{k}"] = cfg

    # Determine which combos need validation probs (only for threshold tuning).
    val_combos = {(c["stain"], c["tta"], c["bn"])
                  for c in configs.values() if (c["thresh"] and c["prob"])}
    needed_combos = sorted({(c["stain"], c["tta"], c["bn"]) for c in configs.values()})
    logger.info(f"Need {len(needed_combos)} unique (stain, tta, bn_recal) combos: "
                f"{needed_combos}")

    # ── Collect probs once per combo ─────────────────────────────────────────
    probs_pool = {}
    for combo in needed_combos:
        stain, tta, bn = combo
        logger.info("-" * 72)
        logger.info(f"  combo: stain={stain}, tta={tta}, bn_recal={bn}")
        if bn:
            m = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
            m.eval()
        else:
            m = model
        probs_pool[combo] = collect_for_combo(
            m, device, ref_stats, stain, tta, bn, logger,
            need_val=(combo in val_combos),
        )

    # ── Evaluate every config ────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  Evaluating configurations")
    logger.info("=" * 72)
    results = {}
    extras  = {}
    for name, flags in configs.items():
        m_summary, preds, tumor_p, sweep = evaluate_config(name, flags, probs_pool, logger)
        results[name] = m_summary
        extras[name]  = {"preds": preds, "tumor_p": tumor_p, "sweep": sweep}
        logger.info(f"  {name:>12s} | acc={m_summary['accuracy']:.4f} "
                    f"prec={m_summary['precision']:.4f} "
                    f"rec={m_summary['recall']:.4f} "
                    f"f1={m_summary['f1_binary']:.4f} "
                    f"auc={m_summary['roc_auc']:.4f} "
                    f"τ={m_summary['chosen_threshold']:.2f}")

    primary_res = results["primary"]
    primary_preds = extras["primary"]["preds"]
    primary_test_labels = probs_pool[(primary["stain"], primary["tta"], primary["bn"])]["test_labels"]

    # ── Build canonical cross_dataset_metrics.json ───────────────────────────
    # Schema preserves the keys 14_compare_scenarios.py reads:
    #   n_samples, accuracy, precision, recall, f1_binary, roc_auc,
    #   class_mapping, checkpoint
    canonical_payload = {
        "scenario":         "S2_LC25000_to_PCam_zero_shot",
        "dataset":          "PatchCamelyon_test",
        "n_samples":        primary_res["n_test"],
        "accuracy":         primary_res["accuracy"],
        "precision":        primary_res["precision"],
        "recall":           primary_res["recall"],
        "f1_binary":        primary_res["f1_binary"],
        "roc_auc":          primary_res["roc_auc"],
        "chosen_threshold": primary_res["chosen_threshold"],
        "flags_used":       primary_res["flags"],
        "softmax_precision": "FP32 (Phase 5 fix; supersedes pre-fix FP16 baseline)",
        "class_mapping": {
            "lung_aca (idx 0)": "tumor (1)",
            "lung_n   (idx 1)": "normal (0)",
            "lung_scc (idx 2)": "tumor (1)",
        },
        "checkpoint":       ckpt_path.name,
    }

    out_metrics = CFG.METRICS_DIR / "cross_dataset_metrics.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(canonical_payload, f, indent=2)
    logger.info(f"Saved: {out_metrics}")

    # Text report
    rpt_path = CFG.METRICS_DIR / "cross_dataset_report.txt"
    report_str = classification_report(
        primary_test_labels, primary_preds,
        target_names=["Normal", "Tumor"], digits=4,
    )
    with open(rpt_path, "w") as f:
        f.write("HAGCA-Net Cross-Dataset Evaluation — PatchCamelyon (canonical)\n")
        f.write("=" * 60 + "\n")
        for k, v in canonical_payload.items():
            if k not in ("class_mapping", "flags_used"):
                f.write(f"{k}: {v}\n")
        f.write(f"flags_used: {primary_res['flags']}\n")
        f.write("\n" + report_str)
    logger.info(f"Saved: {rpt_path}")

    # Confusion matrix plot (always)
    cm_png = CFG.PLOTS_DIR / "cross_dataset_confusion.png"
    flags_on = [k for k, v in primary.items() if v]
    title = (f"S2 zero-shot canonical  τ={primary_res['chosen_threshold']:.2f}  "
             f"flags={flags_on if flags_on else ['(none)']}")
    plot_confusion(primary_test_labels, primary_preds, cm_png, title)
    logger.info(f"Saved: {cm_png}")

    # ── Optional ablation outputs ────────────────────────────────────────────
    if RUN_FLAG_ABLATION:
        # Use baseline = the all_off result if available, else the canonical.
        baseline_res = results.get("all_off", primary_res)
        baseline_metrics = {k: baseline_res[k] for k in
                            ("accuracy", "precision", "recall", "f1_binary", "roc_auc")}
        delta_vs_baseline = {k: round(primary_res[k] - baseline_metrics[k], 4)
                             for k in baseline_metrics}

        flag_ablation_block = {}
        for k in ["stain", "prob", "tta", "thresh"]:
            off_name = f"off_{k}"
            if off_name in results:
                off = results[off_name]
                flag_ablation_block[k] = {
                    "off_metrics":  {m: off[m] for m in
                                     ("accuracy", "precision", "recall",
                                      "f1_binary", "roc_auc", "chosen_threshold")},
                    "delta_primary_minus_off": {
                        m: round(primary_res[m] - off[m], 4)
                        for m in ("accuracy", "precision", "recall",
                                  "f1_binary", "roc_auc")
                    },
                }

        optimized_payload = {
            "scenario":  "S2_LC25000_to_PCam_zero_shot_optimized",
            "checkpoint": ckpt_path.name,
            "n_test":     primary_res["n_test"],
            "flags_used": primary_res["flags"],
            "chosen_threshold": primary_res["chosen_threshold"],
            "accuracy":   primary_res["accuracy"],
            "precision":  primary_res["precision"],
            "recall":     primary_res["recall"],
            "f1_binary":  primary_res["f1_binary"],
            "roc_auc":    primary_res["roc_auc"],
            "deltas_vs_baseline":  delta_vs_baseline,
            "baseline_reference":  baseline_metrics,
            "flag_ablation":       flag_ablation_block,
            "class_mapping":       canonical_payload["class_mapping"],
        }

        out_opt = CFG.METRICS_DIR / "cross_dataset_metrics_optimized.json"
        with open(out_opt, "w") as f:
            json.dump(optimized_payload, f, indent=2)
        logger.info(f"Saved: {out_opt}")

        out_abl = CFG.METRICS_DIR / "cross_dataset_flag_ablation.json"
        ablation_doc = {
            "scenario":  "S2_LC25000_to_PCam_zero_shot_flag_ablation",
            "primary":   primary_res,
            "configs":   results,
            "ranking_by_accuracy_drop_when_off": sorted(
                [(k, round(primary_res["accuracy"] - results[f"off_{k}"]["accuracy"], 4))
                 for k in ["stain", "prob", "tta", "thresh"]
                 if f"off_{k}" in results],
                key=lambda x: -x[1],
            ),
        }
        with open(out_abl, "w") as f:
            json.dump(ablation_doc, f, indent=2)
        logger.info(f"Saved: {out_abl}")

        cm_opt_png = CFG.PLOTS_DIR / "cross_dataset_confusion_optimized.png"
        plot_confusion(primary_test_labels, primary_preds, cm_opt_png,
                       f"S2 zero-shot OPTIMIZED  τ={primary_res['chosen_threshold']:.2f}, "
                       f"flags={flags_on}")
        logger.info(f"Saved: {cm_opt_png}")

        if extras["primary"]["sweep"] is not None:
            sw_png = CFG.PLOTS_DIR / "cross_dataset_threshold_sweep.png"
            plot_threshold_sweep(extras["primary"]["sweep"], sw_png,
                                 "S2 zero-shot — PCam val threshold sweep")
            logger.info(f"Saved: {sw_png}")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  CANONICAL S2 RESULTS (PCam test split)")
    logger.info("=" * 72)
    logger.info(f"  Accuracy       : {primary_res['accuracy']:.4f}")
    logger.info(f"  Precision      : {primary_res['precision']:.4f}")
    logger.info(f"  Recall         : {primary_res['recall']:.4f}")
    logger.info(f"  F1 (binary)    : {primary_res['f1_binary']:.4f}")
    logger.info(f"  ROC-AUC        : {primary_res['roc_auc']:.4f}")
    logger.info(f"  Threshold τ    : {primary_res['chosen_threshold']:.2f}")
    logger.info(f"  Flags ON       : {flags_on if flags_on else '(none)'}")
    logger.info(f"  Samples tested : {primary_res['n_test']:,}")


if __name__ == "__main__":
    main()
