"""
10b_cross_dataset_optimized.py
==============================
Phase 5: S2 zero-shot optimization (LC25000 -> PCam).

The LC25000-trained checkpoint (checkpoints/hagcanet_best.pth) is evaluated
on PCam test set with a stack of inference-time + post-hoc tricks. NO PCam
training. NO use of PCam test labels. PCam validation labels are used only
for threshold tuning.

Five toggleable flags (top of file). The script runs one "primary" config
(all default-on flags ON, BN_RECALIBRATION off) plus a flag-ablation:
each individual default-on flag turned off in turn, plus a fully-off
sanity-check run that must reproduce 10_cross_dataset.py's baseline to 4dp.

Outputs:
  results/metrics/cross_dataset_metrics_optimized.json
  results/metrics/cross_dataset_flag_ablation.json
  results/plots/cross_dataset_confusion_optimized.png
  results/plots/cross_dataset_threshold_sweep.png
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
#  TOGGLEABLE FLAGS  (defaults define the "primary" config)
# ════════════════════════════════════════════════════════════════════════════

USE_LC25000_STAIN_NORM  = True   # Reinhard (LC25000 ref stats) + CLAHE
USE_PROBABILITY_MAPPING = True   # P(tumor)=p_aca+p_scc; otherwise argmax-based
USE_8_AUG_TTA           = True   # mean softmax over 4 flips x {orig, rot90}
USE_THRESHOLD_TUNING    = True   # sweep tau on PCam val labels (allowed)
USE_BN_RECALIBRATION    = False  # off by default; toggle for BN-TTA ablation

# Ablation runs each default-ON flag turned off in turn, plus a fully-off
# baseline-reproduction sanity check.
RUN_FLAG_ABLATION = True

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

# Vanilla transform = exactly what 10_cross_dataset.py uses.
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
    """Mean softmax over {orig, hflip, vflip, hvflip} x {orig, rot90}."""
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
    """
    probs3 : (N, 3) softmax over [lung_aca, lung_n, lung_scc]
    Returns:
      preds   : (N,) binary 0/1
      tumor_p : (N,) score used for AUC
    """
    p_aca = probs3[:, 0]; p_n = probs3[:, 1]; p_scc = probs3[:, 2]
    if use_prob_map:
        # P(tumor) = (p_aca + p_scc) / (p_aca + p_n + p_scc)  — already sums to 1
        denom = p_aca + p_n + p_scc
        denom = np.where(denom < 1e-12, 1.0, denom)
        tumor_p = (p_aca + p_scc) / denom
        thr = float(tau) if use_thresh else 0.5
        preds = (tumor_p >= thr).astype(int)
    else:
        # Argmax-based binary (mirrors 10_cross_dataset.py exactly).
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

def collect_for_combo(model, device, ref_stats, stain, tta, bn_recal, logger):
    """Returns dict with val_probs, val_labels, test_probs, test_labels."""
    tf = make_stain_transform(ref_stats) if stain else VANILLA_TRANSFORM

    val_ds  = PCamDataset(PCAM_IMG_VAL,  PCAM_LBL_VAL,  tf, max_samples=MAX_SAMPLES)
    test_ds = PCamDataset(PCAM_IMG_TEST, PCAM_LBL_TEST, tf, max_samples=MAX_SAMPLES)

    val_loader  = DataLoader(val_ds,  batch_size=CFG.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)

    if bn_recal:
        logger.info(f"  BN recalibration over PCam test set "
                    f"(stain={stain}, tta={tta}) ...")
        bn_recalibrate(model, test_loader, device, logger)

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
    """
    flags = dict {stain, prob, tta, thresh, bn}
    probs_pool keyed by (stain, tta, bn) -> dict of arrays.
    """
    key = (flags["stain"], flags["tta"], flags["bn"])
    pool = probs_pool[key]
    val_probs3, val_y = pool["val_probs"], pool["val_labels"]
    tst_probs3, tst_y = pool["test_probs"], pool["test_labels"]

    chosen_tau = 0.5
    if flags["thresh"] and flags["prob"]:
        # Tune tau on val using the prob-mapped tumor scores.
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
    logger = get_logger("phase5_zeroshot")

    logger.info("=" * 72)
    logger.info("  PHASE 5: S2 zero-shot optimization (LC25000 -> PCam)")
    logger.info("=" * 72)
    logger.info(f"  primary flags: stain={USE_LC25000_STAIN_NORM}, "
                f"prob_map={USE_PROBABILITY_MAPPING}, tta={USE_8_AUG_TTA}, "
                f"thresh={USE_THRESHOLD_TUNING}, bn_recal={USE_BN_RECALIBRATION}")
    logger.info(f"  ablation: {RUN_FLAG_ABLATION}")

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
    logger.info(f"Loaded {ckpt_path.name}  (epoch {ckpt.get('epoch','?')}, "
                f"val_f1={val_f1:.4f})" if isinstance(val_f1, float)
                else f"Loaded {ckpt_path.name}")

    # ── Reference stats (LC25000) ────────────────────────────────────────────
    ref_stats = _load_or_compute_lc25000_ref_stats(logger)

    # ── Configs to evaluate ──────────────────────────────────────────────────
    primary = {"stain": USE_LC25000_STAIN_NORM,
               "prob":  USE_PROBABILITY_MAPPING,
               "tta":   USE_8_AUG_TTA,
               "thresh": USE_THRESHOLD_TUNING,
               "bn":     USE_BN_RECALIBRATION}

    configs = {"all_on": primary}
    if RUN_FLAG_ABLATION:
        # Sanity-check baseline reproduction: every flag off.
        configs["all_off"] = {"stain": False, "prob": False,
                              "tta": False, "thresh": False, "bn": False}
        # Each default-on flag turned off in turn (others = primary).
        for k in ["stain", "prob", "tta", "thresh"]:
            if primary[k]:
                cfg = dict(primary); cfg[k] = False
                configs[f"off_{k}"] = cfg

    needed_combos = sorted({(c["stain"], c["tta"], c["bn"]) for c in configs.values()})
    logger.info(f"Need {len(needed_combos)} unique (stain, tta, bn_recal) combos: "
                f"{needed_combos}")

    # ── Collect probs once per combo ─────────────────────────────────────────
    probs_pool = {}
    for combo in needed_combos:
        stain, tta, bn = combo
        logger.info("-" * 72)
        logger.info(f"  combo: stain={stain}, tta={tta}, bn_recal={bn}")
        # Fresh model for BN-recal combos so we don't mutate stats across combos.
        if bn:
            m = HAGCANet(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
            m.eval()
        else:
            m = model
        probs_pool[combo] = collect_for_combo(m, device, ref_stats, stain, tta, bn, logger)

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

    # ── Baseline (from existing JSON) for delta reporting ────────────────────
    baseline_path = CFG.METRICS_DIR / "cross_dataset_metrics.json"
    with open(baseline_path) as f:
        baseline = json.load(f)
    baseline_metrics = {k: float(baseline[k]) for k in
                        ("accuracy", "precision", "recall", "f1_binary", "roc_auc")}

    # ── Sanity check: all_off must reproduce baseline to 4dp ─────────────────
    if "all_off" in results:
        m = results["all_off"]
        diffs = {k: round(m[k] - baseline_metrics[k], 4) for k in baseline_metrics}
        max_abs = max(abs(v) for v in diffs.values())
        logger.info("-" * 72)
        logger.info(f"  [sanity] all_off vs 10_cross_dataset baseline: max|Δ|={max_abs:.4f}")
        for k, v in diffs.items():
            logger.info(f"    {k:<10s}: optimized={m[k]:.4f}  baseline={baseline_metrics[k]:.4f}  Δ={v:+.4f}")
        if max_abs > 1e-4:
            logger.error("  SANITY CHECK FAILED — all_off diverges from baseline > 1e-4")
            logger.error("  Refusing to write outputs; investigate before proceeding.")
            sys.exit(2)
        logger.info("  [sanity] PASS — all_off reproduces baseline to 4dp.")

    # ── Build outputs ────────────────────────────────────────────────────────
    primary_res = results["all_on"]
    delta_vs_baseline = {k: round(primary_res[k] - baseline_metrics[k], 4)
                         for k in baseline_metrics}

    # Per-flag contribution: primary - off_X (positive = flag helped).
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
        "phase":     5,
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
        "class_mapping": {
            "lung_aca (idx 0)": "tumor (1)",
            "lung_n   (idx 1)": "normal (0)",
            "lung_scc (idx 2)": "tumor (1)",
        },
    }

    out_metrics = CFG.METRICS_DIR / "cross_dataset_metrics_optimized.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(optimized_payload, f, indent=2)
    logger.info(f"Saved: {out_metrics}")

    out_ablation = CFG.METRICS_DIR / "cross_dataset_flag_ablation.json"
    ablation_doc = {
        "scenario":  "S2_LC25000_to_PCam_zero_shot_flag_ablation",
        "phase":     5,
        "primary":   primary_res,
        "configs":   results,
        "ranking_by_accuracy_drop_when_off": sorted(
            [(k, round(primary_res["accuracy"] - results[f"off_{k}"]["accuracy"], 4))
             for k in ["stain", "prob", "tta", "thresh"]
             if f"off_{k}" in results],
            key=lambda x: -x[1],
        ),
    }
    with open(out_ablation, "w") as f:
        json.dump(ablation_doc, f, indent=2)
    logger.info(f"Saved: {out_ablation}")

    # ── Plots ────────────────────────────────────────────────────────────────
    cm_png = CFG.PLOTS_DIR / "cross_dataset_confusion_optimized.png"
    plot_confusion(probs_pool[(primary["stain"], primary["tta"], primary["bn"])]["test_labels"],
                   extras["all_on"]["preds"], cm_png,
                   f"S2 zero-shot OPTIMIZED  τ={primary_res['chosen_threshold']:.2f}, "
                   f"flags={[k for k,v in primary.items() if v]}")
    logger.info(f"Saved: {cm_png}")

    if extras["all_on"]["sweep"] is not None:
        sw_png = CFG.PLOTS_DIR / "cross_dataset_threshold_sweep.png"
        plot_threshold_sweep(extras["all_on"]["sweep"], sw_png,
                             "S2 zero-shot — PCam val threshold sweep (8-aug TTA, stain-norm)")
        logger.info(f"Saved: {sw_png}")

    # ── Pretty print ─────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info("  PHASE 5 SUMMARY")
    logger.info("=" * 72)
    logger.info(f"  {'metric':<10s} | {'baseline':>10s} | {'all_on':>10s} | {'Δ':>10s}")
    logger.info("  " + "-" * 50)
    for k in ("accuracy", "precision", "recall", "f1_binary", "roc_auc"):
        b = baseline_metrics[k]; v = primary_res[k]; d = round(v - b, 4)
        logger.info(f"  {k:<10s} | {b:>10.4f} | {v:>10.4f} | {d:>+10.4f}")
    logger.info("  " + "-" * 50)
    logger.info(f"  chosen_threshold = {primary_res['chosen_threshold']:.2f}")

    if RUN_FLAG_ABLATION and flag_ablation_block:
        logger.info("\n  FLAG-ABLATION (acc drop when individual flag is OFF):")
        for k, drop in ablation_doc["ranking_by_accuracy_drop_when_off"]:
            logger.info(f"    {k:<8s}: Δacc when off = {-drop:+.4f}  "
                        f"(contribution = {drop:+.4f})")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
