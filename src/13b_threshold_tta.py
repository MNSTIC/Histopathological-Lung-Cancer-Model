"""
13b_threshold_tta.py  -  Phase 1 post-hoc evaluation
=====================================================
Loads the existing PCam best checkpoint, runs 8-augmentation TTA
(mean softmax over {orig, hflip, vflip, hvflip} x {orig, rot90}),
tunes the decision threshold tau on the validation set to maximize
binary F1, then evaluates the test set with that tuned tau.

This script does NOT retrain. It only re-evaluates an existing ckpt.
The training script 13_pcam_train_test.py is not modified.

Outputs:
    results/metrics/pcam_train_test_metrics_threshold.json
    results/metrics/pcam_threshold_sweep.json
    results/plots/pcam_threshold_sweep.png
    results/plots/pcam_confusion_matrix_tuned.png

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\13b_threshold_tta.py
"""

import sys, os, json, importlib.util, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, setup_device, ensure_dirs, get_logger


def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse model + dataset + transforms from the training script.
_train_mod = _load("train13", "13_pcam_train_test.py")
HAGCANet         = _load("model", "06_model_hagcanet.py").HAGCANet
PCamDataset      = _train_mod.PCamDataset
make_transforms  = _train_mod.make_transforms
PCAM_ROOT        = _train_mod.PCAM_ROOT
PCAM_CKPT        = _train_mod.PCAM_CKPT
PCAM_NUM_CLASSES = _train_mod.PCAM_NUM_CLASSES
PCAM_CLASSES     = _train_mod.PCAM_CLASSES
MAX_VAL_SAMPLES  = _train_mod.MAX_VAL_SAMPLES
MAX_TEST_SAMPLES = _train_mod.MAX_TEST_SAMPLES
PrefetchLoader   = _train_mod.PrefetchLoader


# ---------- 8-augmentation TTA ------------------------------------------------
@torch.no_grad()
def _eight_aug_probs(model, imgs, device):
    """Mean softmax prob over 8 TTA augs:
       {orig, hflip, vflip, hvflip} x {orig, rot90}."""
    flipped = [
        imgs,                                       # original
        torch.flip(imgs, dims=[3]),                 # H-flip
        torch.flip(imgs, dims=[2]),                 # V-flip
        torch.flip(imgs, dims=[2, 3]),              # H+V flip
    ]
    all_probs = []
    use_amp = (device.type == "cuda" and CFG.AMP)
    with torch.amp.autocast("cuda", enabled=use_amp):
        for x in flipped:
            all_probs.append(softmax(model(x), dim=1))
            x_rot = torch.rot90(x, k=1, dims=(2, 3))
            all_probs.append(softmax(model(x_rot), dim=1))
    return torch.stack(all_probs, 0).mean(0)


@torch.no_grad()
def _collect_probs(model, loader, device, logger, name):
    all_probs, all_labels = [], []
    n_batches = len(loader)
    t0 = time.time()
    for b_idx, (imgs, labels) in enumerate(PrefetchLoader(loader, device)):
        probs_t = _eight_aug_probs(model, imgs, device)
        all_probs.extend(probs_t[:, 1].cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        if (b_idx + 1) % 50 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t0
            logger.info(f"  [{name}] batch {b_idx+1}/{n_batches}  ({elapsed:.0f}s)")
    return np.array(all_probs), np.array(all_labels)


def _sweep_threshold(val_probs, val_labels, lo=0.05, hi=0.95, step=0.01):
    taus = np.arange(lo, hi + 1e-9, step)
    f1s, accs, recs, precs = [], [], [], []
    for tau in taus:
        preds = (val_probs >= tau).astype(int)
        f1s.append(f1_score(val_labels, preds, average="binary", zero_division=0))
        accs.append(accuracy_score(val_labels, preds))
        precs.append(precision_score(val_labels, preds, average="binary", zero_division=0))
        recs.append(recall_score(val_labels, preds, average="binary", zero_division=0))
    f1s = np.array(f1s); accs = np.array(accs); precs = np.array(precs); recs = np.array(recs)
    best = int(np.argmax(f1s))
    return {
        "taus":  taus.tolist(),
        "f1":    f1s.tolist(),
        "acc":   accs.tolist(),
        "prec":  precs.tolist(),
        "rec":   recs.tolist(),
        "best_idx": best,
        "best_tau": float(taus[best]),
        "best_f1":  float(f1s[best]),
    }


def _plot_sweep(sweep, out_path):
    taus = np.array(sweep["taus"])
    f1   = np.array(sweep["f1"])
    prec = np.array(sweep["prec"])
    rec  = np.array(sweep["rec"])
    best_tau = sweep["best_tau"]
    best_f1  = sweep["best_f1"]

    plt.figure(figsize=(9, 5))
    plt.plot(taus, f1,   label="F1 (binary)", linewidth=2)
    plt.plot(taus, prec, label="Precision",   linestyle="--", alpha=0.7)
    plt.plot(taus, rec,  label="Recall",      linestyle="--", alpha=0.7)
    plt.axvline(0.5, color="grey", linestyle=":", alpha=0.6, label="default tau=0.5")
    plt.axvline(best_tau, color="red", linestyle="-", alpha=0.6,
                label=f"tau*={best_tau:.2f}, F1={best_f1:.4f}")
    plt.xlabel("Threshold tau (P(tumor) >= tau -> predict tumor)")
    plt.ylabel("Score")
    plt.title("PCam validation: threshold sweep (8-aug TTA)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_confusion(labels, preds, out_path, title):
    cm   = confusion_matrix(labels, preds)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, sub in zip(
        axes, [cm, norm], ["d", ".2f"], ["Counts", "Normalised (row %)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=PCAM_CLASSES, yticklabels=PCAM_CLASSES,
                    ax=ax, linewidths=0.5)
        ax.set_title(f"Confusion Matrix -- {sub}", fontsize=12)
        ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
#  Main
# ============================================================================

def main():
    setup_device()
    ensure_dirs()
    device = torch.device(CFG.DEVICE)
    logger = get_logger("phase1_threshold_tta")

    logger.info("=" * 60)
    logger.info("  PHASE 1: post-hoc threshold tuning + 8-aug TTA")
    logger.info("=" * 60)

    if not PCAM_CKPT.exists():
        logger.error(f"Checkpoint not found: {PCAM_CKPT}")
        sys.exit(1)

    # ---- Build val + test loaders (same factory as 13_pcam_train_test.py) ----
    img_train = PCAM_ROOT / "pcam" / "training_split.h5"
    img_val   = PCAM_ROOT / "pcam" / "validation_split.h5"
    lbl_val   = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_valid_y.h5"
    img_test  = PCAM_ROOT / "pcam" / "test_split.h5"
    lbl_test  = PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"

    _, val_tf = make_transforms(img_train)

    val_ds  = PCamDataset(img_val,  lbl_val,  val_tf, max_samples=MAX_VAL_SAMPLES)
    test_ds = PCamDataset(img_test, lbl_test, val_tf, max_samples=MAX_TEST_SAMPLES)
    logger.info(f"Val samples : {len(val_ds):,}")
    logger.info(f"Test samples: {len(test_ds):,}")

    val_loader  = DataLoader(val_ds,  batch_size=CFG.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=False)

    # Trigger Reinhard ref-stat compute (same as 13_pcam_train_test.py).
    val_ds[0]

    # ---- Load checkpoint ----
    logger.info(f"Loading checkpoint: {PCAM_CKPT.name}")
    model = HAGCANet(num_classes=PCAM_NUM_CLASSES, pretrained=False).to(device)
    ckpt  = torch.load(PCAM_CKPT, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info(f"  epoch={ckpt['epoch']}  ckpt_val_F1={ckpt['val_f1']:.4f}")

    # ---- Collect val probs (8-aug TTA), sweep tau ----
    logger.info("Collecting val probs (8-aug TTA) ...")
    val_probs, val_labels = _collect_probs(model, val_loader, device, logger, "val")
    logger.info("Sweeping threshold on val set ...")
    sweep = _sweep_threshold(val_probs, val_labels, 0.05, 0.95, 0.01)
    tau_star = sweep["best_tau"]
    logger.info(f"Best val tau* = {tau_star:.2f}  (val F1 = {sweep['best_f1']:.4f})")

    # ---- Collect test probs (8-aug TTA) ----
    logger.info("Collecting test probs (8-aug TTA) ...")
    test_probs, test_labels = _collect_probs(model, test_loader, device, logger, "test")

    # ---- Apply tau* to test set ----
    preds_tuned = (test_probs >= tau_star).astype(int)
    acc  = accuracy_score(test_labels, preds_tuned)
    prec = precision_score(test_labels, preds_tuned, average="binary", zero_division=0)
    rec  = recall_score(test_labels, preds_tuned, average="binary", zero_division=0)
    f1   = f1_score(test_labels, preds_tuned, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(test_labels, test_probs)
    except Exception:
        auc = float("nan")
    report = classification_report(test_labels, preds_tuned,
                                   target_names=PCAM_CLASSES, digits=4)

    # ---- Load baseline metrics (the unmodified Phase 0 reference) ----
    baseline_path = CFG.METRICS_DIR / "pcam_train_test_metrics_baseline.json"
    with open(baseline_path) as f:
        baseline = json.load(f)

    delta = {
        "accuracy":   round(float(acc)  - float(baseline["accuracy"]),  4),
        "precision":  round(float(prec) - float(baseline["precision"]), 4),
        "recall":     round(float(rec)  - float(baseline["recall"]),    4),
        "f1_binary":  round(float(f1)   - float(baseline["f1_binary"]), 4),
        "roc_auc":    round(float(auc)  - float(baseline["roc_auc"]),   4),
    }

    metrics_out = {
        "scenario":         "PCam_trained_PCam_tested",
        "phase":            1,
        "tta":              "8-aug",
        "tta_detail":       "{orig,hflip,vflip,hvflip} x {orig,rot90}",
        "chosen_threshold": round(float(tau_star), 4),
        "n_test":           int(len(test_labels)),
        "accuracy":         round(float(acc),  4),
        "precision":        round(float(prec), 4),
        "recall":           round(float(rec),  4),
        "f1_binary":        round(float(f1),   4),
        "roc_auc":          round(float(auc),  4),
        "delta_vs_baseline": delta,
        "checkpoint":       PCAM_CKPT.name,
        "ckpt_epoch":       int(ckpt["epoch"]),
        "ckpt_val_f1":      round(float(ckpt["val_f1"]), 4),
        "val_n":            int(len(val_labels)),
        "val_best_tau":     round(float(sweep["best_tau"]), 4),
        "val_best_f1":      round(float(sweep["best_f1"]), 4),
    }

    out_metrics = CFG.METRICS_DIR / "pcam_train_test_metrics_threshold.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as f:
        json.dump(metrics_out, f, indent=2)
    logger.info(f"Saved: {out_metrics}")

    out_sweep = CFG.METRICS_DIR / "pcam_threshold_sweep.json"
    with open(out_sweep, "w") as f:
        json.dump(sweep, f, indent=2)
    logger.info(f"Saved: {out_sweep}")

    out_sweep_png = CFG.PLOTS_DIR / "pcam_threshold_sweep.png"
    _plot_sweep(sweep, out_sweep_png)
    logger.info(f"Saved: {out_sweep_png}")

    out_cm_png = CFG.PLOTS_DIR / "pcam_confusion_matrix_tuned.png"
    _plot_confusion(test_labels, preds_tuned, out_cm_png,
                    f"PCam test: 8-aug TTA + tuned tau={tau_star:.2f}")
    logger.info(f"Saved: {out_cm_png}")

    # ---- Side-by-side print ----
    logger.info("\n" + "=" * 72)
    logger.info("  PHASE 1 SIDE-BY-SIDE  (baseline vs tuned)")
    logger.info("=" * 72)
    header = f"  {'Metric':<10s} | {'Baseline (tau=0.5, 4-flip)':>28s} | {'Tuned (tau*, 8-aug)':>22s} | {'Delta':>8s}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for k_pretty, k in [("Accuracy", "accuracy"), ("Precision", "precision"),
                        ("Recall", "recall"), ("F1", "f1_binary"), ("ROC-AUC", "roc_auc")]:
        b = float(baseline[k])
        t = float(metrics_out[k])
        d = t - b
        logger.info(f"  {k_pretty:<10s} | {b:>28.4f} | {t:>22.4f} | {d:>+8.4f}")
    logger.info("=" * 72)
    logger.info(f"  chosen_threshold = {tau_star:.2f}   (default was 0.5)")
    logger.info(f"  TTA              = 8-aug (was 4-flip)")
    logger.info("=" * 72)
    logger.info("\nClassification report (tuned tau, test set):\n" + report)

    return metrics_out


if __name__ == "__main__":
    main()
