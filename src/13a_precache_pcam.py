"""
13a_precache_pcam.py  -  Phase 2 precaching
============================================
Runs Reinhard + CLAHE + Resize-to-IMG_SIZE once over all PCam splits and
writes preprocessed HxWx3 uint8 tensors to:
    data/external_test/pcam_preprocessed.h5

Datasets in the output file:
    train_x  (n_train, 224, 224, 3) uint8     | RGB after Reinhard+CLAHE+Resize
    train_y  (n_train,)              uint8     | binary label
    val_x    (n_val,   224, 224, 3) uint8
    val_y    (n_val,)                uint8
    test_x   (n_test,  224, 224, 3) uint8
    test_y   (n_test,)               uint8

Idempotent: if the file already exists with the right datasets and shapes,
exits without recomputing.

Bit-identical guarantee: uses the exact same _reinhard_normalize, _apply_clahe
and torchvision.transforms.Resize as 13_pcam_train_test.py. The Reinhard
reference stats are computed from the same 200 train images (seed=42) so
caching is a pure substitution for the per-sample preprocessing in the
loader.

Usage:
    conda activate lung_cancer
    cd C:\\ml_project
    python src\\13a_precache_pcam.py
"""

import sys, importlib.util, time
from pathlib import Path

import numpy as np
import h5py
from PIL import Image
from torchvision import transforms

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))
from config import CFG, ensure_dirs, get_logger


def _load(alias, fname):
    spec = importlib.util.spec_from_file_location(alias, SRC / fname)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pull math + paths from the training module so they stay in sync.
_train = _load("train13", "13_pcam_train_test.py")
_compute_pcam_ref_stats = _train._compute_pcam_ref_stats
_reinhard_normalize     = _train._reinhard_normalize
_apply_clahe            = _train._apply_clahe
PCAM_ROOT               = _train.PCAM_ROOT
IMG_SIZE                = _train.IMG_SIZE
REF_STAT_SAMPLES        = _train.REF_STAT_SAMPLES
SEED                    = _train.SEED


OUT_PATH = CFG.PROJECT_ROOT / "data" / "external_test" / "pcam_preprocessed.h5"

SPLITS = [
    ("train", PCAM_ROOT / "pcam" / "training_split.h5",
              PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_train_y.h5"),
    ("val",   PCAM_ROOT / "pcam" / "validation_split.h5",
              PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_valid_y.h5"),
    ("test",  PCAM_ROOT / "pcam" / "test_split.h5",
              PCAM_ROOT / "Labels" / "Labels" / "camelyonpatch_level_2_split_test_y.h5"),
]


def _is_cache_complete(out_path: Path) -> bool:
    if not out_path.exists():
        return False
    try:
        with h5py.File(out_path, "r") as f:
            keys = set(f.keys())
            for split, src_x_path, _ in SPLITS:
                kx, ky = f"{split}_x", f"{split}_y"
                if kx not in keys or ky not in keys:
                    return False
                with h5py.File(src_x_path, "r") as fx:
                    n_src = fx["x"].shape[0]
                if f[kx].shape != (n_src, IMG_SIZE, IMG_SIZE, 3):
                    return False
                if f[ky].shape != (n_src,):
                    return False
        return True
    except Exception:
        return False


def main():
    ensure_dirs()
    logger = get_logger("phase2_precache")

    logger.info("=" * 60)
    logger.info("  PHASE 2 PRECACHE: Reinhard + CLAHE + Resize -> uint8 224x224")
    logger.info("=" * 60)
    logger.info(f"  Output: {OUT_PATH}")

    if _is_cache_complete(OUT_PATH):
        logger.info("Cache already complete and consistent -- nothing to do.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Reference stats: identical to what 13_pcam_train_test.py computes. -
    img_train = SPLITS[0][1]
    logger.info(f"Computing Reinhard ref stats from {REF_STAT_SAMPLES} train images "
                f"(seed={SEED}) ...")
    ref_stats = _compute_pcam_ref_stats(img_train, n_sample=REF_STAT_SAMPLES, seed=SEED)
    logger.info(f"Ref stats: {[round(v, 2) for v in ref_stats]}")

    # ---- Use the SAME Resize transform the loader uses (PIL bilinear). ------
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE))

    with h5py.File(OUT_PATH, "w") as fout:
        for split, src_x_path, src_y_path in SPLITS:
            t0 = time.time()
            with h5py.File(src_x_path, "r") as fx, h5py.File(src_y_path, "r") as fy:
                n = int(fx["x"].shape[0])
                logger.info(f"[{split}] processing {n:,} images")

                ds_x = fout.create_dataset(
                    f"{split}_x",
                    shape=(n, IMG_SIZE, IMG_SIZE, 3),
                    dtype="uint8",
                    chunks=(min(64, n), IMG_SIZE, IMG_SIZE, 3),
                    compression="gzip",
                    compression_opts=4,
                )
                labels_arr = fy["y"][:, 0, 0, 0].astype("uint8")
                fout.create_dataset(f"{split}_y", data=labels_arr)

                CHUNK = 256
                last_log = time.time()
                for i0 in range(0, n, CHUNK):
                    i1 = min(i0 + CHUNK, n)
                    src_chunk = fx["x"][i0:i1]
                    out_chunk = np.empty((i1 - i0, IMG_SIZE, IMG_SIZE, 3),
                                         dtype=np.uint8)
                    for j in range(i1 - i0):
                        arr = src_chunk[j].astype(np.uint8)
                        arr = _reinhard_normalize(arr, ref_stats)
                        arr = _apply_clahe(arr)
                        img = Image.fromarray(arr)
                        img = resize(img)
                        out_chunk[j] = np.asarray(img, dtype=np.uint8)
                    ds_x[i0:i1] = out_chunk
                    if time.time() - last_log > 30.0:
                        elapsed = time.time() - t0
                        rate = i1 / max(elapsed, 1e-6)
                        eta = (n - i1) / max(rate, 1e-6)
                        logger.info(f"  [{split}] {i1:>7}/{n:,}  "
                                    f"rate={rate:.1f} img/s  eta={eta:.0f}s")
                        last_log = time.time()
            t = time.time() - t0
            logger.info(f"[{split}] done in {t:.0f}s ({n/t:.1f} img/s)")

    sz = OUT_PATH.stat().st_size / 1e9
    logger.info(f"Cache written: {OUT_PATH}")
    logger.info(f"Cache file size: {sz:.2f} GB")


if __name__ == "__main__":
    main()
