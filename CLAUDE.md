# Histopathological Lung Cancer ML Project - Session Log

> This file is maintained by Claude across sessions. Read at the start of every new session.

---

## Environment & Paths

| Item | Value |
|------|-------|
| Conda env | lung_cancer |
| Python | C:\Users\bmsah\anaconda3\envs\lung_cancer\python.exe |
| Project root | C:\ml_project\ |
| Source code | C:\ml_project\src\ |
| Data | C:\ml_project\data\ |
| Checkpoints | C:\ml_project\checkpoints\ |
| Results | C:\ml_project\results\ |
| Logs | C:\ml_project\logs\ |

**Run any script from C:\ml_project:**
  conda activate lung_cancer
  cd C:\ml_project
  python src\<script_name>.py

---

## Dataset Facts (verified)

| Dataset | Details |
|---------|---------|
| LC25000 train | 3 lung classes x ~4500 images |
| LC25000 test | 3 lung classes x ~500 images |
| PatchCamelyon | HDF5 format in data\external_test\archive\pcam\ |

Lung classes: lung_aca, lung_n, lung_scc (colon excluded)

---

## Key Packages in lung_cancer env

torch 2.11.0+cu128, torchvision, timm 1.0.26, albumentations, opencv,
grad-cam 1.5.5, staintools + spams-bin, h5py, scikit-learn, pandas, numpy,
matplotlib, seaborn

NOTE: torch.cuda.is_available() = False. Training runs on CPU.
torch-geometric NOT installed. Graph module uses pure PyTorch.

---

## Pipeline Status

| Step | File | Status | Notes |
|------|------|--------|-------|
| 1 | Dataset Collection | DONE | Data already downloaded |
| 2 | 01_data_cleaning.py | DONE + TESTED | 805 dups removed, 0 corrupted |
| 3 | 02_data_splitting.py | DONE + TESTED | 9936/1987/2272 train/val/test |
| 4 | 03_preprocessing.py | WRITTEN | Stain norm + CLAHE |
| 5 | 04_augmentation.py | PENDING | |
| 5b | 05_gan_augment.py | PENDING | |
| 6 | 06_model_hagcanet.py | PENDING | |
| 7 | 07_train.py | PENDING | |
| 8 | 08_evaluate.py | PENDING | |
| 9 | 09_gradcam.py | PENDING | |
| 10 | 10_cross_dataset.py | PENDING | |
| 11 | 11_ablation.py | PENDING | |

---

## Cleaning Results (Step 2 - verified run)

- Total scanned: 15,000 (lung classes only)
- Valid: 14,195
- Corrupted: 0
- Duplicates removed: 805 (LC25000 known issue)
- Per class: lung_aca=4727, lung_n=4744, lung_scc=4724

## Split Results (Step 3 - verified run)

- Train: 9,936 (70%) — lung_aca:3308, lung_n:3321, lung_scc:3307
- Val:   1,987 (14%) — lung_aca:662,  lung_n:664,  lung_scc:661
- Test:  2,272 (16%) — lung_aca:757,  lung_n:759,  lung_scc:756
- Group leakage: NONE detected
- Files: data\splits\train.csv, val.csv, test.csv, full_split.csv, manifest.csv

---

## Preprocessing Design (Step 4)

- Macenko stain normalization via staintools (reference = first train image)
- CLAHE applied in LAB color space (L-channel only)
- Outputs to: data\processed\<split>\<class>\<filename>
- Fallback: if stain norm fails on an image, CLAHE-only is applied
- Produces: train_processed.csv, val_processed.csv, test_processed.csv

---

## Project Folder Structure

`
C:\ml_project\
    src\
        config.py               DONE
        01_data_cleaning.py     DONE + TESTED
        02_data_splitting.py    DONE + TESTED
        03_preprocessing.py     WRITTEN (run next)
    data\
        lc25000\train\ test\    raw data (DO NOT MODIFY)
        external_test\          PatchCamelyon HDF5
        splits\                 manifest + train/val/test CSVs
        processed\              preprocessed images go here
    checkpoints\
    results\plots\ metrics\ gradcam\ ablation\
    logs\
    run_pipeline.bat
    run_cleaning.bat
`

---

## Session Log

### Session 1 - 2026-04-25

**Work Done:**
- Read and analyzed NewWork(Group Anuska).docx
- Explored data: LC25000 + PatchCamelyon already downloaded and organized
- Checked conda env lung_cancer: all major packages confirmed
- Installed spams-bin (needed by staintools for stain normalization)
- Created full project folder structure in C:\ml_project\
- Wrote config.py, 01_data_cleaning.py, 02_data_splitting.py, 03_preprocessing.py
- Created run_pipeline.bat and run_cleaning.bat
- User ran Steps 2 and 3 successfully - outputs verified

**Key Findings:**
- LC25000 has 805 duplicate images (known dataset issue) - all removed
- No corrupted images
- Split is clean with zero group leakage
- CUDA not available despite +cu128 build (GPU driver mismatch likely)

### What To Do Next Session

1. User runs: python src\03_preprocessing.py  (takes 15-30 min for 14k images)
2. Write 04_augmentation.py (Dataset class + torchvision transforms)
3. Write 05_gan_augment.py (DCGAN for synthetic lung_scc samples)
4. Write 06_model_hagcanet.py (the main HAGCA-Net model)

### Session 8 - 2026-04-29 - Phase 0 (Optimization Pipeline Baseline)

**Work Done:**
- Started post-hoc optimization protocol (4 phases) on PCam Scenario 3.
- Verified baseline metrics file: accuracy=0.8969, F1=0.8870, AUC=0.9732, best_epoch=7.
- The protocol's `best_epoch >= 8` gate failed; user explicitly overrode with
  rationale that the gate was a ghost-checkpoint heuristic, not a quality
  criterion. Run early-stopped legitimately after ~13 total epochs.
- Froze baseline artifacts:
  - checkpoints/hagcanet_pcam_baseline.pth (copy of hagcanet_pcam_best.pth)
  - results/metrics/pcam_train_test_metrics_baseline.json
- Created OPTIMIZATION_STATUS.md at project root with baseline_provenance field.
- Git already initialized; skipped re-init / v1.0-paper-results substeps.

**Phase 0 Metrics (baseline reference):**
| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.8969 |
| Precision | 0.9805 |
| Recall    | 0.8097 |
| F1        | 0.8870 |
| ROC-AUC   | 0.9732 |

**Files Created/Modified:**
- C:\ml_project\OPTIMIZATION_STATUS.md (new)
- C:\ml_project\checkpoints\hagcanet_pcam_baseline.pth (copy)
- C:\ml_project\results\metrics\pcam_train_test_metrics_baseline.json (copy)
- C:\ml_project\CLAUDE.md (this entry)

**Next:** Phase 1 — post-hoc threshold tuning + 8-aug TTA (no retraining).

### Session 8 (cont.) - 2026-04-29 - Phase 1 (Threshold + 8-aug TTA)

**Work Done:**
- Wrote src/13b_threshold_tta.py: post-hoc evaluation with 8-augmentation TTA
  ({orig,hflip,vflip,hvflip} x {orig,rot90}, mean softmax) + threshold sweep
  on val set (tau in [0.05, 0.95] step 0.01, maximize binary F1).
- Did NOT touch 13_pcam_train_test.py or 06_model_hagcanet.py.
- Ran on the existing 7-epoch checkpoint (hagcanet_pcam_best.pth).
- Best val tau* = 0.16 (val F1 = 0.9144 with 8-aug TTA, 16,384 val samples).
- Applied tau* to test set (32,768 samples).
- Wall time: ~25 min (val ~9 min + test ~17 min on RTX 5060 batch=16).

**Phase 1 Metrics:**
| Metric    | Baseline | Tuned (tau=0.16, 8-aug) | Delta   |
|-----------|----------|-------------------------|---------|
| Accuracy  | 0.8969   | 0.9166                  | +0.0197 |
| Precision | 0.9805   | 0.9547                  | -0.0258 |
| Recall    | 0.8097   | 0.8745                  | +0.0648 |
| F1        | 0.8870   | 0.9129                  | +0.0259 |
| ROC-AUC   | 0.9732   | 0.9735                  | +0.0003 |

**Verification:** ΔF1 = +0.0259 >= 0.02 -> PASSED gate.

The low tau* (0.16) confirms the diagnosis from Session 7: the model's
discriminative power was strong (AUC ~0.973) but the default 0.5 threshold
was systematically biased against the tumor class, suppressing recall. Tuning
the threshold to match the validation distribution recovers ~6.5pp of recall
at a smaller (~2.6pp) precision cost, net F1 +2.6pp.

**Files Created/Modified:**
- C:\ml_project\src\13b_threshold_tta.py (new)
- C:\ml_project\results\metrics\pcam_train_test_metrics_threshold.json (new)
- C:\ml_project\results\metrics\pcam_threshold_sweep.json (new)
- C:\ml_project\results\plots\pcam_threshold_sweep.png (new)
- C:\ml_project\results\plots\pcam_confusion_matrix_tuned.png (new)
- C:\ml_project\OPTIMIZATION_STATUS.md (updated)
- C:\ml_project\CLAUDE.md (this entry)

**Next:** Phase 2 — training-speed optimization (precache + channels_last +
torch.compile flag + bigger eval batch). Verification: bit-identical eval
metrics, then smoke-train test.

### Session 8 (cont.) - 2026-04-29 - Phase 2 (Speed Optimizations)

**Work Done:**
- Wrote src/13a_precache_pcam.py: idempotent one-shot Reinhard+CLAHE+Resize
  pipeline that writes preprocessed 224x224x3 uint8 arrays to
  data/external_test/pcam_preprocessed.h5 (datasets train_x/y, val_x/y,
  test_x/y). Reuses _reinhard_normalize / _apply_clahe from the training
  module so the math stays in lockstep. Same Reinhard ref-stats (200
  samples, seed=42) as the loader.
- Modified src/13_pcam_train_test.py:
  - Added USE_PRECACHE, USE_TORCH_COMPILE, EVAL_BATCH_SIZE,
    RAM_CACHE_LIMIT_BYTES constants.
  - make_transforms() takes use_cached flag; skips Reinhard+CLAHE+Resize
    prefix when cache provides them.
  - PCamDataset now accepts preprocessed_h5_path / cache_split_key /
    ram_cache. RAM-loads per-split arrays under the 8 GB ceiling.
  - build_pcam_loaders auto-detects the cache and uses EVAL_BATCH_SIZE=64
    for val/test loaders (train batch unchanged).
  - Added _to_channels_last(t) and _wrap_model_speed(model, logger):
    converts model to channels_last on CUDA, optionally wraps with
    torch.compile(mode="reduce-overhead") with try/except fallback.
  - Train/eval loops now call .to(memory_format=torch.channels_last) on
    inputs before forward.
  - evaluate_test() uses the cache + EVAL_BATCH_SIZE too.
- Cache build wall time: ~28 min (train 21 min + val 3 min + test 3 min)
  for 327,680 images. Cache file: 44.32 GB (gzip-4 compressed).
- Verification (eval-only) results:
  | Metric    | Baseline | Phase 2 | Delta   |
  |-----------|----------|---------|---------|
  | Accuracy  | 0.8969   | 0.8969  | 0.0000  |
  | Precision | 0.9805   | 0.9804  | -0.0001 |
  | Recall    | 0.8097   | 0.8098  | +0.0001 |
  | F1        | 0.8870   | 0.8870  | 0.0000  |
  | ROC-AUC   | 0.9732   | 0.9732  | 0.0000  |
- One sample (1 / 32,768) flips at the tau=0.5 boundary. Caused by cuDNN
  kernel selection differing under channels_last + EVAL_BATCH_SIZE=64 +
  FP16 AMP. Cache by itself is bit-identical (same Reinhard ref stats and
  Resize op as the in-loader path). User accepted the drift via Option C
  (sub-natural-variance, AUC/F1/accuracy unchanged at 4dp).
- Smoke-train SKIPPED at user request (token conservation; Phase 3 full
  retrain will demonstrate training-time correctness).
- Worktree note: my initial eval-only run hit the worktree copy at
  C:\ml_project\.claude\worktrees\stupefied-rhodes-07196d\src\ which had
  the OLD pre-Phase-2 file. Project root and worktree have diverged source
  trees. Going forward all scripts MUST be run with `cd /c/ml_project &&`
  prefix to ensure the edited code at C:\ml_project\src\ is what runs.

**Files Created/Modified:**
- C:\ml_project\src\13a_precache_pcam.py (new)
- C:\ml_project\src\13_pcam_train_test.py (modified)
- C:\ml_project\data\external_test\pcam_preprocessed.h5 (new, 44.32 GB,
  gitignored under data/)
- C:\ml_project\OPTIMIZATION_STATUS.md (updated)
- C:\ml_project\CLAUDE.md (this entry)

**Next:** Phase 3 — full-data retrain with optimized code. User will launch
the run themselves outside Claude Code to save tokens; resume here for the
post-training analysis.

### Session 8 (cont.) - 2026-04-30 - Phase 2 hotfix (cache for eval-only)

**Bug discovered during Phase 3 launch:**
User launched `python src/13_pcam_train_test.py --force-retrain` at 11:35.
After 2h13m of wall-clock, Phase 1 epoch 1 still hadn't completed. Live
diagnosis:
- Python PID 18528 alive, ~43h CPU time accumulated (heavy thrashing).
- nvidia-smi: GPU at 16% utilization, 2.7/8.1 GB used. Severely I/O-bound.
- No checkpoint written yet (epoch 1 not finished).

**Root cause:** my Phase 2 cache uses `chunks=(64,224,224,3)` with
`compression="gzip" level=4`. For sequential reads (val/test) this is fine
- neighbours share a chunk, decompressed once. For shuffled training reads,
every batch of 32 random samples touches ~32 different gzip chunks => 32
gzip decompressions of ~9.6 MB each per batch. Gzip decompression on
Windows became the bottleneck. The cache HELPED eval (~10 min on test) but
HARMED training (~7x slower than expected).

**Why the smoke-train would have caught this** but didn't run (skipped per
user direction to conserve tokens). Lesson: smoke-train protects against
exactly this class of regression.

**Fix (Option A, user choice):** train always reads from raw PCam H5 (the
pre-Phase-2 fast random-access path); val/test continue to use the cache
(sequential access, fast). No change to model, transforms math, augs,
or hyperparameters.

**Code changes (this hotfix):**
- `make_transforms` now accepts independent `use_cached_train` and
  `use_cached_eval` kwargs. Legacy `use_cached` still works (sets both).
- `build_pcam_loaders` forces `use_cache_train=False`, keeps
  `use_cache_eval=cache_available`. Train DS gets `preprocessed_h5_path=None`,
  val/test DS get the cache. Reinhard ref-stat warmup always runs (train
  needs them).
- `evaluate_test` unchanged (already used the cache via legacy single-flag
  call).

**Verification:** import smoke test confirms:
- legacy call `use_cached=False`: train_tf 11 ops (3 prefix + 8 augs),
  val_tf 5 ops (3 prefix + ToTensor + Normalize).
- new call `use_cached_train=False, use_cached_eval=True`: train_tf 11 ops
  (with raw prefix), val_tf 2 ops (no prefix). As expected.
- Full eval-only re-run skipped (eval path code is unchanged; same cuDNN
  drift +/- 0.0001 on 1 sample expected).

**Files Modified:**
- C:\ml_project\src\13_pcam_train_test.py (make_transforms +
  build_pcam_loaders)
- C:\ml_project\OPTIMIZATION_STATUS.md (updated)
- C:\ml_project\CLAUDE.md (this entry)

**Next:** User re-launches Phase 3 retrain. Expected ~12-20 min/epoch x 18
epochs ~= 4-6 hours total (matches the original Session 7 scaling for
130K -> 262K samples).

### Session 8 (cont.) - 2026-05-02 - Phase 3 (Full-data retrain post-processing)

**Work Done:**
- Confirmed Case A on Step 0: user had launched the full-data retrain
  outside of Claude between Apr 30 and May 1; ckpt mtime May 1 04:34 is
  newer than the phase-2-complete tag commit (Apr 30 14:12). Two protocol
  preparation steps were skipped before that run:
  - Step 1 constants edit (MAX_TRAIN_SAMPLES=None, PHASE2_EPOCHS=18,
    EARLY_STOP=8) was NOT done. Retrain ran on full 262K samples but
    with the post-Phase-2 defaults (3 warmup + 15 main epochs, EARLY_STOP=6).
  - Step 2 backup (hagcanet_pcam_best_pre_full_run.pth) was skipped.
- Retroactive backup: copied checkpoints/hagcanet_pcam_baseline.pth
  (Phase 0) -> checkpoints/hagcanet_pcam_best_pre_full_run.pth so the
  safety net is in place. The Phase 0 ckpt is the appropriate
  "pre-full-run" snapshot since it's what existed before the May 1 retrain.
- Ran src/13b_threshold_tta.py on the new ckpt (hagcanet_pcam_best.pth,
  best_epoch=4 per checkpoint state, val_f1=0.8967). Wall time: ~25 min.
  Val sweep selected tau* = 0.20 (val F1 = 0.9126 with 8-aug TTA).
- Updated results/metrics/pcam_train_test_metrics.json with the
  threshold-tuned numbers (replacing the stale Phase-0 baseline values
  that were still in the file).
- Ran src/14_compare_scenarios.py to refresh the three-scenario summary.

**Phase 3 Metrics (final tuned, tau=0.20, 8-aug TTA):**
| Metric    | Phase 0 baseline | Phase 1 (old ckpt + tune) | Phase 3 (new ckpt + tune) | Δ vs P0 | Δ vs P1 |
|-----------|------------------|---------------------------|---------------------------|---------|---------|
| Accuracy  | 0.8969           | 0.9166                    | 0.9232                    | +0.0263 | +0.0066 |
| Precision | 0.9805           | 0.9547                    | 0.9525                    | -0.0280 | -0.0022 |
| Recall    | 0.8097           | 0.8745                    | 0.8906                    | +0.0809 | +0.0161 |
| F1        | 0.8870           | 0.9129                    | 0.9205                    | +0.0335 | +0.0076 |
| ROC-AUC   | 0.9732           | 0.9735                    | 0.9765                    | +0.0033 | +0.0030 |

**Outcome categorization:** Final acc 0.9232 falls below the 0.96
publication-grade band, but the new model is genuinely better than both
the Phase 0 untuned baseline AND the Phase 1 tuned-on-old-checkpoint
result on every metric except precision (which trades down for recall, as
expected when tau drops from 0.5 to 0.20). **Not a true regression** —
backup not restored. Hypothesis for not reaching ≥0.96: the retrain ran
with only 3+15=18 max epochs and EARLY_STOP=6 (not the prescribed
PHASE2_EPOCHS=18 + EARLY_STOP=8 = 26 max). Best_epoch landed at epoch 4,
which strongly suggests early-stopping fired before the model fully
converged. A future Phase 3.1 with the prescribed settings could close
the gap to the 0.96 target.

**Files Created/Modified:**
- C:\ml_project\checkpoints\hagcanet_pcam_best_pre_full_run.pth (new,
  retroactive copy of Phase 0 baseline)
- C:\ml_project\results\metrics\pcam_train_test_metrics_threshold.json
  (overwritten by 13b_threshold_tta.py with Phase 3 tuned numbers)
- C:\ml_project\results\metrics\pcam_threshold_sweep.json (overwritten)
- C:\ml_project\results\plots\pcam_threshold_sweep.png (overwritten)
- C:\ml_project\results\plots\pcam_confusion_matrix_tuned.png (overwritten)
- C:\ml_project\results\metrics\pcam_train_test_metrics.json (updated to
  Phase 3 tuned numbers; previously held stale Phase 0 baseline)
- C:\ml_project\results\summary\scenario_comparison.txt (refreshed)
- C:\ml_project\results\summary\scenario_comparison.json (refreshed)
- C:\ml_project\results\plots\scenario_comparison.png (refreshed)
- C:\ml_project\OPTIMIZATION_STATUS.md (updated)
- C:\ml_project\CLAUDE.md (this entry)

**Next:** Phase 4 (documentation pass) — gated on user confirmation.


---

### Session - 2026-05-05 (Phase 4 closeout)

**Phase 4 partial / abandoned.** Header docstrings present for
config.py, 01_data_cleaning.py, 02_data_splitting.py, 03_preprocessing.py
(these were already in place and survive as the documented surface).
Abandoned to prioritize Phase 5 and conserve token budget. Remaining
src/*.py files retain their original (sparser) header docstrings. No
README, no full type-hint pass. Will not return to it.

**Next:** Phase 5 — S2 zero-shot optimization (LC25000→PCam, no PCam
training, no PCam test labels). Deliverable: src/10b_cross_dataset_optimized.py.

---

### Session 9 — 2026-05-06 — **★ S2 FP16 SOFTMAX BUG FIX (paper-relevant correction) ★**

**This is the most important note in the optimization log.** A latent
numerical-precision bug in `src/10_cross_dataset.py` was producing an
artificially-deflated cross-dataset AUC. Discovered while running the
Phase 5 sanity check; fixed with a single-line change.

#### What the bug was

`logits_to_binary` in `src/10_cross_dataset.py` computed:

```python
with torch.amp.autocast("cuda", enabled=...):
    logits = model(imgs)            # logits exit autocast as FP16
bin_preds, cancer_prob = logits_to_binary(logits)

def logits_to_binary(logits):
    probs = softmax(logits, dim=1)  # softmax of FP16 logits → FP16 probs
    cancer_prob = (probs[:, 0] + probs[:, 2]).cpu().numpy()
    ...
```

`torch.amp.autocast` only forces softmax to FP32 *inside* the autocast
block. Calling softmax outside autocast on FP16 logits runs in FP16,
which quantizes near-extreme values (probs ≈ 0 or 1) and destroys
fine-grained ranking information. Argmax-based metrics (acc, prec,
rec, F1) are unaffected because argmax is order-preserving across
dtype precision. ROC-AUC is a *continuous* ranking metric and was
artificially depressed by ~0.09.

#### Why argmax-based metrics were unaffected

For any monotonic dtype conversion (FP16 ⇄ FP32) on the same logits,
`argmax(logits)` returns the same index. So `acc`, `prec`, `rec`, `F1`
— all computed from the binary prediction `argmax != lung_n` — are
identical to 4dp regardless of which dtype the softmax used. Only
ROC-AUC, which scores the continuous ranking of `p_aca + p_scc`,
sees the precision loss.

#### How it was caught

`src/10b_cross_dataset_optimized.py` (Phase 5) applies softmax *inside*
the autocast block, where PyTorch's autocast policy forces it to FP32.
When 10b's `all_off` config (every flag disabled) ran the built-in
sanity check against the existing `cross_dataset_metrics.json`, it found:

```
accuracy  : 0.5753 vs 0.5754  Δ=-0.0001  ✓ (cuDNN nondeterminism)
precision : 0.5440 vs 0.5441  Δ=-0.0001  ✓
recall    : 0.9286 vs 0.9286  Δ=+0.0000  ✓
f1_binary : 0.6861 vs 0.6861  Δ=+0.0000  ✓
roc_auc   : 0.6852 vs 0.5987  Δ=+0.0865  ✗ ← bug signature
```

Four discrete metrics matching to 0.0001 + AUC differing by 0.09 = a
precision bug, not a logic bug.

#### The fix

One line, in `logits_to_binary`:

```python
def logits_to_binary(logits):
    logits = logits.float()      # FP32 softmax (Phase 5 fix; FP16 was depressing AUC)
    probs  = softmax(logits, dim=1)
    ...
```

`logits.float()` upcasts FP16 → FP32 before softmax. Argmax behaviour
is preserved exactly (monotonic conversion). No model re-training, no
re-evaluation of S1 / S3 needed (those scripts apply softmax in
different code paths).

#### Corrected S2 baseline (FP32 softmax)

| Metric    | Old (FP16) | Corrected (FP32) | Δ |
|-----------|------------|------------------|---|
| Accuracy  | 0.5754     | 0.5753           | -0.0001 (cuDNN) |
| Precision | 0.5441     | 0.5440           | -0.0001 (cuDNN) |
| Recall    | 0.9286     | 0.9286           |  0.0000 |
| F1-binary | 0.6861     | 0.6861           |  0.0000 |
| **ROC-AUC** | **0.5987**   | **0.6851**           | **+0.0864** |

#### Paper implications

The corrected AUC of **0.6851** **supersedes** the old published value
of 0.5987 in the paper. The old value was an artifact of an FP16
quantization bug, not a property of the model. All other metrics in
the paper are unchanged (within cuDNN-nondeterminism noise).

The stale baseline JSON has been archived at
`results/metrics/cross_dataset_metrics_v1_stale.json` for traceability.

---

### Session 9 (cont.) — 2026-05-06 — Phase 5 (S2 zero-shot optimization)

**Work Done:**
- Inherited `src/10b_cross_dataset_optimized.py` (675 lines, untracked)
  matching the Phase 5 spec point-for-point: 5 toggleable flags
  (`USE_LC25000_STAIN_NORM`, `USE_PROBABILITY_MAPPING`, `USE_8_AUG_TTA`,
  `USE_THRESHOLD_TUNING`, `USE_BN_RECALIBRATION` off-by-default), built-in
  flag-ablation, and a strict 4dp sanity check against
  `10_cross_dataset.py`'s baseline. Used as-is.
- Backed up the LC25000 training checkpoint:
  `checkpoints/hagcanet_lc25000_baseline.pth` (= copy of
  `hagcanet_best.pth`, the LC25000-trained 3-class model).
- First run (against the FP16-buggy baseline) failed the sanity check on
  AUC only. Investigation traced this to the FP16 softmax bug above. Fix
  applied to `src/10_cross_dataset.py`; baseline regenerated.
- Re-ran 10b against the corrected baseline. Sanity check passed
  cleanly: max|Δ| = 0.0000 on all five metrics. Wall-time: ~80 min for
  4 unique (stain, tta, bn) combos × val+test (3 with 8-aug TTA, 1 plain).
- Refreshed three-scenario comparison via `src/14_compare_scenarios.py`.

**Phase 5 Final Results (against corrected FP32 baseline):**

| Config              | Acc    | Prec   | Rec    | F1     | AUC    | τ    |
|---------------------|--------|--------|--------|--------|--------|------|
| baseline (10_cross) | 0.5753 | 0.5440 | 0.9286 | 0.6861 | 0.6851 | 0.50 |
| all_off (sanity)    | 0.5753 | 0.5440 | 0.9286 | 0.6861 | 0.6851 | 0.50 |
| **all_on**          | **0.6161** | 0.5808 | 0.8332 | 0.6845 | 0.6706 | 0.87 |
| off_stain           | 0.5937 | 0.5609 | 0.8616 | 0.6795 | 0.6832 | 0.95 |
| off_prob            | 0.6182 | 0.5765 | 0.8895 | 0.6996 | 0.6706 | 0.50 |
| off_tta             | **0.6212** | 0.5799 | 0.8785 | 0.6986 | 0.6806 | 0.95 |
| off_thresh          | 0.6175 | 0.5758 | 0.8915 | 0.6997 | 0.6706 | 0.50 |

**Flag contribution (Δ accuracy, primary `all_on` minus `off_X`):**
| Flag                       | Δacc       | Verdict                          |
|----------------------------|------------|----------------------------------|
| **LC25000 stain norm**     | **+0.0224** | **Only flag that genuinely helps** |
| Threshold tuning (τ*=0.87) | -0.0014    | Marginally hurts                 |
| Probability mapping        | -0.0021    | Marginally hurts                 |
| 8-aug TTA                  | -0.0051    | Slightly hurts                   |

**Outcome categorization:**
- Δacc all_on vs corrected baseline = **+0.0408** (just under the 0.05
  publication-grade threshold).
- Best single config (`off_tta`) = **+0.0459** vs baseline (also under
  0.05). Per protocol, do NOT overwrite `cross_dataset_metrics.json`
  with the optimized numbers — leave the corrected zero-shot baseline
  as-is so the paper can cite both numbers honestly.
- Stain normalization is the **only** technique with a positive
  contribution. Probability mapping, TTA, and threshold tuning each
  slightly hurt, individually. They mostly trade recall for precision
  without improving the F1 / AUC ranking.
- Hypothesis why stain norm helps: PCam patches differ from LC25000
  in H&E staining intensity (lymph node vs lung tissue, different
  scanner / lab batch). Reinhard normalization to LC25000 reference
  stats reduces this gap, recovering ~2pp accuracy.
- Hypothesis why TTA hurts: the LC25000 model is biased toward
  predicting "tumor" on PCam (recall 0.93 at default τ); averaging
  flipped views amplifies the prior rather than the signal.

**Files Created/Modified:**
- `src/10_cross_dataset.py` — single-line FP32 softmax fix in
  `logits_to_binary`.
- `src/10b_cross_dataset_optimized.py` — newly added (was untracked).
- `results/metrics/cross_dataset_metrics.json` — refreshed (corrected
  AUC 0.6851).
- `results/metrics/cross_dataset_metrics_v1_stale.json` — archived
  pre-fix baseline (AUC 0.5987 FP16) for traceability.
- `results/metrics/cross_dataset_metrics_optimized.json` — new (Phase 5
  optimized numbers + flag-ablation block).
- `results/metrics/cross_dataset_flag_ablation.json` — new (per-flag
  delta breakdown + ranking by accuracy drop when off).
- `results/metrics/cross_dataset_report.txt` — refreshed.
- `results/plots/cross_dataset_confusion.png` — refreshed.
- `results/plots/cross_dataset_confusion_optimized.png` — new.
- `results/plots/cross_dataset_threshold_sweep.png` — new.
- `results/summary/scenario_comparison.{txt,json}` — refreshed (S2 AUC
  0.5988 → 0.6851).
- `results/plots/scenario_comparison.png` — refreshed.
- `checkpoints/hagcanet_lc25000_baseline.pth` — new (copy of
  `hagcanet_best.pth`).
- `OPTIMIZATION_STATUS.md` — updated.
- `CLAUDE.md` — this entry.

**Next:** Phase 5 complete; awaiting user decision on next phase.
