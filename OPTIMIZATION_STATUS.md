---
last_phase_completed: 2
next_phase: 3
last_metrics:
  accuracy: 0.8969
  precision: 0.9804
  recall: 0.8098
  f1_binary: 0.8870
  roc_auc: 0.9732
  chosen_threshold: 0.5
last_commit_sha: 1384e1b8cda3ea995d7b6733fe5519f90620501d
last_commit_tag: phase-2-complete
timestamp: 2026-04-29T20:25:00+05:30
notes: Phase 2 complete + hotfix; train uses raw H5, eval uses cache. 1-sample CUDA drift accepted via Option C.
baseline_provenance: v2_run_session_7_early_stopped_at_epoch_7_total_13_epochs_run_accepted_via_override
behavioral_drift_notes: "Phase 2 introduces 1-sample (1/32768) precision/recall drift of ±0.0001 vs baseline at default τ=0.5. AUC, F1, accuracy unchanged at 4dp. Cause: cuDNN kernel selection under channels_last + EVAL_BATCH_SIZE=64 + FP16 AMP. Below natural CUDA non-determinism; accepted via Option C."
---

# HAGCA-Net Optimization Status

Phase 2 complete. Training-speed optimizations applied with bit-identical
eval semantics (modulo a single-sample CUDA non-determinism flip — see
`behavioral_drift_notes` above).

## Phase 2 changes
- New file: src/13a_precache_pcam.py (idempotent Reinhard+CLAHE+Resize cache)
- Modified: src/13_pcam_train_test.py
  - USE_PRECACHE / USE_TORCH_COMPILE / EVAL_BATCH_SIZE / RAM_CACHE_LIMIT_BYTES constants
  - make_transforms() now takes use_cached_train + use_cached_eval (legacy use_cached still works)
  - PCamDataset gains preprocessed_h5_path / cache_split_key / ram_cache args
  - build_pcam_loaders: cache for val/test only (use_cache_train=False, use_cache_eval=cache_available)
    -- HOTFIX: train always reads raw PCam H5 because gzip-compressed cache is slow
       under shuffle=True random access (caught during Phase 3 launch attempt; GPU at 16%
       utilization confirmed I/O-bound). EVAL_BATCH_SIZE=64 for val/test.
  - _wrap_model_speed: model.to(memory_format=torch.channels_last) + optional torch.compile (off by default)
  - Inputs pushed to channels_last before forward in train/eval loops
- New artifact: data/external_test/pcam_preprocessed.h5 (44.32 GB, gitignored)

## Phase 2 verification (eval-only on the same 7-epoch ckpt)
- Accuracy   : 0.8969  (baseline 0.8969)
- Precision  : 0.9804  (baseline 0.9805,  delta -0.0001)
- Recall     : 0.8098  (baseline 0.8097,  delta +0.0001)
- F1 (binary): 0.8870  (baseline 0.8870)
- ROC-AUC    : 0.9732  (baseline 0.9732)
- Smoke-train SKIPPED per user direction (token conservation; Phase 3 will exercise training).

## Operational note (worktree)
The /c/ml_project/.claude/worktrees/stupefied-rhodes-07196d/src/ tree has
the pre-Phase-2 source code; all real work lives at /c/ml_project/src/.
Run scripts with `cd /c/ml_project &&` so the project-root code is loaded.

## Frozen artifacts
- checkpoints/hagcanet_pcam_baseline.pth                   (Phase 0)
- results/metrics/pcam_train_test_metrics_baseline.json    (Phase 0)
- results/metrics/pcam_train_test_metrics_threshold.json   (Phase 1)
- results/metrics/pcam_threshold_sweep.json                (Phase 1)
- results/plots/pcam_threshold_sweep.png                   (Phase 1)
- results/plots/pcam_confusion_matrix_tuned.png            (Phase 1)
- src/13b_threshold_tta.py                                 (Phase 1)
- src/13a_precache_pcam.py                                 (Phase 2)
- data/external_test/pcam_preprocessed.h5                  (Phase 2, 44.32 GB)

## Next
Phase 3 — full-data retrain. User will launch the run themselves outside
Claude Code; this session resumes for post-training analysis (threshold
re-tune via 13b_threshold_tta.py, scenario comparison refresh, etc.).
