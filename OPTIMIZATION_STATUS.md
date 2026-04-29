---
last_phase_completed: 1
next_phase: 2
last_metrics:
  accuracy: 0.9166
  precision: 0.9547
  recall: 0.8745
  f1_binary: 0.9129
  roc_auc: 0.9735
  chosen_threshold: 0.16
last_commit_sha: 55537855ba9c0b69a1a750e069ec0e9a8e91737d
last_commit_tag: phase-1-complete
timestamp: 2026-04-29T11:15:00+05:30
notes: Phase 1 PASSED (delta F1 +0.0259). 8-aug TTA + tuned tau=0.16. Same ckpt.
baseline_provenance: v2_run_session_7_early_stopped_at_epoch_7_total_13_epochs_run_accepted_via_override
---

# HAGCA-Net Optimization Status

Phase 1 complete. Post-hoc evaluation only — no retraining, no architectural
or hyperparameter changes. The 7-epoch baseline checkpoint is unchanged.

## Phase 1 results (8-aug TTA + tuned threshold)
- Accuracy   : 0.9166  (baseline 0.8969,  +0.0197)
- Precision  : 0.9547  (baseline 0.9805,  -0.0258)
- Recall     : 0.8745  (baseline 0.8097,  +0.0648)
- F1 (binary): 0.9129  (baseline 0.8870,  +0.0259)
- ROC-AUC    : 0.9735  (baseline 0.9732,  +0.0003)
- chosen tau : 0.16
- val tau*   : 0.16  (val F1 0.9144 on 16,384 samples)

## Verification
ΔF1 = +0.0259 ≥ 0.02 gate -> PASSED.

## Frozen artifacts
- checkpoints/hagcanet_pcam_baseline.pth                   (Phase 0)
- results/metrics/pcam_train_test_metrics_baseline.json    (Phase 0)
- results/metrics/pcam_train_test_metrics_threshold.json   (Phase 1)
- results/metrics/pcam_threshold_sweep.json                (Phase 1)
- results/plots/pcam_threshold_sweep.png                   (Phase 1)
- results/plots/pcam_confusion_matrix_tuned.png            (Phase 1)
- src/13b_threshold_tta.py                                 (Phase 1)

## Next
Phase 2 — training-speed optimization (precache Reinhard+CLAHE+resize,
channels_last, torch.compile flag, EVAL_BATCH_SIZE=64, optional RAM-load).
Hard requirement: eval metrics must match Phase 0 baseline to 4 dp.
