---
last_phase_completed: 0
next_phase: 1
last_metrics:
  accuracy: 0.8969
  precision: 0.9805
  recall: 0.8097
  f1_binary: 0.8870
  roc_auc: 0.9732
  chosen_threshold: null
last_commit_sha: 94d561c3cbae278a8ba87731363de61fe31623de
last_commit_tag: phase-0-complete
timestamp: 2026-04-29T10:31:00+05:30
notes: Baseline frozen from 7-epoch PCam v2 ckpt; baseline gate overridden by user.
baseline_provenance: v2_run_session_7_early_stopped_at_epoch_7_total_13_epochs_run_accepted_via_override
---

# HAGCA-Net Optimization Status

Phase 0 baseline established. The 7-epoch checkpoint was accepted via explicit user
override (the `best_epoch >= 8` gate was a heuristic against ghost-checkpoint bugs,
not a model-quality criterion). The run early-stopped legitimately after ~13 total
epochs and represents real converged metrics.

## Baseline metrics (Phase 0 reference)
- Accuracy   : 0.8969
- Precision  : 0.9805
- Recall     : 0.8097
- F1 (binary): 0.8870
- ROC-AUC    : 0.9732
- val_F1     : 0.8924
- best_epoch : 7

## Frozen artifacts
- checkpoints/hagcanet_pcam_baseline.pth
- results/metrics/pcam_train_test_metrics_baseline.json

## Next
Phase 1 — Post-hoc evaluation (threshold tuning + 8-aug TTA), no retraining.
