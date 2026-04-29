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
