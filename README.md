# ecg-fairness-external-validation

Code and derived results for:

> **Internal Fairness Gains Do Not Survive External Validation: Race Loss-Reweighting
> for Bundle Branch Block Detection Across Two Independent ECG Cohorts**
> Chuelwon Lee. *Submitted to JAMIA (Research and Applications), 2026.*

The study trains six 1-D ResNet-34 bundle branch block (BBB) detectors on
MIMIC-IV-ECG — three lead configurations (12-lead, 3-lead, 1-lead) × two training
regimes (standard vs. race inverse-frequency loss reweighting) — and then validates
all six on two independent cohorts that were never used in development:
**EchoNext** (PhysioNet) and the **SNUH LYDUS ECG 50K** cohort (Korea Health Data
Platform).

---

## Relationship to prior work by the same author

This repository shares its MIMIC-IV-ECG cohort construction, preprocessing, and
1-D ResNet-34 architecture with an earlier internal-only fairness audit
(`this14/bbb-racial-fairness-audit`). The present study is distinct in question and
in data: it asks whether the internal fairness intervention transfers, and it is
built on two external cohorts absent from the earlier work. The shared upstream
components are reproduced here so that this repository stands alone; they are not
new contributions of this paper.

---

## Layout

```
code/
  train_bbb_leadcount.py       the script that trained the six checkpoints, run once
                               per (mode, lead_set) pair: mode ∈ {vanilla, debiased},
                               lead_set ∈ {1lead, 3lead, 12lead}
  resnet1d.py                  1-D ResNet-34 backbone with the two-unit head the
                               checkpoints carry, plus LEAD_SETS and RACE_WEIGHTS
  cohort_builder.py            MIMIC-IV-ECG race-labelled BBB cohort construction
  preprocess.py                waveform preprocessing
  evaluate.py                  internal (MIMIC test split) evaluation
  run_external_validation.py   external validation pipeline (EchoNext + SNUH)
  fastmetrics.py               rank-based AUROC, Youden threshold, bootstrap CIs
  delong.py                    DeLong variance; independent-samples AUROC test
  ext_summary.py               assembles Table 3 / Table 4 numbers
  score_final.py               re-scores cached probabilities under any label rule
  sens_rhythm.py               paced / atrial-fibrillation sensitivity analysis
  make_figure1.py              Figure 1 (internal vs. external dumbbell chart)

data/
  mimic_scale.npz              per-lead standardisation reference from the MIMIC
                               training split (mu, sd; 12 values each). MIMIC-IV-ECG
                               is stored per-lead standardised, so these are ~N(0,1);
                               they are shipped explicitly so external cohorts are
                               scaled to the training distribution rather than
                               per-record z-scored.
  train_balanced_idx.npy       the fixed internal split, as int64 record indices into
  train_imbalanced_idx.npy     the 27,144-record analytic cohort:
  val_idx.npy                    train 19,000 · val 2,715 · test 5,429
  test_idx.npy                   (mutually disjoint; union = 0…27,143 exactly)
                               `train_imbalanced_idx` (n=4,384) is the majority-skewed
                               primary training set, a strict subset of
                               `train_balanced_idx`. These four arrays are what the
                               paper means by "verified bit-exact on every run".

results/
  snuh_probs.npz               per-record predicted probability for all 50 000 SNUH
                               records × 6 models (new_id, model_names, p0..p5)
  snuh_all_labels.json         the three SNUH label definitions (strict / qrsd120 /
                               complete) used for re-scoring
  snuh_relabel_results.json    full metric set under every label definition
  snuh_rhythm_sensitivity.json paced / AF exclusion sensitivity analysis
  echonext_delong.json         EchoNext subgroup AUROCs with DeLong variances
  External_validation_RESULTS_final.md   all seven result tables in one document

figures/
  Figure1_internal_vs_external.{png,pdf}
```

No waveform data and no patient-level identifiers are contained in this repository.
`snuh_probs.npz` is keyed by the de-identified sequential record index distributed
with the public SNUH cohort.

---

## Reproducing the external validation

```bash
pip install -r requirements.txt

# EchoNext (PhysioNet credentialed access required)
python code/run_external_validation.py --stage echonext

# SNUH LYDUS ECG 50K (Korea Health Data Platform registration required)
python code/run_external_validation.py --stage snuh --label_mode complete
```

All paths are overridable by environment variable so the pipeline runs unchanged on
Linux/GPU and Windows/CPU:

| variable | meaning |
|---|---|
| `EV_MODEL_DIR` | directory holding the six `.pt` checkpoints |
| `EV_ECHO_DIR`  | EchoNext waveform + metadata root |
| `EV_SNUH_DIR`  | SNUH LYDUS root |
| `EV_OUT_DIR`   | where probabilities, caches and JSON results are written |
| `EV_SCALE_NPZ` | path to `mimic_scale.npz` (defaults beside the script) |

Per-model probability caches make a killed run resumable, and every reported metric
can be recomputed from `results/snuh_probs.npz` alone — without waveform access —
via `code/score_final.py`.

The pipeline was run end to end on two independent platforms (Linux + T4 GPU,
torch 2.x; Windows + CPU, torch 2.9.1). All six models and every reported stratum
agreed to four decimal places.

---

## Data access

| cohort | access |
|---|---|
| MIMIC-IV-ECG v1.0 (doi:10.13026/4nqg-sb35) | Open Data Commons ODbL v1.0. Linked demographics come from MIMIC-IV (doi:10.13026/kpb9-mt58), credentialed access. |
| EchoNext v1.1.1 (doi:10.13026/7cfw-d091) | PhysioNet Restricted Health Data License 1.5.0, credentialed access. |
| SNUH LYDUS ECG 50K v1.1.1 | Listed as freely available on the Korea Health Data Platform: https://khdp.net/database/data-search-detail/LYDUS-SNUH-ECG-50K — downloadable by registered users, no data use agreement or application. |

## License

MIT (see `LICENSE`). Please cite the paper if you use this code.

## Contact

Chuelwon Lee — lcw@huinno.com
RAQA Team, HUINNO Co., Ltd., 3F, 19 Apgujeong-ro 79-gil, Gangnam-gu, Seoul 06011,
Republic of Korea
