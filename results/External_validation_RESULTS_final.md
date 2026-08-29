# External validation of the bundle branch block models — complete results

All six checkpoints, two external cohorts, three label definitions on SNUH.
Point estimates are exact; 95% intervals are percentile bootstrap (seed 20260824).

Alignment was verified before any of this was computed: re-scoring the cached
probabilities reproduced the original SNUH run to four decimal places
(AUROC 0.8061, sensitivity 0.4686, specificity 0.9314 for the standard 12-lead
model), and the stored `new_id` vector matches the metadata row order exactly.

---

## Table 3. Discrimination (AUROC at threshold 0.50)

Standard / Reweighted, with Δ = reweighted − standard.

| Cohort and label | 12-lead | 3-lead | 1-lead |
|---|---|---|---|
| **MIMIC-IV-ECG internal test** (N=5,429; 6.89%) | 0.9849 / 0.9757  **−0.0092** | 0.9645 / 0.9583  **−0.0062** | 0.9188 / 0.9321  **+0.0133** |
| **SNUH**, `bbb` morphology (N=50,000; 26.25%) | 0.8061 / 0.7408  **−0.0653** | 0.8373 / 0.7387  **−0.0986** | 0.7609 / 0.7189  **−0.0420** |
| **SNUH**, QRS ≥ 120 ms (N=48,164; 24.50%) | 0.8478 / 0.7153  **−0.1325** | 0.8307 / 0.8332  **+0.0025** | 0.7268 / 0.7812  **+0.0544** |
| **SNUH**, morphology AND QRS ≥ 120 (N=48,164; 13.61%) | 0.8865 / 0.7288  **−0.1577** | 0.8947 / 0.8051  **−0.0896** | 0.8145 / 0.7691  **−0.0454** |
| **EchoNext**, QRS ≥ 120 ms (N=4,626; 12.26%) | 0.9392 / 0.8203  **−0.1189** | 0.9385 / 0.9240  **−0.0145** | 0.8866 / 0.8787  **−0.0079** |
| **EchoNext**, LVEF ≤ 45% (cross-task; N=4,626; 18.72%) | 0.6692 / 0.6073  **−0.0619** | 0.6810 / 0.6648  **−0.0162** | 0.6955 / 0.6223  **−0.0732** |

Fifteen external comparisons. Reweighting reduced discrimination in **thirteen**
and increased it in **two** — both under the QRS ≥ 120 ms label on SNUH, at three
leads (+0.0025, within noise) and at one lead (+0.0544).

---

## Table 4. The harm is concentrated at twelve leads

At 12 leads the direction is the same in every cohort and under every label, and
the magnitude is many times the internal cost.

| Cohort and label | Δ at 12 leads | multiple of the internal cost (0.0092) |
|---|---|---|
| SNUH, `bbb` morphology | −0.0653 | 7.1× |
| EchoNext, LVEF (cross-task) | −0.0619 | 6.7× |
| EchoNext, QRS ≥ 120 ms | −0.1189 | 12.9× |
| SNUH, QRS ≥ 120 ms | −0.1325 | 14.4× |
| **SNUH, morphology AND QRS ≥ 120** | **−0.1577** | **17.1×** |

At one lead the pattern reverses: reweighting was the only intervention that
helped internally (+0.0133) and it helps again on SNUH under the QRS label
(+0.0544). A plausible reading is that the twelve-lead model had the capacity to
fit a majority-dominated decision boundary tightly, so de-emphasising that
majority destroyed a well-fitted boundary; the single-lead model was underfitted,
so the same reweighting acted as useful regularisation. This is an interpretation
of the pattern, not something the data establish.

---

## Table 5. Why SNUH and EchoNext disagreed — an exact decomposition

Standard 12-lead model.

| Component | Value | Share |
|---|---|---|
| Observed gap, EchoNext (QRS ≥ 120) − SNUH (`bbb`) | 0.1331 | 100% |
| **Label definition**: SNUH QRS ≥ 120 − SNUH `bbb` | 0.0417 | 31% |
| **Cohort and domain**: EchoNext − SNUH, both at QRS ≥ 120 | 0.0914 | 69% |

The two components sum exactly. The label component exists because SNUH's `bbb`
column does not separate complete from incomplete block: only 52.2% of its
positive records have QRS ≥ 120 ms, and RBBB — where the incomplete form is a
common benign variant — accounts for 82% of them. Requiring both morphology and
QRS ≥ 120 ms yields a 13.61% prevalence, next to EchoNext's 12.26%, against
6.89% internally.

Under that clinically correct definition the standard model reaches 0.8865 on
SNUH and 0.9392 on EchoNext, against 0.9849 internally. The model transfers.

---

## Table 6. Operating points, SNUH, morphology AND QRS ≥ 120 ms

| Model | Youden J | Sens @0.50 | Spec @0.50 | Sens @J | Spec @J |
|---|---|---|---|---|---|
| Standard (12-lead) | 0.0605 | 0.6378 | 0.9003 | **0.8514** | **0.8073** |
| Reweighted (12-lead) | 0.6188 | 0.7109 | 0.6706 | 0.6598 | 0.7316 |
| Standard (3-lead) | 0.2042 | 0.6982 | 0.9053 | **0.8305** | **0.8267** |
| Reweighted (3-lead) | 0.0112 | 0.5654 | 0.8278 | 0.8418 | 0.6786 |
| Standard (1-lead) | 0.0875 | 0.2723 | 0.9661 | 0.7378 | 0.7566 |
| Reweighted (1-lead) | 0.0953 | 0.2964 | 0.9303 | 0.7368 | 0.6823 |

Each model evaluated at its own Youden-optimal threshold: the standard model
dominates the reweighted model on both axes at twelve and at three leads. The
thresholds themselves span 0.0028 to 0.6188 across the full set — a 220-fold
range — so no single fixed operating point can deploy this family of models.

---

## Table 7. Subgroup spread, SNUH, morphology AND QRS ≥ 120 ms

Between-group AUROC difference, standard → reweighted.

| Leads | Sex gap | Age gap (>65 vs <50) |
|---|---|---|
| 12 | 0.0046 → 0.0189 (**4.1×**) | 0.0168 → 0.0479 (**2.9×**, direction reversed) |
| 3 | 0.0075 → 0.0240 (3.2×) | 0.0159 → 0.0034 (narrowed) |
| 1 | 0.0013 → 0.0032 (2.5×) | 0.0432 → 0.0358 (narrowed) |

At twelve leads both gaps widen. The age gap widens by levelling down: patients
over 65 — the group with the highest prevalence — fall from 0.8928 to 0.7200, a
loss of 0.1728, while those under 50 fall by 0.1081.

On EchoNext, where race is recorded, the same pattern appears across race at
twelve leads: the between-group spread among the four well-powered groups widened
from 0.0311 to 0.0597 (1.92×), the White-minus-Black difference from +0.0137 to
+0.0484 (3.5×), and Black patients — already the worst-served group — lost 1.31
times as much discrimination as White patients.

---

## What the record now supports, and what it does not

**Supported.** The standard model transfers to both external cohorts when asked a
clinically coherent question (0.8865 SNUH, 0.9392 EchoNext, against 0.9849
internally). Race inverse-frequency reweighting costs 0.062 to 0.158 AUROC at
twelve leads in every external cohort and under every label — six to seventeen
times its internal cost — and widens rather than narrows the subgroup spread at
twelve leads on both cohorts. Sharpening the label makes the standard model
better (+0.0804) and the reweighted model slightly worse (−0.0120).

**Not supported.** That reweighting is uniformly harmful: it helps at one lead on
SNUH under the QRS label (+0.0544), consistent with its being the one internal
setting where it also helped. That the lead-count ordering inverts externally:
under both near-task labels the ordering is 12 ≈ 3 > 1, as internally; only the
cross-task LVEF label produced the inversion. What survives is narrower — the
twelve-lead advantage disappears externally (internal 0.020 over three leads;
external 0.0007 on EchoNext, and −0.008 on SNUH).

**Still open.** The MIMIC label-derivation rule has not been recovered, so no
external label can be asserted to be equivalent to the training target. QRS ≥ 120
ms is a superset of bundle branch block, capturing intraventricular conduction
delay, pacing and ventricular rhythms, and is measured by the recording device
rather than adjudicated. SNUH carries a 13.7% "possible pacing" annotation and a
sensitivity analysis excluding those records is still required. No training seed
was set, so the checkpoints cannot be regenerated bit-for-bit; all evaluation
reported here is deterministic and was reproduced exactly on re-running.
