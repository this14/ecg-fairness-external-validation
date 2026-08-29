"""
run_external_validation.py  (v2 -- paths and schemas resolved by the probe)
==========================================================================
Internal re-verification (MIMIC-IV-ECG) + external evaluation on
SNUH LYDUS (cross-institutional / cross-ethnic) and EchoNext (cross-task, LVEF)
for the bundle branch block (BBB) models.

WHAT THE PROBE ESTABLISHED
--------------------------
  * checkpoints live in  VitalDBK-MIMIC/model_bbb_leadcount/
  * TWO different X.npy exist:
        preprocessed_27178_v2/X.npy      27,144 rows  <- matches y/groups/subjects
        VitalDBK-MIMIC/preprocessed_bbb/X.npy  27,178 rows  <- DOES NOT match
    Using the wrong one silently misaligns every waveform against its label.
    MIMIC_DIR below points at the self-consistent copy, and stage `mimic`
    asserts the alignment and reproduces the published numbers.
  * SNUH stores EIGHT independent leads (50000, 8, 5000) int16, not twelve.
    III/aVR/aVL/aVF are reconstructed from I and II below.
  * SNUH row order is NOT the CSV row order: npz['keys'] is matched to
    CSV['new_id'] explicitly (new_id is not a contiguous 0..N-1 range).
  * EchoNext val split is 4,626 records, float64, 250 Hz, and the metadata
    carries race_ethnicity / sex / age -- so subgroup fairness is testable
    in an external cohort.

USAGE
-----
    !python run_external_validation.py --stage probe
    !python run_external_validation.py --stage mimic        # RUN THIS SECOND
    !python run_external_validation.py --stage snuh
    !python run_external_validation.py --stage echonext

Run `mimic` before the external stages. If it does not reproduce the
published internal numbers, nothing downstream can be trusted.

Requires: torch, numpy, pandas, scikit-learn, scipy
"""

import argparse, json, os, warnings
import numpy as np

# ----------------------------------------------------------------------------
# Every path can be overridden by an environment variable, so the same file runs
# unchanged on Colab (Google Drive) and on Kaggle (/kaggle/input/<slug>).
_E = os.environ.get

DRIVE     = _E("EV_DRIVE",     "/content/drive/MyDrive")
MODEL_DIR = _E("EV_MODEL_DIR", f"{DRIVE}/VitalDBK-MIMIC/model_bbb_leadcount")
MIMIC_DIR = _E("EV_MIMIC_DIR", f"{DRIVE}/preprocessed_27178_v2")
SPLIT_DIR = _E("EV_SPLIT_DIR", f"{MIMIC_DIR}/imbalanced_split")
SNUH_DIR  = _E("EV_SNUH_DIR",  f"{DRIVE}/SNUH LYDUS ECG 50K")
ECHO_DIR  = _E("EV_ECHO_DIR",  f"{DRIVE}/EchoNext")
OUT_DIR   = _E("EV_OUT_DIR",   f"{DRIVE}/external_validation_out")

SNUH_NPZ_500 = _E("EV_SNUH_NPZ_500", f"{SNUH_DIR}/LYDUS_SNUH_ECG_ANONYMIZED_50k_500hz_1219.npz")
SNUH_NPZ_100 = _E("EV_SNUH_NPZ_100", f"{SNUH_DIR}/LYDUS_SNUH_ECG_ANONYMIZED_50k_100hz_1219.npz")
SNUH_CSV     = _E("EV_SNUH_CSV",     f"{SNUH_DIR}/LYDUS_SNUH_ECG_ANONYMIZED_50K_0512.csv")
ECHO_WAVE    = _E("EV_ECHO_WAVE",    f"{ECHO_DIR}/EchoNext_val_waveforms.npy")
ECHO_META    = _E("EV_ECHO_META",    f"{ECHO_DIR}/echonext_metadata_100k_v1.1.1.csv")

SEED = 20260824

# Per-lead reference statistics of MIMIC-IV-ECG, computed once from 512 records
# drawn with RandomState(20260824). The stored waveform array is already per-lead
# standardised, so these are zero mean and unit variance to six decimal places;
# they are recorded explicitly so the preprocessing can be reproduced without the
# 6.5 GB array. A mimic_scale.npz beside the script takes precedence.
MIMIC_SCALE_MU = [-0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0]
MIMIC_SCALE_SD = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.999023, 1.0, 1.0, 1.0]

LEAD_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LEAD_SETS  = {"1lead": [1], "3lead": [0,1,10], "12lead": list(range(12))}
GROUP_NAMES = ['Asian','Black','Hispanic/Latino','White','Other']
MODELS = [
    ("Standard (12-lead)",   "model_vanilla_12lead.pt",  "12lead"),
    ("Reweighted (12-lead)", "model_debiased_12lead.pt", "12lead"),
    ("Standard (3-lead)",    "model_vanilla_3lead.pt",   "3lead"),
    ("Reweighted (3-lead)",  "model_debiased_3lead.pt",  "3lead"),
    ("Standard (1-lead)",    "model_vanilla_1lead.pt",   "1lead"),
    ("Reweighted (1-lead)",  "model_debiased_1lead.pt",  "1lead"),
]
# SNUH stores 8 independent leads; this is the assumed channel order.
SNUH_8_ORDER = ["I","II","V1","V2","V3","V4","V5","V6"]

# published internal values, for the reproduction check in stage `mimic`
PUBLISHED = {
    "Standard (12-lead)":   (0.9849, 0.7861, 0.9949),
    "Reweighted (12-lead)": (0.9757, 0.9064, 0.9426),
    "Standard (3-lead)":    (0.9645, 0.7005, 0.9931),
    "Reweighted (3-lead)":  (0.9583, 0.6765, 0.9838),
    "Standard (1-lead)":    (0.9188, 0.4385, 0.9905),
    "Reweighted (1-lead)":  (0.9321, 0.5909, 0.9782),
}


# ----------------------------------------------------------------------------
# Model (mirrors resnet1d.py)
# ----------------------------------------------------------------------------
def build_model(in_channels):
    import torch.nn as nn

    class BasicBlock1D(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride, 3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, 7, 1, 3, bias=False)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = None
            if stride != 1 or in_ch != out_ch:
                self.downsample = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm1d(out_ch))

        def forward(self, x):
            idt = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                idt = self.downsample(x)
            out += idt
            return self.relu(out)

    class ResNet1D34(nn.Module):
        def __init__(self, in_channels=12, num_classes=2, base_width=64):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(in_channels, base_width, 15, 2, 7, bias=False),
                nn.BatchNorm1d(base_width), nn.ReLU(inplace=True),
                nn.MaxPool1d(3, 2, 1))
            widths, blocks = [base_width, base_width*2, base_width*4, base_width*8], [3,4,6,3]
            layers, in_ch = [], base_width
            for i, (w, n) in enumerate(zip(widths, blocks)):
                for j in range(n):
                    layers.append(BasicBlock1D(in_ch, w, stride=2 if (j == 0 and i > 0) else 1))
                    in_ch = w
            self.layers = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(in_ch, num_classes)

        def forward(self, x):                     # x: (B, T, C)
            x = x.permute(0, 2, 1)
            x = self.stem(x); x = self.layers(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    return ResNet1D34(in_channels=in_channels, num_classes=2)


def load_checkpoint_strict(path, in_channels):
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint missing: {path}")
    model = build_model(in_channels)
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)      # raises on ANY key/shape mismatch
    print(f"    [ckpt] {os.path.basename(path)}: strict load OK, {len(sd)} tensors")
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def auc_(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p)

def sens_(y, p, threshold=0.5):
    tp = ((p >= threshold) & (y == 1)).sum(); fn = ((p < threshold) & (y == 1)).sum()
    return float(tp/(tp+fn)) if (tp+fn) else float("nan")

def spec_(y, p, threshold=0.5):
    tn = ((p < threshold) & (y == 0)).sum(); fp = ((p >= threshold) & (y == 0)).sum()
    return float(tn/(tn+fp)) if (tn+fp) else float("nan")

def delong_var(y, p):
    m, n = int((y == 1).sum()), int((y == 0).sum())
    if m == 0 or n == 0:
        return np.nan, np.nan
    pos, neg = p[y == 1], p[y == 0]
    V10 = np.array([np.mean(a > neg) + .5*np.mean(a == neg) for a in pos])
    V01 = np.array([np.mean(pos > b) + .5*np.mean(pos == b) for b in neg])
    s10 = np.var(V10, ddof=1) if len(V10) > 1 else 0.0
    s01 = np.var(V01, ddof=1) if len(V01) > 1 else 0.0
    return auc_(y, p), s10/m + s01/n

def delong_z_vs_ref(y_a, p_a, y_b, p_b):
    """Independent-samples AUROC z-test using DeLong variances (disjoint groups)."""
    from scipy.stats import norm
    a1, v1 = delong_var(y_a, p_a); a2, v2 = delong_var(y_b, p_b)
    if not np.isfinite(v1) or not np.isfinite(v2) or (v1 + v2) <= 0:
        return np.nan, np.nan, np.nan
    z = (a1 - a2) / np.sqrt(v1 + v2)
    return a1, a2, float(2 * (1 - norm.cdf(abs(z))))

def boot_ci(y, p, fn, n_boot=1000, seed=SEED, **kw):
    rng = np.random.RandomState(seed); n, vals = len(y), []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i], **kw))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])

def summarize(y, p, threshold, label, n_boot=1000):
    y = np.asarray(y).astype(int); p = np.asarray(p).astype(float)
    assert len(y) == len(p) and len(y) > 0, f"{label}: empty cohort"
    assert len(np.unique(y)) == 2, (
        f"{label}: degenerate labels ({np.bincount(y, minlength=2).tolist()}). "
        "A single-class cohort cannot produce a meaningful AUROC.")
    a, var = delong_var(y, p)
    lo, hi = boot_ci(y, p, auc_, n_boot)
    s, sp = sens_(y, p, threshold), spec_(y, p, threshold)
    slo, shi = boot_ci(y, p, sens_, n_boot, threshold=threshold)
    plo, phi = boot_ci(y, p, spec_, n_boot, threshold=threshold)
    print(f"    {label:<32s} N={len(y):6d} pos={int(y.sum()):5d} ({y.mean()*100:4.1f}%)  "
          f"AUROC={a:.4f} [{lo:.4f}-{hi:.4f}]  Sens={s:.4f}  Spec={sp:.4f}")
    return dict(n=int(len(y)), n_pos=int(y.sum()), prevalence=float(y.mean()),
                threshold=float(threshold), auroc=float(a), auroc_ci=[lo, hi],
                auroc_delong_var=float(var), sens=s, sens_ci=[slo, shi],
                spec=sp, spec_ci=[plo, phi],
                prob_min=float(p.min()), prob_max=float(p.max()))

def check_dynamic_range(p, cohort, model_name):
    rng = float(p.max() - p.min())
    if rng < 0.05:
        raise RuntimeError(
            f"{cohort}/{model_name}: probability range {rng:.4f} "
            f"(min={p.min():.4f}, max={p.max():.4f}). Signature of an unloaded model "
            "or total domain collapse. Refusing to emit metrics.")
    return rng


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def mimic_reference_stats(n_sample=512):
    # A cached copy of the reference statistics lets the external stages run on a
    # machine that does not hold the 6.5 GB MIMIC waveform array. The cache is
    # written by stage `mimic` and by --export_scale.
    cache = _E("EV_SCALE_NPZ", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "mimic_scale.npz"))
    if os.path.exists(cache):
        z = np.load(cache)
        mu, sd = z["mu"], z["sd"]
        print(f"    [mimic] scale reference loaded from {os.path.basename(cache)}: "
              f"mean {mu.min():+.4f}..{mu.max():+.4f}, std {sd.min():.4f}..{sd.max():.4f}")
        return mu, sd
    x_path = os.path.join(MIMIC_DIR, "X.npy")
    if not os.path.exists(x_path):
        if MIMIC_SCALE_MU is not None:
            mu = np.asarray(MIMIC_SCALE_MU, dtype=np.float64)
            sd = np.asarray(MIMIC_SCALE_SD, dtype=np.float64)
            print("    [mimic] scale reference from the built-in constant "
                  f"(mean {mu.min():+.6f}..{mu.max():+.6f}, "
                  f"std {sd.min():.6f}..{sd.max():.6f})")
            print("            MIMIC-IV-ECG is stored per-lead standardised, so this "
                  "reference is unit variance and zero mean by construction.")
            return mu, sd
        raise FileNotFoundError(
            f"No scale reference. Neither {cache} nor {x_path} exists.\n"
            "Falling back to a per-record z-score would silently change preprocessing "
            "and would not reproduce the published values. Generate the cache once on a "
            "machine that holds X.npy:\n"
            "    python run_external_validation.py --export_scale mimic_scale.npz\n"
            "then copy that 1 KB file next to this script.")
    X = np.load(x_path, mmap_mode="r")
    idx = np.sort(np.random.RandomState(SEED).choice(
        X.shape[0], size=min(n_sample, X.shape[0]), replace=False))
    sub = np.asarray(X[idx], dtype=np.float64)
    mu, sd = sub.mean(axis=(0, 1)), sub.std(axis=(0, 1))
    print(f"    [mimic] scale reference from {len(idx)} records: "
          f"mean {mu.min():+.4f}..{mu.max():+.4f}, std {sd.min():.4f}..{sd.max():.4f}")
    try:
        np.savez(cache, mu=mu, sd=sd)
        print(f"    [mimic] cached to {cache}")
    except OSError:
        pass
    return mu, sd


def reconstruct_12_from_8(x8, order=SNUH_8_ORDER):
    """x8: (B, 8, T) -> (B, 12, T) in LEAD_ORDER. III/aVR/aVL/aVF are exact
    linear combinations of I and II (Einthoven / Goldberger)."""
    idx = {n: i for i, n in enumerate(order)}
    I, II = x8[:, idx["I"]], x8[:, idx["II"]]
    out = np.empty((x8.shape[0], 12, x8.shape[2]), dtype=np.float32)
    out[:, 0] = I
    out[:, 1] = II
    out[:, 2] = II - I                 # III
    out[:, 3] = -(I + II) / 2.0        # aVR
    out[:, 4] = I - II / 2.0           # aVL
    out[:, 5] = II - I / 2.0           # aVF
    for k in range(6):
        out[:, 6 + k] = x8[:, idx[f"V{k+1}"]]
    return out


def check_lead_polarity(x12, n=256):
    """Physiological sanity: lead II should be R-positive dominant and aVR
    R-negative dominant. If not, the assumed 8-channel order is wrong."""
    s = x12[:n]
    def dom_sign(ch):
        a = s[:, ch, :]
        k = np.argmax(np.abs(a), axis=1)
        return np.sign(a[np.arange(len(a)), k])
    ii_pos = float((dom_sign(1) > 0).mean())
    avr_neg = float((dom_sign(3) < 0).mean())
    print(f"    [polarity] lead II positive-dominant {ii_pos*100:.0f}% | "
          f"aVR negative-dominant {avr_neg*100:.0f}%")
    if ii_pos < 0.6 or avr_neg < 0.6:
        print("    *** WARNING: polarity check failed. The assumed 8-channel order "
              f"{SNUH_8_ORDER} is probably wrong. Verify before trusting any metric. ***")
    return {"lead_II_positive_dominant": ii_pos, "aVR_negative_dominant": avr_neg}


def resample_time(x, target_len=5000):
    from scipy.signal import resample
    return x if x.shape[1] == target_len else resample(x, target_len, axis=1)


def apply_scale(x, mu, sd, src_mu=None, src_sd=None):
    """x: (B, 5000, 12). Map onto the MIMIC per-lead scale using FIXED cohort
    statistics (src_mu/src_sd) so that every batch is transformed identically.
    Falls back to per-record z-score when no MIMIC reference is available."""
    if mu is not None and sd is not None:
        cm = x.mean(axis=(0, 1)) if src_mu is None else src_mu
        cs = (x.std(axis=(0, 1)) if src_sd is None else src_sd) + 1e-8
        return ((x - cm) / cs) * sd + mu
    m = x.mean(axis=1, keepdims=True); s = x.std(axis=1, keepdims=True) + 1e-8
    return (x - m) / s


def cohort_scale_stats(sample_bt12):
    """sample_bt12: (n, 5000, 12) -> per-lead mean/std over the sample."""
    mu = sample_bt12.mean(axis=(0, 1)); sd = sample_bt12.std(axis=(0, 1))
    print(f"    [cohort] scale stats from {len(sample_bt12)} records: "
          f"mean {mu.min():+.2f}..{mu.max():+.2f}, std {sd.min():.2f}..{sd.max():.2f}")
    return mu, sd


def predict_stream(model, n, get_batch, lead_indices, batch=128):
    import torch, time                                       # PROGRESS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev); out = []
    t0 = time.time(); nb = (n + batch - 1) // batch
    print(f"    [device] {dev}  |  {nb} batches of {batch}", flush=True)
    with torch.no_grad():
        for k, i in enumerate(range(0, n, batch)):
            j = min(i + batch, n)
            xb = np.ascontiguousarray(get_batch(i, j)[:, :, lead_indices], dtype=np.float32)
            logits = model(torch.from_numpy(xb).to(dev))
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            if k % 20 == 0 or k == nb - 1:
                el = time.time() - t0
                eta = el / (k + 1) * (nb - k - 1)
                print(f"      batch {k+1}/{nb}  ({j}/{n})  "
                      f"경과 {el/60:.1f}분  남은 {eta/60:.1f}분", flush=True)
    return np.concatenate(out)


def subgroup_report(y, p, keys, name, results, thr=0.5, min_n=30, n_boot=500):
    for g in sorted(set(map(str, keys))):
        m = (np.asarray(list(map(str, keys))) == g)
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        results[f"{name}={g}"] = summarize(y[m], p[m], thr, f"  {name}={g}", n_boot)


# ----------------------------------------------------------------------------
# STAGE: probe
# ----------------------------------------------------------------------------
def find_on_drive(names, roots=(DRIVE,), max_depth=6):
    found = {n: [] for n in names}
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.count("/") - base >= max_depth:
                dirnames[:] = []; continue
            for n in names:
                if n in filenames:
                    found[n].append(os.path.join(dirpath, n))
    return found


def stage_probe():
    import pandas as pd
    print("="*78); print("SCHEMA PROBE"); print("="*78)
    targets = [m[1] for m in MODELS] + ["X.npy","y.npy","groups.npy","subjects.npy",
               "train_imbalanced_idx.npy","val_idx.npy","test_idx.npy","train_balanced_idx.npy"]
    hits = find_on_drive(targets)
    print("\n[0] File locations")
    for n in targets:
        if not hits[n]:
            print(f"    NOT FOUND  {n}")
        for q in hits[n]:
            print(f"    found      {n}  ({os.path.getsize(q):,} B)\n               {q}")
    print("\n[1] Configured paths")
    for label, p in [("MODEL_DIR", MODEL_DIR), ("MIMIC_DIR", MIMIC_DIR),
                     ("SPLIT_DIR", SPLIT_DIR), ("SNUH_CSV", SNUH_CSV),
                     ("ECHO_WAVE", ECHO_WAVE), ("ECHO_META", ECHO_META)]:
        print(f"    {'OK ' if os.path.exists(p) else 'MISSING'}  {label}: {p}")


# ----------------------------------------------------------------------------
# STAGE: mimic  (alignment + reproduction of the published internal numbers)
# ----------------------------------------------------------------------------
def stage_mimic():
    print("="*78); print("MIMIC-IV-ECG -- alignment check and reproduction"); print("="*78)
    X = np.load(os.path.join(MIMIC_DIR, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(MIMIC_DIR, "y.npy"), allow_pickle=True)
    g = np.load(os.path.join(MIMIC_DIR, "groups.npy"), allow_pickle=True).astype(str)
    print(f"    X {X.shape} | y {y.shape} | groups {g.shape}")
    assert X.shape[0] == len(y) == len(g), (
        f"MISALIGNED: X has {X.shape[0]} rows but y/groups have {len(y)}/{len(g)}. "
        "Point MIMIC_DIR at the self-consistent copy (preprocessed_27178_v2).")
    print(f"    [OK] X/y/groups aligned at {X.shape[0]} records")

    test_idx = np.load(os.path.join(SPLIT_DIR, "test_idx.npy"))
    y_t, g_t = y[test_idx].astype(int), g[test_idx]
    print(f"    test N={len(test_idx)}  positives={int(y_t.sum())} ({y_t.mean()*100:.2f}%)")
    print(f"    subgroup N: " + ", ".join(f"{k}={int((g_t==k).sum())}" for k in GROUP_NAMES))

    mu, sd = None, None                    # X is already in training scale
    out = {"n_test": int(len(test_idx)), "n_pos_test": int(y_t.sum()),
           "prevalence_test": float(y_t.mean()), "models": {}}
    for name, fn, lk in MODELS:
        path = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(path):
            print(f"    [skip] {name}: {path} missing"); continue
        print(f"\n  {name}")
        li = LEAD_SETS[lk]
        model = load_checkpoint_strict(path, len(li))
        get = lambda a, b: np.asarray(X[test_idx[a:b]], dtype=np.float32)
        p = predict_stream(model, len(test_idx), get, li)
        check_dynamic_range(p, "MIMIC", name)
        r = {"overall": summarize(y_t, p, 0.5, "overall @0.5"), "subgroups": {}}
        subgroup_report(y_t, p, g_t, "race", r["subgroups"])
        wm = (g_t == "White")
        for gg in GROUP_NAMES:
            if gg == "White":
                continue
            m = (g_t == gg)
            if m.sum() < 30 or len(np.unique(y_t[m])) < 2:
                continue
            _, _, pv = delong_z_vs_ref(y_t[m], p[m], y_t[wm], p[wm])
            r["subgroups"].setdefault(f"race={gg}", {})["p_vs_White"] = pv
            print(f"      {gg:>16s} vs White: p={pv:.4f}")
        if name in PUBLISHED:
            pa, ps, pp = PUBLISHED[name]
            da = abs(r["overall"]["auroc"] - pa)
            ds = abs(r["overall"]["sens"] - ps)
            ok = da < 0.005 and ds < 0.01
            r["reproduction"] = {"published": [pa, ps, pp], "delta_auroc": da,
                                 "delta_sens": ds, "reproduced": bool(ok)}
            print(f"    [repro] published AUROC={pa:.4f} Sens={ps:.4f} -> "
                  f"{'MATCH' if ok else 'MISMATCH'} (dAUROC={da:.4f}, dSens={ds:.4f})")
        out["models"][name] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    q = os.path.join(OUT_DIR, "mimic_reproduction.json")
    json.dump(out, open(q, "w"), indent=2); print(f"\n[written] {q}")


# ----------------------------------------------------------------------------
# STAGE: snuh
# ----------------------------------------------------------------------------
SNUH_BBB_STRICT = ["RBBB", "LBBB", "BFB(RBBB+LAFB)", "BFB(RBBB+LPFB)", "BFB"]
SNUH_BBB_BROAD  = SNUH_BBB_STRICT + ["LAFB", "LPFB"]

def stage_snuh(npz_path, label_mode, max_n, group_cols, batch):
    import pandas as pd
    print("="*78); print("SNUH LYDUS -- cross-institutional external validation"); print("="*78)

    df = pd.read_csv(SNUH_CSV, low_memory=False)
    if label_mode in ("qrsd", "complete"):
        qd = pd.to_numeric(df["qrsd"], errors="coerce")
        keep_q = qd.notna().values
        wide = (qd.fillna(-1).values >= 120)
        morph = df["bbb"].astype(str).str.strip().isin(SNUH_BBB_STRICT).values
        if label_mode == "qrsd":
            pos_set = ["qrsd >= 120 ms (machine-measured, matches EchoNext)"]
            y_all = wide.astype(int)
        else:
            pos_set = ["(" + "|".join(SNUH_BBB_STRICT) + ") AND qrsd >= 120 ms"]
            y_all = (morph & wide).astype(int)
        print(f"    CSV rows={len(df)}  label_mode={label_mode}")
        print(f"    qrsd: median={qd.median():.1f} "
              f"IQR={qd.quantile(.25):.1f}-{qd.quantile(.75):.1f} "
              f"range={qd.min():.1f}-{qd.max():.1f}  missing={int((~keep_q).sum())}")
        print(f"    morphology-positive={int(morph.sum())}  qrs-wide={int(wide.sum())}  "
              f"both={int((morph & wide).sum())}")
        df = df[keep_q].reset_index(drop=True); y_all = y_all[keep_q]
        print(f"    positives={int(y_all.sum())} ({y_all.mean()*100:.2f}%)   label: {pos_set[0]}")
    else:
        pos_set = SNUH_BBB_BROAD if label_mode == "broad" else SNUH_BBB_STRICT
        y_all = df["bbb"].astype(str).str.strip().isin(pos_set).astype(int).values
        print(f"    CSV rows={len(df)}  label_mode={label_mode}  positives={int(y_all.sum())} "
              f"({y_all.mean()*100:.2f}%)   classes counted: {pos_set}")
    assert y_all.sum() > 0 and y_all.sum() < len(y_all), "degenerate label mapping"

    z = np.load(npz_path, allow_pickle=True)
    keys, V = z["keys"].astype(str), z["vals"]
    print(f"    npz keys={keys.shape}  vals={V.shape} {V.dtype}")

    # --- explicit id alignment ---
    # npz keys are zero-padded 6-char strings ('000005'); CSV new_id is an int (5),
    # and new_id is NOT a contiguous 0..N-1 range (max 167,196). Match on integers.
    try:
        pos_of = {int(k): i for i, k in enumerate(keys)}
        csv_ids = df["new_id"].astype("int64").values
        mode = "integer"
    except (ValueError, TypeError):
        pos_of = {str(k).strip(): i for i, k in enumerate(keys)}
        csv_ids = df["new_id"].astype(str).str.strip().values
        mode = "string"
    sel = np.array([pos_of.get(c, -1) for c in csv_ids])
    ok = sel >= 0
    print(f"    id match ({mode}): {ok.sum()}/{len(csv_ids)} CSV rows found in npz keys")
    assert ok.mean() > 0.95, "id alignment failed; refusing to proceed on guesswork"
    df, y_all, sel = df[ok].reset_index(drop=True), y_all[ok], sel[ok]

    if max_n and max_n < len(sel):
        pick = np.sort(np.random.RandomState(SEED).choice(len(sel), max_n, replace=False))
        df, y_all, sel = df.iloc[pick].reset_index(drop=True), y_all[pick], sel[pick]
        print(f"    [subsample] random {max_n} (seed={SEED}) -- report as a subsample")

    n = len(sel)
    mu, sd = mimic_reference_stats()

    n_probe = min(1024, n)
    probe12 = reconstruct_12_from_8(np.asarray(V[sel[:n_probe]], dtype=np.float32))
    polarity = check_lead_polarity(probe12)
    probe_bt12 = resample_time(np.transpose(probe12, (0, 2, 1)), 5000)
    src_mu, src_sd = cohort_scale_stats(probe_bt12)
    del probe12, probe_bt12

    def get_batch(a, b):
        x8 = np.asarray(V[sel[a:b]], dtype=np.float32)        # (B, 8, T)
        x12 = reconstruct_12_from_8(x8)                       # (B, 12, T)
        x = np.transpose(x12, (0, 2, 1))                      # (B, T, 12)
        x = resample_time(x, 5000)
        return apply_scale(x, mu, sd, src_mu, src_sd).astype(np.float32)

    results = {"cohort": "SNUH LYDUS", "label_mode": label_mode,
               "positive_classes": pos_set, "n": int(n),
               "prevalence": float(y_all.mean()), "lead_polarity_check": polarity,
               "npz": os.path.basename(npz_path), "models": {}}
    prob_dump = {}                                            # PROB_DUMP
    cache = os.path.join(OUT_DIR, "snuh_prob_cache")          # RESUMABLE
    os.makedirs(cache, exist_ok=True)
    for name, fn, lk in MODELS:
        path = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(path):
            print(f"    [skip] {name}: checkpoint missing"); continue
        print(f"\n  {name}")
        li = LEAD_SETS[lk]
        cf = os.path.join(cache, fn.replace(".pt", "") + f"_n{n}.npy")
        if os.path.exists(cf):
            p = np.load(cf)
            if len(p) == n:
                print(f"    [cache] 이전 실행 결과 재사용: {os.path.basename(cf)}")
            else:
                p = None
        else:
            p = None
        if p is None:
            model = load_checkpoint_strict(path, len(li))
            p = predict_stream(model, n, get_batch, li, batch=batch)
            np.save(cf, p.astype(np.float32))
            print(f"    [saved] {os.path.basename(cf)}  <- 여기까지는 다시 계산하지 않습니다")
            del model
        prob_dump[name] = p.astype(np.float32)                # PROB_DUMP
        check_dynamic_range(p, "SNUH", name)
        thr = youden_threshold(y_all, p)
        r = {"overall_at_0.5": summarize(y_all, p, 0.5, "overall @0.5"),
             "overall_at_youden": summarize(y_all, p, thr, f"overall @J={thr:.4f}"),
             "subgroups": {}}
        for gc in [c.strip() for c in group_cols.split(",") if c.strip()]:
            if gc not in df.columns:
                print(f"      [skip subgroup] '{gc}' not a column"); continue
            keysg = df[gc].astype(str).values
            if gc == "age":
                a = pd.to_numeric(df[gc], errors="coerce")
                keysg = pd.cut(a, [0,50,65,200], labels=["<50","50-65",">65"]).astype(str).values
            subgroup_report(y_all, p, keysg, gc, r["subgroups"])
        results["models"][name] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    q = os.path.join(OUT_DIR, f"snuh_results_{label_mode}.json")
    json.dump(results, open(q, "w"), indent=2); print(f"\n[written] {q}")

    if prob_dump:                                             # PROB_DUMP
        pq = os.path.join(OUT_DIR, "snuh_probs.npz")
        np.savez_compressed(
            pq, new_id=df["new_id"].astype("int64").values,
            model_names=np.array(list(prob_dump.keys())),
            **{f"p{i}": v for i, v in enumerate(prob_dump.values())})
        print(f"[written] {pq}  ({os.path.getsize(pq)/1e6:.2f} MB, "
              f"{len(prob_dump)} models x {n} records)")
        print("  -> 이 파일 하나로 어떤 라벨 정의든 재채점할 수 있습니다 (모델 재실행 불필요)")


# ----------------------------------------------------------------------------
# STAGE: echonext
# ----------------------------------------------------------------------------
def stage_echonext(label_col, split, group_cols, batch, label_thresh=None):
    import pandas as pd
    print("="*78); print("EchoNext -- cross-task validation (BBB model vs echo label)"); print("="*78)

    meta = pd.read_csv(ECHO_META, low_memory=False)
    m = meta[meta["split"] == split].reset_index(drop=True)
    W = np.load(ECHO_WAVE, mmap_mode="r")
    print(f"    waveforms {W.shape} {W.dtype} | metadata split='{split}' rows={len(m)}")
    assert W.shape[0] == len(m), (
        f"row mismatch: waveforms {W.shape[0]} vs metadata {len(m)}")

    raw = pd.to_numeric(m[label_col], errors="coerce").values
    keep = ~np.isnan(raw)
    idx = np.where(keep)[0]
    raw = raw[keep]; m = m[keep].reset_index(drop=True)
    if label_thresh is not None:
        y = (raw >= label_thresh).astype(int)
        print(f"    label: {label_col} >= {label_thresh}  (continuous -> binary)")
        print(f"    {label_col}: median={np.median(raw):.1f} "
              f"IQR={np.percentile(raw,25):.1f}-{np.percentile(raw,75):.1f} "
              f"range={raw.min():.1f}-{raw.max():.1f}")
    else:
        u = np.unique(raw)
        assert set(u.tolist()) <= {0.0, 1.0}, (
            f"'{label_col}' is not binary (values {u[:8]}); pass --label_thresh")
        y = raw.astype(int)
    print(f"    dropped {int((~keep).sum())} rows with missing {label_col}")
    print(f"    usable N={len(y)}  positives={int(y.sum())} ({y.mean()*100:.2f}%)")
    assert 0 < y.sum() < len(y), "degenerate label on EchoNext; refusing to proceed"

    mu, sd = mimic_reference_stats()

    n_probe = min(1024, len(idx))
    ws = np.asarray(W[idx[:n_probe]], dtype=np.float32)
    if ws.ndim == 4 and ws.shape[1] == 1:
        ws = ws[:, 0]
    src_mu, src_sd = cohort_scale_stats(resample_time(ws, 5000))
    del ws

    def get_batch(a, b):
        w = np.asarray(W[idx[a:b]], dtype=np.float32)         # (B,1,2500,12)
        if w.ndim == 4 and w.shape[1] == 1:
            w = w[:, 0]
        w = resample_time(w, 5000)                            # 250 -> 500 Hz
        return apply_scale(w, mu, sd, src_mu, src_sd).astype(np.float32)

    results = {"cohort": "EchoNext", "split": split, "label_col": label_col,
               "label_thresh": label_thresh,
               "task": "cross-task: BBB-trained model scored against an echo label",
               "n": int(len(y)), "prevalence": float(y.mean()), "models": {}}
    prob_dump = {}                                            # PROB_DUMP
    for name, fn, lk in MODELS:
        path = os.path.join(MODEL_DIR, fn)
        if not os.path.exists(path):
            print(f"    [skip] {name}: checkpoint missing"); continue
        print(f"\n  {name}")
        li = LEAD_SETS[lk]
        model = load_checkpoint_strict(path, len(li))
        p = predict_stream(model, len(y), get_batch, li, batch=batch)
        prob_dump[name] = p.astype(np.float32)                # PROB_DUMP
        check_dynamic_range(p, "EchoNext", name)
        thr = youden_threshold(y, p)
        r = {"overall_at_0.5": summarize(y, p, 0.5, "overall @0.5"),
             "overall_at_youden": summarize(y, p, thr, f"overall @J={thr:.4f}"),
             "subgroups": {}}
        for gc in [c.strip() for c in group_cols.split(",") if c.strip()]:
            if gc not in m.columns:
                print(f"      [skip subgroup] '{gc}' not a column"); continue
            keysg = m[gc].astype(str).values
            if gc == "age_at_ecg":
                a = pd.to_numeric(m[gc], errors="coerce")
                keysg = pd.cut(a, [0,50,65,200], labels=["<50","50-65",">65"]).astype(str).values
            subgroup_report(y, p, keysg, gc, r["subgroups"])
        results["models"][name] = r

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = label_col if label_thresh is None else f"{label_col}_ge{label_thresh:g}"
    q = os.path.join(OUT_DIR, f"echonext_results_{tag}.json")
    json.dump(results, open(q, "w"), indent=2); print(f"\n[written] {q}")

    if prob_dump:                                             # PROB_DUMP
        pq = os.path.join(OUT_DIR, "echonext_probs.npz")
        np.savez_compressed(
            pq, ecg_key=m["ecg_key"].astype(str).values if "ecg_key" in m.columns
                        else np.arange(len(y)),
            model_names=np.array(list(prob_dump.keys())),
            **{f"p{i}": v for i, v in enumerate(prob_dump.values())})
        print(f"[written] {pq}  ({os.path.getsize(pq)/1e6:.2f} MB)")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe","mimic","snuh","echonext"])
    ap.add_argument("--export_scale", default=None,
                    help="write the MIMIC per-lead scale reference to this .npz and exit")
    ap.add_argument("--npz", default=SNUH_NPZ_500, help="SNUH waveform npz (500 Hz default)")
    ap.add_argument("--label_mode", default="strict",
                    choices=["strict","broad","qrsd","complete"],
                    help="strict = RBBB/LBBB/BFB only; broad = also LAFB/LPFB")
    ap.add_argument("--max_n", type=int, default=0, help="SNUH random subsample (0 = all)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--group_cols", default="")
    ap.add_argument("--label_col", default="lvef_lte_45_flag")
    ap.add_argument("--label_thresh", type=float, default=None,
                    help="binarize a continuous label_col at >= this value "
                         "(e.g. --label_col qrs_duration --label_thresh 120)")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)

    if a.export_scale:
        os.environ.pop("EV_SCALE_NPZ", None)
        globals()["_FORCE_RECOMPUTE"] = True
        x_path = os.path.join(MIMIC_DIR, "X.npy")
        X = np.load(x_path, mmap_mode="r")
        idx = np.sort(np.random.RandomState(SEED).choice(
            X.shape[0], size=min(512, X.shape[0]), replace=False))
        sub = np.asarray(X[idx], dtype=np.float64)
        mu, sd = sub.mean(axis=(0, 1)), sub.std(axis=(0, 1))
        np.savez(a.export_scale, mu=mu, sd=sd)
        print(f"[written] {a.export_scale}  ({os.path.getsize(a.export_scale)} bytes)")
        print("mu =", [round(v, 6) for v in mu.tolist()])
        print("sd =", [round(v, 6) for v in sd.tolist()])
        return
    if not a.stage:
        raise SystemExit("--stage is required unless --export_scale is given")

    if a.stage == "probe":
        stage_probe()
    elif a.stage == "mimic":
        stage_mimic()
    elif a.stage == "snuh":
        stage_snuh(a.npz, a.label_mode, a.max_n,
                   a.group_cols or "sex,age", a.batch)
    else:
        stage_echonext(a.label_col, a.split,
                       a.group_cols or "race_ethnicity,sex,age_at_ecg", a.batch,
                       a.label_thresh)


if __name__ == "__main__":
    main()
