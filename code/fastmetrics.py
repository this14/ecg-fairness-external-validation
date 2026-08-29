import numpy as np
from scipy.stats import rankdata

SEED = 20260824


def fast_auc(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def sens_spec(y, p, thr):
    pred = p >= thr
    pos = y == 1
    tp = np.count_nonzero(pred & pos); fn = np.count_nonzero(~pred & pos)
    tn = np.count_nonzero(~pred & ~pos); fp = np.count_nonzero(pred & ~pos)
    s = tp / (tp + fn) if (tp + fn) else float("nan")
    sp = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(s), float(sp)


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def boot(y, p, thr, n_boot=500, seed=SEED):
    """Percentile bootstrap for AUROC, sensitivity and specificity together."""
    rng = np.random.RandomState(seed)
    n = len(y)
    A, S, P = [], [], []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        yb = y[i]
        if yb.min() == yb.max():
            continue
        pb = p[i]
        A.append(fast_auc(yb, pb))
        s, sp = sens_spec(yb, pb, thr)
        S.append(s); P.append(sp)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(A), q(S), q(P)


def summarize(y, p, thr, label, n_boot=500, quiet=False):
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    assert len(np.unique(y)) == 2, f"{label}: degenerate labels"
    a = fast_auc(y, p)
    s, sp = sens_spec(y, p, thr)
    (alo, ahi), (slo, shi), (plo, phi) = boot(y, p, thr, n_boot)
    if not quiet:
        print(f"    {label:<34s} N={len(y):6d} pos={int(y.sum()):5d} "
              f"({y.mean()*100:5.2f}%)  AUROC={a:.4f} [{alo:.4f}-{ahi:.4f}]  "
              f"Sens={s:.4f}  Spec={sp:.4f}")
    return dict(n=int(len(y)), n_pos=int(y.sum()), prevalence=float(y.mean()),
                threshold=float(thr), auroc=a, auroc_ci=[alo, ahi],
                sens=s, sens_ci=[slo, shi], spec=sp, spec_ci=[plo, phi])


def subgroups(y, p, keys, name, thr, res, min_n=30, n_boot=300):
    keys = np.asarray(list(map(str, keys)))
    for g in sorted(set(keys)):
        m = keys == g
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        res[f"{name}={g}"] = summarize(y[m], p[m], thr, f"  {name}={g}", n_boot)
