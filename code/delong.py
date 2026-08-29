import numpy as np
from scipy.stats import norm

def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x)
    T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def delong_auc_var(y, p):
    """Single-model AUC and its DeLong variance."""
    y = np.asarray(y).astype(int); p = np.asarray(p, dtype=float)
    pos = p[y == 1]; neg = p[y == 0]
    m, n = len(pos), len(neg)
    tx = _midrank(pos); ty = _midrank(neg); tz = _midrank(np.r_[pos, neg])
    auc = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:m] - tx) / n          # structural components, positives
    v10 = 1.0 - (tz[m:] - ty) / m    # structural components, negatives
    var = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    return float(auc), float(var)

def independent_auc_test(y1, p1, y2, p2):
    """z-test for two AUCs measured on DISJOINT samples."""
    a1, v1 = delong_auc_var(y1, p1)
    a2, v2 = delong_auc_var(y2, p2)
    se = np.sqrt(v1 + v2)
    z = (a1 - a2) / se if se > 0 else np.nan
    return a1, a2, float(z), float(2 * norm.sf(abs(z)))
