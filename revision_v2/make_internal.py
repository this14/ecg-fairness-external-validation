import json, os
from math import comb
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
B = '/home/this14/ev_out'
CF = ['12-lead', '3-lead', '1-lead']
RACES = ['Asian', 'Black', 'Hispanic/Latino', 'Other', 'White']
D = {}
for s in range(42, 52):
    f = f'{B}/seed{s}/mimic_reproduction.json'
    if os.path.exists(f):
        D[s] = json.load(open(f))
print('seeds loaded:', sorted(D))
def ms(v):
    m = sum(v) / len(v)
    return m, (sum((a - m) ** 2 for a in v) / (len(v) - 1)) ** .5
def sign_p(v):
    n = sum(1 for x in v if x != 0); k = sum(1 for x in v if x < 0)
    if n == 0: return 1.0
    t = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(t + 1)) / 2 ** n)
def wil_p(v):
    if wilcoxon is None or all(x == 0 for x in v): return float('nan')
    try: return float(wilcoxon(v).pvalue)
    except Exception: return float('nan')
def num(x):
    return None if x is None or x != x else float(x)
def ov(d, t, c, k):
    try: return num(d['models'][f'{t} ({c})']['overall'][k])
    except KeyError: return None
def sgv(d, t, c, race, k):
    sg = d['models'][f'{t} ({c})']['subgroups']
    for key in sg:
        if key.endswith('=' + race) or key == race:
            return num(sg[key].get(k))
    return None
def line(*a):
    print(','.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in a))

print('\n## internal_overall (test split, threshold 0.5): config,metric,std_mean,std_sd,rw_mean,rw_sd,delta_mean,delta_sd,n_neg,n,sign_p,wilcoxon_p')
DL = {}
for c in CF:
    for k in ['auroc', 'sens', 'spec']:
        S = []; R = []; X = []
        for d in D.values():
            a = ov(d, 'Standard', c, k); b = ov(d, 'Reweighted', c, k)
            if a is None or b is None: continue
            S.append(a); R.append(b); X.append(b - a)
        DL[(c, k)] = X
        a1, a2 = ms(S); b1, b2 = ms(R); g1, g2 = ms(X)
        line(c, k, a1, a2, b1, b2, g1, g2, sum(x < 0 for x in X), len(X), sign_p(X), wil_p(X))

print('\n## internal_trend (difference in delta vs 12-lead): comparison,metric,mean,sd,n_neg,n,sign_p,wilcoxon_p')
for c in ['3-lead', '1-lead']:
    for k in ['auroc', 'sens']:
        X = [p - q for p, q in zip(DL[(c, k)], DL[('12-lead', k)])]
        g1, g2 = ms(X)
        line(f'{c} vs 12-lead', k, g1, g2, sum(x < 0 for x in X), len(X), sign_p(X), wil_p(X))

def measures(d, t, c):
    m = {}
    for k in ['auroc', 'sens', 'spec']:
        v = [sgv(d, t, c, r, k) for r in RACES]
        if None in v: return None
        m['bw_' + k] = v[1] - v[4]
        m['spread_' + k] = max(v) - min(v)
    return m
print('\n## internal_disparity: config,measure,std_mean,std_sd,rw_mean,rw_sd,change_mean,change_sd,narrowed,n,sign_p,wilcoxon_p')
print('# bw_* = Black minus White (change = |RW| - |Std|); spread_* = max-min across 5 race groups (change = RW - Std); negative change = narrowed')
for c in CF:
    rows = {}
    for d in D.values():
        a = measures(d, 'Standard', c); b = measures(d, 'Reweighted', c)
        if a is None or b is None: continue
        for key in a:
            rows.setdefault(key, []).append((a[key], b[key]))
    for key, pr in rows.items():
        S = [p[0] for p in pr]; R = [p[1] for p in pr]
        X = [abs(r) - abs(s) for s, r in pr] if key.startswith('bw_') else [r - s for s, r in pr]
        a1, a2 = ms(S); b1, b2 = ms(R); g1, g2 = ms(X)
        line(c, key, a1, a2, b1, b2, g1, g2, sum(x < 0 for x in X), len(X), sign_p(X), wil_p(X))

print('\n## reproduction block of seed 42 (published vs retrained)')
d0 = D[min(D)]
for c in CF:
    for t in ['Standard', 'Reweighted']:
        print(f'{t} ({c}):', json.dumps(d0['models'][f'{t} ({c})'].get('reproduction'))[:300])
