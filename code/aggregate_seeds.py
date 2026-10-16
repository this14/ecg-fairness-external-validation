import json, sys, os, csv, math

BASE = '/home/this14/ev_out'
SEEDS = range(42, 52)

def flat(x, p=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, p + (str(k),))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        if not (isinstance(x, float) and math.isnan(x)):
            yield p, float(x)

def ms(v):
    n = len(v); m = sum(v) / n
    sd = (sum((a - m) ** 2 for a in v) / (n - 1)) ** 0.5 if n > 1 else float('nan')
    return m, sd

name = sys.argv[1]
data = {}
for s in SEEDS:
    f = f'{BASE}/seed{s}/{name}'
    if os.path.exists(f):
        data[s] = dict(flat(json.load(open(f))))
    else:
        print('MISSING', f)
print(f'{name}: {len(data)} seeds loaded: {sorted(data)}')

paths = sorted(set().union(*[set(d) for d in data.values()]))
stem = name[:-5]

with open(f'{BASE}/agg_{stem}.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['path', 'n_seeds', 'mean', 'sd', 'min', 'max'])
    for p in paths:
        v = [d[p] for d in data.values() if p in d]
        m, sd = ms(v); w.writerow(['/'.join(p), len(v), m, sd, min(v), max(v)])

drows = []
for p in paths:
    segs = [i for i, s in enumerate(p) if s.startswith('Reweighted')]
    if not segs:
        continue
    q = tuple(s.replace('Reweighted', 'Standard', 1) if s.startswith('Reweighted') else s for s in p)
    dl = [d[p] - d[q] for d in data.values() if p in d and q in d]
    if not dl:
        continue
    m, sd = ms(dl)
    drows.append(['/'.join(p), len(dl), m, sd, sum(x > 0 for x in dl), sum(x < 0 for x in dl)])

with open(f'{BASE}/delta_{stem}.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['path(Reweighted-Standard)', 'n_seeds', 'mean_delta', 'sd_delta', 'n_pos', 'n_neg'])
    w.writerows(drows)

print('--- overall Reweighted - Standard (mean +/- SD, #seeds delta>0 / <0) ---')
for r in drows:
    last = r[0].split('/')[-1]
    if 'overall_at_0.5' in r[0] and last in ('sens', 'spec', 'auroc'):
        print(f'{r[0]:60s} {r[2]:+.4f} +/- {r[3]:.4f}  (+{r[4]}/-{r[5]})')
print(f'saved: {BASE}/agg_{stem}.csv , {BASE}/delta_{stem}.csv')
