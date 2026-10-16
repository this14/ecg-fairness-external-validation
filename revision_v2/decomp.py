import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import run_external_validation as M
OUT = '/home/this14/ev_out'
df = pd.read_csv(M.SNUH_CSV, low_memory=False)
qd = pd.to_numeric(df['qrsd'], errors='coerce'); keep = qd.notna().values
bb = df['bbb'].astype(str).str.strip()
wide = (qd.fillna(-1).values >= 120)
yB = bb.isin(M.SNUH_BBB_STRICT).values[keep].astype(int)
yBr = bb.isin(M.SNUH_BBB_BROAD).values[keep].astype(int)
yW = wide[keep].astype(int)
n = int(keep.sum()); print('n', n, 'bbb strict prev', yB.mean(), 'broad', yBr.mean(), 'qrs120 prev', yW.mean())
print('share of strict-bbb positives with QRS>=120:', yW[yB == 1].mean())
meta = pd.read_csv(M.ECHO_META, low_memory=False)
m = meta[meta['split'] == 'val'].reset_index(drop=True)
r = pd.to_numeric(m['qrs_duration'], errors='coerce').values; r = r[~np.isnan(r)]
yE = (r >= 120).astype(int)
CF = [('12-lead', '12lead'), ('3-lead', '3lead'), ('1-lead', '1lead')]
TY = [('Standard', 'vanilla'), ('Reweighted', 'debiased')]
rows = []
for S in range(42, 52):
    en = np.load(f'{OUT}/seed{S}/echonext_probs.npz', allow_pickle=True)
    names = [str(a) for a in en['model_names']]
    for c, lk in CF:
        for t, fn in TY:
            ps = np.load(f'{OUT}/seed{S}/snuh_prob_cache/model_{fn}_{lk}_n{n}.npy')
            pe = en['p%d' % names.index(f'{t} ({c})')]
            aE, aW, aB, aBr = M.auc_(yE, pe), M.auc_(yW, ps), M.auc_(yB, ps), M.auc_(yBr, ps)
            rows.append(dict(seed=S, cfg=c, type=t, aE=aE, aW=aW, aB=aB, aBr=aBr, gap=aE-aB, label=aW-aB, domain=aE-aW, gap_broad=aE-aBr, label_broad=aW-aBr))
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/decomp_rows.csv', index=False)
print('type,cfg,aE,aW,aB,gap,label,domain,label_share_pct,gap_broad,label_broad (mean; sd for gap/label/domain)')
for t, _ in TY:
    for c, _ in CF:
        a = R[(R.type == t) & (R.cfg == c)]
        mm = a[['aE','aW','aB','gap','label','domain','gap_broad','label_broad']].mean()
        sd = a[['gap','label','domain']].std(ddof=1)
        print(f"{t},{c},{mm.aE:.4f},{mm.aW:.4f},{mm.aB:.4f},{mm.gap:.4f}({sd.gap:.4f}),{mm.label:.4f}({sd.label:.4f}),{mm.domain:.4f}({sd.domain:.4f}),{100*mm.label/mm.gap:.0f},{mm.gap_broad:.4f},{mm.label_broad:.4f}")
