import os, sys, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.getcwd())
import run_external_validation as M
OUT = '/home/this14/ev_out'
CODE = '/home/this14/ecg-fairness-external-validation/code'
CF = [('12-lead', '12lead'), ('3-lead', '3lead'), ('1-lead', '1lead')]
TY = [('Standard', 'vanilla'), ('Reweighted', 'debiased')]
X = np.load(os.path.join(M.MIMIC_DIR, 'X.npy'), mmap_mode='r')
y = np.load(os.path.join(M.MIMIC_DIR, 'y.npy'), allow_pickle=True).astype(int)
vi = np.load(os.path.join(M.SPLIT_DIR, 'val_idx.npy'))
yv = y[vi]
print('val N', len(vi), 'pos', int(yv.sum()))
df = pd.read_csv(M.SNUH_CSV, low_memory=False)
qd = pd.to_numeric(df['qrsd'], errors='coerce'); keep = qd.notna().values
morph = df['bbb'].astype(str).str.strip().isin(M.SNUH_BBB_STRICT).values
ys = ((morph) & (qd.fillna(-1).values >= 120)).astype(int)[keep]
print('SNUH n', len(ys), 'prev', ys.mean())
meta = pd.read_csv(M.ECHO_META, low_memory=False)
print('echo splits', meta['split'].unique())
sp = [s for s in ['val', 'validation', 'valid'] if s in set(meta['split'])][0]
m = meta[meta['split'] == sp].reset_index(drop=True)
def ey(col, thr=None):
    r = pd.to_numeric(m[col], errors='coerce').values; r = r[~np.isnan(r)]
    return (r >= thr).astype(int) if thr else r.astype(int)
ye = {'EchoNext-QRS': ey('qrs_duration', 120), 'EchoNext-LVEF': ey('lvef_lte_45_flag')}
print({k: (len(v), int(v.sum())) for k, v in ye.items()})
rows = []
for S in range(42, 52):
    md = f'{CODE}/flat/seed{S}'
    en = np.load(f'{OUT}/seed{S}/echonext_probs.npz', allow_pickle=True)
    names = [str(a) for a in en['model_names']]
    for c, lk in CF:
        li = M.LEAD_SETS[lk]
        for t, fn in TY:
            mod = M.load_checkpoint_strict(f'{md}/model_{fn}_{lk}.pt', len(li))
            get = lambda a, b: np.asarray(X[vi[a:b]], dtype=np.float32)
            pv = M.predict_stream(mod, len(vi), get, li)
            th = M.youden_threshold(yv, pv)
            ps = np.load(f'{OUT}/seed{S}/snuh_prob_cache/model_{fn}_{lk}_n{len(ys)}.npy')
            pe = en['p%d' % names.index(f'{t} ({c})')]
            for coh, yy, p in [('SNUH-BBB', ys, ps), ('EchoNext-QRS', ye['EchoNext-QRS'], pe), ('EchoNext-LVEF', ye['EchoNext-LVEF'], pe)]:
                assert len(p) == len(yy), (coh, len(p), len(yy))
                rows.append(dict(seed=S, cfg=c, type=t, cohort=coh, thr=th, auroc=M.auc_(yy, p), sens=M.sens_(yy, p, th), spec=M.spec_(yy, p, th)))
    print('seed', S, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/youden_ext_rows.csv', index=False)
try:
    j = json.load(open(f'{OUT}/seed42/snuh_results_complete.json'))
    a = j['models']['Standard (12-lead)']['overall_at_0.5']['auroc']
    b = R[(R.seed == 42) & (R.cohort == 'SNUH-BBB') & (R.cfg == '12-lead') & (R.type == 'Standard')].auroc.iloc[0]
    print('CHECK SNUH auroc json vs mine', a, b)
except Exception as e:
    print('check skipped', e)
print('cohort,cfg,metric,std_mean,rw_mean,delta_mean,delta_sd,n_neg,n,sign_p,wilcoxon_p,mean_thr_std,mean_thr_rw')
for coh in ['EchoNext-QRS', 'EchoNext-LVEF', 'SNUH-BBB']:
    for c, _ in CF:
        for me in ['sens', 'spec']:
            a = R[(R.cohort == coh) & (R.cfg == c)].pivot(index='seed', columns='type', values=me)
            d = a['Reweighted'] - a['Standard']
            nn = int((d < 0).sum()); nz = int((d != 0).sum())
            try: w = stats.wilcoxon(d).pvalue
            except Exception: w = float('nan')
            th = R[(R.cohort == coh) & (R.cfg == c)].groupby('type').thr.mean()
            print(f"{coh},{c},{me},{a['Standard'].mean():.4f},{a['Reweighted'].mean():.4f},{d.mean():.4f},{d.std(ddof=1):.4f},{nn},{len(d)},{stats.binomtest(nn, nz, 0.5).pvalue:.4f},{w:.4f},{th['Standard']:.3f},{th['Reweighted']:.3f}")
