import json, os, csv
from math import comb
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
B='/home/this14/ev_out'
FILES={'EchoNext-QRS':'echonext_results_qrs_duration_ge120.json','EchoNext-LVEF':'echonext_results_lvef_lte_45_flag.json','SNUH-BBB':'snuh_results_complete.json'}
CF=['12-lead','3-lead','1-lead']
M=[('overall_at_0.5','auroc'),('overall_at_0.5','sens'),('overall_at_0.5','spec'),('overall_at_youden','sens'),('overall_at_youden','spec')]
def load(nm):
    return [json.load(open(f'{B}/seed{s}/{nm}')) for s in range(42,52) if os.path.exists(f'{B}/seed{s}/{nm}')]
def ms(v):
    m=sum(v)/len(v)
    return m,(sum((a-m)**2 for a in v)/(len(v)-1))**.5
def sign_p(v):
    n=sum(1 for x in v if x!=0); k=sum(1 for x in v if x<0)
    if n==0: return 1.0
    t=min(k,n-k)
    return min(1.0,2*sum(comb(n,i) for i in range(t+1))/2**n)
def wil_p(v):
    if wilcoxon is None or all(x==0 for x in v): return float('nan')
    try: return float(wilcoxon(v).pvalue)
    except Exception: return float('nan')
def fx(d,t,c,sec,k):
    try: x=d['models'][f'{t} ({c})'][sec][k]
    except KeyError: return None
    return None if x is None or x!=x else float(x)
def sg(d,t,c,g,k):
    try: x=d['models'][f'{t} ({c})']['subgroups'][g][k]
    except KeyError: return None
    return None if x is None or x!=x else float(x)
T1=[['cohort','config','threshold','metric','std_mean','std_sd','rw_mean','rw_sd','delta_mean','delta_sd','n_neg','n','sign_p','wilcoxon_p']]
T2=[['cohort','pair','config','metric','gapStd_mean','gapStd_sd','gapRW_mean','gapRW_sd','dabsgap_mean','dabsgap_sd','narrowed','n','sign_p','wilcoxon_p']]
T3=[['cohort','comparison','threshold','metric','mean','sd','n_neg','n','sign_p','wilcoxon_p']]
for cn,nm in FILES.items():
    D=load(nm)
    dl={}
    for c in CF:
        for sec,k in M:
            S=[];R=[];X=[]
            for d in D:
                s=fx(d,'Standard',c,sec,k); r=fx(d,'Reweighted',c,sec,k)
                if s is None or r is None: continue
                S.append(s);R.append(r);X.append(r-s)
            if len(X)<2: continue
            dl[(c,sec,k)]=X
            a,b=ms(S);e,f=ms(R);g,h=ms(X)
            T1.append([cn,c,sec,k,a,b,e,f,g,h,sum(x<0 for x in X),len(X),sign_p(X),wil_p(X)])
    for c in ['3-lead','1-lead']:
        for sec,k in [('overall_at_0.5','auroc'),('overall_at_0.5','sens'),('overall_at_youden','sens')]:
            if (c,sec,k) in dl and ('12-lead',sec,k) in dl:
                X=[p-q for p,q in zip(dl[(c,sec,k)],dl[('12-lead',sec,k)])]
                g,h=ms(X)
                T3.append([cn,f'{c} minus 12-lead (delta)',sec,k,g,h,sum(x<0 for x in X),len(X),sign_p(X),wil_p(X)])
    if cn.startswith('Echo'):
        P=[('race_ethnicity=black','race_ethnicity=white'),('sex=female','sex=male'),('age_at_ecg=>65','age_at_ecg=<50')]
    else:
        P=[('sex=F','sex=M'),('age=>65','age=<50')]
    for A,Bg in P:
        for c in CF:
            for k in ['auroc','sens','spec']:
                gs=[];gr=[];dd=[]
                for d in D:
                    v=[sg(d,t,c,g,k) for t in ('Standard','Reweighted') for g in (A,Bg)]
                    if None in v: continue
                    g1=v[0]-v[1]; g2=v[2]-v[3]
                    gs.append(g1);gr.append(g2);dd.append(abs(g2)-abs(g1))
                if len(dd)<2: continue
                a,b=ms(gs);e,f=ms(gr);g,h=ms(dd)
                T2.append([cn,f'{A} - {Bg}',c,k,a,b,e,f,g,h,sum(x<0 for x in dd),len(dd),sign_p(dd),wil_p(dd)])
for name,T in (('table1_overall',T1),('table2_gaps',T2),('table3_trend',T3)):
    T=[T[0]]+[[round(x,4) if isinstance(x,float) else x for x in r] for r in T[1:]]
    with open(f'{B}/{name}.csv','w',newline='') as fh:
        csv.writer(fh).writerows(T)
    print('##',name)
    for r in T: print(','.join(str(x) for x in r))
