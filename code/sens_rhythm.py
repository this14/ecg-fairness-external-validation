import numpy as np, pandas as pd, json
import fastmetrics as fm
U="/root/.claude/uploads/4eea52cb-5125-5e39-a706-93f7fe1368e1"
rng=np.random.RandomState(20260824)

slim=pd.read_csv(f"{U}/1b509c09-snuh_slim.csv",low_memory=False)
rh=pd.read_csv(f"{U}/00f6f5b5-snuh_rhythm.csv",low_memory=False)
assert np.array_equal(slim["new_id"].values, rh["new_id"].values), "new_id 순서 불일치"
z=np.load("/home/claude/snuh_probs.npz",allow_pickle=True)
assert np.array_equal(z["new_id"], slim["new_id"].astype("int64").values)
names=z["model_names"].tolist(); P={n:z[f"p{i}"].astype(float) for i,n in enumerate(names)}
print("정렬 확인 OK  n =", len(slim))

STRICT=["RBBB","LBBB","BFB(RBBB+LAFB)","BFB(RBBB+LPFB)","BFB"]
bbb=slim["bbb"].astype(str).str.strip().isin(STRICT).values
qd=pd.to_numeric(slim["qrsd"],errors="coerce").values; hq=~np.isnan(qd)
wide=np.where(hq,qd>=120,False)
rhy=rh["rhythm"].astype(str).str.strip().values
pace = rhy=="Possible pacing"
afib = np.isin(rhy,["A.fib","A.flutter"])
print(f"조율 {pace.sum()} ({pace.mean()*100:.2f}%) | 심방세동·조동 {afib.sum()} ({afib.mean()*100:.2f}%)")

LAB={"strict":(bbb.astype(int),np.ones(len(slim),bool)),
     "qrsd120":(wide.astype(int),hq),
     "complete":((bbb&wide).astype(int),hq)}
COH={"전체":np.ones(len(slim),bool),
     "조율 제외":~pace,
     "조율+세동 제외":~(pace|afib)}

def paired(y,a,b,B=1000):
    d0=fm.fast_auc(y,b)-fm.fast_auc(y,a); v=[]
    for _ in range(B):
        i=rng.randint(0,len(y),len(y))
        if y[i].min()==y[i].max(): continue
        v.append(fm.fast_auc(y[i],b[i])-fm.fast_auc(y[i],a[i]))
    v=np.array(v); return d0, *np.percentile(v,[2.5,97.5]), max(2*min((v>=0).mean(),(v<=0).mean()),1/B)

out={}
for lab,(y_all,mask0) in LAB.items():
    print(f"\n{'='*74}\nLABEL = {lab}\n{'='*74}")
    for cn,cm in COH.items():
        m=mask0&cm; y=y_all[m]
        if len(np.unique(y))<2: continue
        row=f"  {cn:16s} N={m.sum():6d}  prev={y.mean()*100:5.2f}%  "
        for L in ["12-lead","3-lead"]:
            a=fm.fast_auc(y,P[f"Standard ({L})"][m]); b=fm.fast_auc(y,P[f"Reweighted ({L})"][m])
            row+=f"| {L}: {a:.4f}/{b:.4f} {b-a:+.4f} "
        print(row)
        out.setdefault(lab,{})[cn]={
            "n":int(m.sum()),"prev":float(y.mean()),
            **{L:{"std":fm.fast_auc(y,P[f'Standard ({L})'][m]),
                  "rw":fm.fast_auc(y,P[f'Reweighted ({L})'][m])} for L in ["12-lead","3-lead","1-lead"]}}

print(f"\n{'='*74}\n짝지은 검정, complete 라벨, 12-lead\n{'='*74}")
y_all,mask0=LAB["complete"]
for cn,cm in COH.items():
    m=mask0&cm; y=y_all[m]
    d,lo,hi,p=paired(y,P["Standard (12-lead)"][m],P["Reweighted (12-lead)"][m])
    print(f"  {cn:16s} N={m.sum():6d}  Δ={d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P={p:.4g}")
json.dump(out,open("/home/claude/snuh_rhythm_sensitivity.json","w"),indent=2)
print("\n[written] snuh_rhythm_sensitivity.json")
