import numpy as np, pandas as pd, json
import fastmetrics as fm

U="/root/.claude/uploads/4eea52cb-5125-5e39-a706-93f7fe1368e1"
d=pd.read_csv(f"{U}/1b509c09-snuh_slim.csv",low_memory=False); n=len(d)
z=np.load("/home/claude/snuh_probs.npz",allow_pickle=True)
names=z["model_names"].tolist()
assert np.array_equal(z["new_id"], d["new_id"].astype("int64").values), "new_id 정렬 불일치"
print("정렬 확인: new_id 일치 OK,", n, "행")
P={nm: z[f"p{i}"].astype(float) for i,nm in enumerate(names)}

STRICT=["RBBB","LBBB","BFB(RBBB+LAFB)","BFB(RBBB+LPFB)","BFB"]
bbb=d["bbb"].astype(str).str.strip().isin(STRICT).values
qd=pd.to_numeric(d["qrsd"],errors="coerce").values
hq=~np.isnan(qd); wide=np.where(hq,qd>=120,False)
age=pd.to_numeric(d["age"],errors="coerce")
agrp=pd.cut(age,[0,50,65,200],labels=["<50","50-65",">65"]).astype(str).values
sex=d["sex"].astype(str).values

LAB={"strict":(bbb.astype(int),np.ones(n,bool)),
     "qrsd120":(wide.astype(int),hq),
     "complete":((bbb&wide).astype(int),hq)}

out={}
for ln,(y_all,mask) in LAB.items():
    print("\n"+"="*84); print(f"LABEL = {ln}   N={mask.sum()}  prevalence={y_all[mask].mean()*100:.2f}%"); print("="*84)
    for nm in names:
        y,p=y_all[mask],P[nm][mask]
        print(f"\n  {nm}")
        thr=fm.youden_threshold(y,p)
        r={"at_0.5":fm.summarize(y,p,0.5,"overall @0.5"),
           "at_youden":fm.summarize(y,p,thr,f"overall @J={thr:.4f}"),
           "subgroups":{}}
        fm.subgroups(y,p,sex[mask],"sex",0.5,r["subgroups"])
        fm.subgroups(y,p,agrp[mask],"age",0.5,r["subgroups"])
        out.setdefault(ln,{})[nm]=r
json.dump(out,open("/home/claude/snuh_all_labels.json","w"),indent=2)
print("\n[written] snuh_all_labels.json")
