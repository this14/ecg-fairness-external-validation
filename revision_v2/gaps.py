import json, os
B='/home/this14/ev_out'
F=['echonext_results_qrs_duration_ge120.json','snuh_results_complete.json','echonext_results_lvef_lte_45_flag.json']
def ms(v):
    m=sum(v)/len(v)
    return m,(sum((a-m)**2 for a in v)/(len(v)-1))**.5
def val(d,c,g,k):
    try:
        x=d['models'][c]['subgroups'][g][k]
    except KeyError:
        return None
    return None if x is None or x!=x else float(x)
for nm in F:
    if nm.startswith('echonext'):
        P=[('race_ethnicity=black','race_ethnicity=white'),('sex=female','sex=male'),('age_at_ecg=>65','age_at_ecg=<50')]
    else:
        P=[('sex=F','sex=M'),('age=>65','age=<50')]
    D=[json.load(open(f'{B}/seed{s}/{nm}')) for s in range(42,52) if os.path.exists(f'{B}/seed{s}/{nm}')]
    print('=====',nm,len(D),'seeds; gap=A-B; d|gap|=|RW|-|Std| (neg=narrowed)')
    for A,Bg in P:
        s0=D[0]['models']['Standard (12-lead)']['subgroups']
        print('###',A,'-',Bg,'n_pos',s0.get(A,{}).get('n_pos'),s0.get(Bg,{}).get('n_pos'))
        for c in ['12-lead','3-lead','1-lead']:
            for k in ['auroc','sens','spec']:
                gs=[];gr=[];dd=[]
                for d in D:
                    v=[val(d,f'{t} ({c})',g,k) for t in ('Standard','Reweighted') for g in (A,Bg)]
                    if None in v: continue
                    g1=v[0]-v[1]; g2=v[2]-v[3]
                    gs.append(g1);gr.append(g2);dd.append(abs(g2)-abs(g1))
                if len(dd)<2: continue
                a,b=ms(gs);e,f=ms(gr);g,h=ms(dd)
                print(f'{c:8s}{k:6s}gapStd {a:+.3f}+/-{b:.3f} gapRW {e:+.3f}+/-{f:.3f} d|gap| {g:+.3f}+/-{h:.3f} narrowed {sum(x<0 for x in dd)}/{len(dd)}')
