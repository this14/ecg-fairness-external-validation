INT = {  # MIMIC internal test, N=5429
 ("12","std"):0.9849, ("12","rw"):0.9757,
 ("3","std"):0.9645,  ("3","rw"):0.9583,
 ("1","std"):0.9188,  ("1","rw"):0.9321,
}
ECHO_QRS = {  # EchoNext, QRS>=120, N=4626, prev 12.26%
 ("12","std"):0.9392, ("12","rw"):0.8203,
 ("3","std"):0.9385,  ("3","rw"):0.9240,
 ("1","std"):0.8866,  ("1","rw"):0.8787,
}
ECHO_LVEF = {
 ("12","std"):0.6692, ("12","rw"):0.6073,
 ("3","std"):0.6810,  ("3","rw"):0.6648,
 ("1","std"):0.6955,  ("1","rw"):0.6223,
}
SNUH = {("12","std"):0.8061, ("12","rw"):0.7408,
        ("3","std"):0.8373,  ("3","rw"):0.7387}

print("=== reweighting effect on AUROC (rw - std) ===")
for nm, D in [("MIMIC internal",INT),("EchoNext QRS>=120",ECHO_QRS),
              ("EchoNext LVEF",ECHO_LVEF),("SNUH BBB",SNUH)]:
    row=[]
    for L in ["12","3","1"]:
        if (L,"std") in D:
            row.append(f"{L}-lead {D[(L,'rw')]-D[(L,'std')]:+.4f}")
    print(f"  {nm:20s} " + "   ".join(row))

print("\n=== internal -> external amplification of the reweighting cost ===")
for nm,D in [("EchoNext QRS>=120",ECHO_QRS),("SNUH BBB",SNUH)]:
    for L in ["12","3"]:
        if (L,"std") in D:
            i=abs(INT[(L,"rw")]-INT[(L,"std")]); e=abs(D[(L,"rw")]-D[(L,"std")])
            print(f"  {nm:20s} {L}-lead: internal {i:.4f}  external {e:.4f}  x{e/i:.1f}")

print("\n=== EchoNext QRS>=120, race AUROC (well-powered groups) ===")
race_std={"black":0.9240,"hispanic":0.9491,"other":0.9551,"white":0.9377}
race_rw ={"black":0.7778,"hispanic":0.8375,"other":0.8043,"white":0.8262}
for k in race_std:
    print(f"  {k:10s} {race_std[k]:.4f} -> {race_rw[k]:.4f}   {race_rw[k]-race_std[k]:+.4f}")
ss=max(race_std.values())-min(race_std.values()); sr=max(race_rw.values())-min(race_rw.values())
print(f"  spread: {ss:.4f} -> {sr:.4f}  ({sr/ss:.2f}x)")
print(f"  worst-served: {min(race_std,key=race_std.get)} -> {min(race_rw,key=race_rw.get)}")
print(f"  Black harmed vs White: {abs(race_rw['black']-race_std['black'])/abs(race_rw['white']-race_std['white']):.2f}x")
print(f"  White - Black:  standard {race_std['white']-race_std['black']:+.4f}   reweighted {race_rw['white']-race_rw['black']:+.4f}")

print("\n=== lead-count ordering (Standard) ===")
for nm,D in [("MIMIC internal",INT),("EchoNext QRS>=120",ECHO_QRS),
             ("EchoNext LVEF",ECHO_LVEF),("SNUH BBB",SNUH)]:
    o=sorted([L for L in ["12","3","1"] if (L,"std") in D], key=lambda L:-D[(L,"std")])
    print(f"  {nm:20s} " + " > ".join(f"{L}({D[(L,'std')]:.4f})" for L in o))
