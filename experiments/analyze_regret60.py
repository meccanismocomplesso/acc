"""analyze_regret60.py - Analisi preregistrata del regret a parita' di budget (vedi eval_regret60.py)."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/eval_regret60_results.csv"
df = pd.read_csv(src).drop_duplicates(["cell", "rep", "method"])
M = ["P_rand_60", "P_l1_60", "P_mix_60", "ACC_v03"]
FIXED = ["P_rand_60", "P_l1_60", "P_mix_60"]
cnt = df.groupby(["cell", "rep"]).method.nunique()
df = df.set_index(["cell", "rep"]).loc[cnt[cnt == len(M)].index].reset_index()
A = df.pivot_table(index=["cell", "rep"], columns="method", values="correct").astype(float)[M]
cells = sorted(A.index.get_level_values(0).unique())
labels = df.drop_duplicates("cell").set_index("cell")[["geom", "tau"]]
arrs = {c: A.loc[c].to_numpy() for c in cells}          # repliche x metodi
print("repliche per cella:", {c: len(v) for c, v in arrs.items()})
print("stratificazioni di ACC_v03:", int(df[df.method == "ACC_v03"].stratified.sum()))


def stats(power):                       # power: celle x metodi
    orc = power[:, :3].max(1, keepdims=True)
    orc_all = power.max(1, keepdims=True)
    reg = orc - power
    reg_all = orc_all - power
    return reg.max(0), reg.mean(0), reg_all.max(0), reg


P = np.array([arrs[c].mean(0) for c in cells])
rmax, rmean, rmax_all, reg = stats(P)

rng = np.random.default_rng(2026)
NB = 5000
bmax, bmean, bmax_all = np.empty((NB, 4)), np.empty((NB, 4)), np.empty((NB, 4))
for b in range(NB):
    Pb = np.array([arrs[c][rng.integers(0, len(arrs[c]), len(arrs[c]))].mean(0) for c in cells])
    bmax[b], bmean[b], bmax_all[b], _ = stats(Pb)


def ci(x):
    return np.percentile(x, [2.5, 97.5])


pd.set_option("display.width", 220)
print("\nPotenza per cella:")
tab = pd.DataFrame(P, columns=M, index=pd.MultiIndex.from_frame(labels.loc[cells]))
print(tab.round(3).to_string())
print("\nRegret per cella (oracolo = migliore tra P_rand_60, P_l1_60, P_mix_60):")
print(pd.DataFrame(reg, columns=M, index=tab.index).round(3).to_string())

print("\n=== Analisi principale: regret massimo, IC 95% bootstrap ===")
for i, m in enumerate(M):
    lo, hi = ci(bmax[:, i])
    worst = tab.index[reg[:, i].argmax()]
    print(f"  {m:10s} {rmax[i]:.3f}  [{lo:.3f}, {hi:.3f}]   cella peggiore {worst[0]}/{worst[1]}")
print("\nDifferenze di regret massimo, ACC_v03 meno alternativa (IC 95% bootstrap appaiato):")
ok = True
for i, m in enumerate(FIXED):
    d = bmax[:, 3] - bmax[:, i]
    lo, hi = ci(d)
    ok &= hi < 0
    print(f"  ACC_v03 - {m:10s} {rmax[3] - rmax[i]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
          f"P(diff >= 0) = {(d >= 0).mean():.3f}  {'ACC minore' if hi < 0 else 'non conclusivo' if lo < 0 else 'ACC maggiore'}")
print(f"\nAffermazione principale (tutti e tre i limiti superiori < 0): {'SOSTENUTA' if ok else 'NON sostenuta'}")

print("\n=== Secondarie ===")
print("Regret medio, IC 95%:")
for i, m in enumerate(M):
    lo, hi = ci(bmean[:, i])
    print(f"  {m:10s} {rmean[i]:.3f}  [{lo:.3f}, {hi:.3f}]")
print("Regret massimo con oracolo che include ACC_v03:")
for i, m in enumerate(M):
    lo, hi = ci(bmax_all[:, i])
    print(f"  {m:10s} {rmax_all[i]:.3f}  [{lo:.3f}, {hi:.3f}]")
print("Potenza media sulle 12 celle:", dict(zip(M, P.mean(0).round(3))))

# oracolo scelto su meta' delle repliche, regret misurato sull'altra meta' (ripetuto 200 volte)
cf = []
for s in range(200):
    g = np.random.default_rng(s)
    regs = []
    for c in cells:
        x = arrs[c]
        idx = g.permutation(len(x))
        a, b2 = x[idx[: len(x) // 2]], x[idx[len(x) // 2:]]
        best = a[:, :3].mean(0).argmax()
        pb = b2.mean(0)
        regs.append(pb[best] - pb)
    cf.append(np.max(regs, axis=0))
cf = np.array(cf)
print("Regret massimo con oracolo scelto su meta' indipendente delle repliche (media su 200 divisioni):",
      dict(zip(M, cf.mean(0).round(3))))
