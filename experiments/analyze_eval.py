"""analyze_eval.py - Ipotesi H1-H4 del piano (sez. 1) sui risultati di eval_central.py."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

src = sys.argv[1] if len(sys.argv) > 1 else "results/eval_central_results.csv"
df = pd.read_csv(src)
pd.set_option("display.width", 250)
ORDER = ["B1", "B3", "B4", "P_rand", "P_l1", "P_mix", "P_adapt_v02", "ACC_v03", "B5", "P_rand_60"]
P_METHODS = ["P_rand", "P_l1", "P_mix", "P_adapt_v02", "ACC_v03", "P_rand_60"]
full = df.groupby(["cell", "rep"]).method.nunique()
df = df.set_index(["cell", "rep"]).loc[full[full == len(ORDER)].index].reset_index()
cells = df.groupby(["geom", "tau"]).rep.nunique()
print("repliche complete per cella:\n", cells.to_string())


def cp(k, n, a=0.05):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


out = []
print("\n=== H1 Validita': scenario nullo, limite superiore IC 95% <= 7% ===")
nul = df[df.geom == "nullo"]
for m in ORDER:
    g = nul[nul.method == m]
    k, n = int(g.reject.sum()), len(g)
    lo, hi = cp(k, n)
    ok = hi <= 0.07
    print(f"  {m:12s} {k:4d}/{n} = {k / n:.4f}  IC95 [{lo:.4f}, {hi:.4f}]  {'OK' if ok else 'NO'}"
          + ("" if m in P_METHODS else "  (riferimento)"))
    out.append(dict(hyp="H1", method=m, value=k / n, hi=hi, pass_=ok))

eff = df[df.geom != "nullo"]
s = eff.groupby(["geom", "tau", "method"]).agg(n=("rep", "size"), power=("correct", "mean"),
                                                reject=("reject", "mean"), covered=("covered", "mean"),
                                                precision=("precision", "mean"), recall=("recall", "mean"),
                                                s1=("s1_in_region", "mean"), bias=("est", "mean"),
                                                truth=("truth", "mean"), sec=("seconds", "mean")).reset_index()
s["mcse"] = np.sqrt(s.power * (1 - s.power) / s.n)
s["bias"] = s.bias - s.truth
s.to_csv(src.replace("_results", "_summary"), index=False)
pw = s.pivot_table(index=["geom", "tau"], columns="method", values="power")[ORDER]
print("\nPotenza (scoperte corrette) per cella:")
print(pw.round(3).to_string())
print("\nPotenza media sulle 12 celle con effetto:")
print(pw.mean().round(3).to_string())

print("\n=== H2 Copertura: tra 93% e 97% in ogni cella (nullo incluso) ===")
cov = df.groupby(["geom", "tau", "method"]).covered.mean().unstack()[ORDER]
print(cov.round(3).to_string())
for m in P_METHODS:
    c = cov[m]
    bad = c[(c < 0.93) | (c > 0.97)]
    print(f"  {m:12s} celle fuori intervallo: {len(bad)}/{len(c)}  min {c.min():.3f}  max {c.max():.3f}"
          + (f"  -> {', '.join(f'{g}/{t}: {v:.3f}' for (g, t), v in bad.items())}" if len(bad) else ""))
    out.append(dict(hyp="H2", method=m, value=len(bad), pass_=len(bad) == 0))

print("\n=== H3 Focalizzazione: potenza > B1 in >= 80% delle celle con effetto ===")
for m in ("P_adapt_v02", "ACC_v03", "P_rand", "P_rand_60"):
    wins = (pw[m] > pw["B1"]).sum()
    print(f"  {m:12s} {wins}/{len(pw)} celle ({wins / len(pw):.0%})  {'OK' if wins / len(pw) >= 0.8 else 'NO'}")
    out.append(dict(hyp="H3", method=m, value=wins / len(pw), pass_=wins / len(pw) >= 0.8))
print("  celle dove B1 >= metodo:", {m: list(pw.index[pw[m] <= pw["B1"]]) for m in ("P_adapt_v02", "ACC_v03")})

print("\n=== H4 Adattivita': regret massimo (oracolo = max(P_rand, P_l1) per cella) ===")
oracle = pw[["P_rand", "P_l1"]].max(axis=1)
reg = pw.sub(oracle, axis=0).mul(-1)
r = pd.DataFrame({"regret_max": reg.max(), "regret_medio": reg.mean(), "cella_peggiore":
                  reg.idxmax().map(lambda x: f"{x[0]}/{x[1]}")}).loc[P_METHODS]
print(r.round(3).to_string())
for m in ("P_adapt_v02", "ACC_v03"):
    ok = all(r.loc[m, "regret_max"] < r.loc[o, "regret_max"] for o in ("P_rand", "P_l1", "P_mix"))
    print(f"  {m:12s} regret massimo {r.loc[m, 'regret_max']:.3f} < P_rand, P_l1, P_mix? {'OK' if ok else 'NO'}")
    out.append(dict(hyp="H4", method=m, value=r.loc[m, "regret_max"], pass_=ok))

print("\n=== Confronti appaiati (stessa popolazione), celle con effetto ===")
pp = eff.pivot_table(index=["geom", "tau", "rep"], columns="method", values="correct").astype(float)
for a, b in (("ACC_v03", "P_adapt_v02"), ("ACC_v03", "P_rand_60"), ("ACC_v03", "P_rand"),
             ("ACC_v03", "B1"), ("ACC_v03", "B4"), ("P_adapt_v02", "P_rand"), ("P_rand_60", "P_rand")):
    x = pp[a] - pp[b]
    cl = x.groupby(level=[0, 1]).mean()
    print(f"  {a:12s} - {b:12s} {x.mean():+.4f} (e.s. {x.sem():.4f}); celle a favore {int((cl > 0).sum())}/{len(cl)}")

print("\nPrecisione media di R_hat (localizzazione):")
print(s.pivot_table(index=["geom", "tau"], columns="method", values="precision")[P_METHODS + ["B4"]].round(3).to_string())
print("\nUnita' di S1 nella regione vera:")
print(s.pivot_table(index=["geom", "tau"], columns="method", values="s1")[P_METHODS].round(1).to_string())
print("\nDistorsione media (stima - valore vero del gruppo testato), celle con effetto:")
print(s.groupby("method").bias.mean()[ORDER].round(3).to_string())
print("\nCosto per scoperta (B / potenza media):", (1000 / pw.mean()).round(0)[ORDER].to_dict())
print("Tempo medio per metodo (s):", df.groupby("method").seconds.mean()[ORDER].round(2).to_dict())
print("ACC_v03 stratificato in", df[df.method == "ACC_v03"].stratified.astype(float).mean(), "dei casi")
pd.DataFrame(out).to_csv(src.replace("_results", "_hypotheses"), index=False)
