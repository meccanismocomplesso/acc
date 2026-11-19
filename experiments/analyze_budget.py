"""analyze_budget.py - Riassunto di study_budget.py: potenza per quota di budget in fase 1."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/study_budget_results.csv"
df = pd.read_csv(src)
pd.set_option("display.width", 220)
order = ["P_rand", "P_l1", "P_mix", "P_adapt"]
df["found"] = df.precision >= 0.05
s = df.groupby(["scenario", "tau", "split", "method"]).agg(
    reps=("rep", "size"), power=("reject", "mean"), s1_in_region=("s1_in_region", "mean"),
    precision=("precision", "mean"), found=("found", "mean"), coverage=("covered", "mean")).reset_index()
s.to_csv(src.replace("_results", "_summary"), index=False)
print("repliche per cella:", sorted(s.reps.unique()))
for col, nd in (("power", 2), ("found", 2), ("precision", 3), ("s1_in_region", 1)):
    print(f"\n{col}:")
    t = s.pivot_table(index=["scenario", "tau"], columns=["method", "split"], values=col)
    print(t.reindex(columns=pd.MultiIndex.from_product([order, [0.2, 0.4, 0.6]])).round(nd).to_string())
eff = df[df.tau > 0]
print("\nPotenza media sulle celle con effetto:")
print(eff.groupby(["method", "split"]).reject.mean().unstack().loc[order].round(3).to_string())
pw = eff.pivot_table(index=["scenario", "tau", "rep", "method"], columns="split", values="reject").astype(float)
print("\nDifferenze appaiate rispetto al 40%:")
for m in order:
    g = pw.xs(m, level="method")
    for sp in (0.2, 0.6):
        x = g[sp] - g[0.4]
        print(f"  {m:8s} {int(sp*100)}% - 40%: {x.mean():+.3f} (e.s. {x.sem():.3f})")
best = s[s.tau > 0].groupby(["scenario", "tau", "split"]).power.max().unstack()
print("\nMiglior metodo per cella, per quota:")
print(best.round(2).to_string())
nul = df[df.scenario == "nullo"]
print("\nErrore di I tipo (nullo):")
print(nul.groupby(["method", "split"]).reject.mean().unstack().loc[order].round(3).to_string())
print("\nCopertura IC media:", df.groupby("split").covered.mean().round(3).to_dict())
print("\nPotenza se la regione e' trovata / non trovata, per quota:")
print(eff.groupby(["split", "found"]).reject.agg(["mean", "size"]).round(3).to_string())
