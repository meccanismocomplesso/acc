"""analyze_competitors.py - Riassunto di study_competitors.py."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/study_competitors_results.csv"
df = pd.read_csv(src)
pd.set_option("display.width", 240)
order = ["ATE", "GATES", "VT", "ENRICH", "ML2S", "ACC40", "ACC60", "ACC_COND"]
pops = [p for p in ["gauss", "t3", "mix"] if p in df["pop"].unique()]
print("repliche:", sorted(df.groupby(["pop", "scenario", "tau"]).rep.nunique().unique()))
s = df.groupby(["pop", "scenario", "tau", "method"]).agg(
    power=("correct", "mean"), precision=("precision", "mean"), covered=("covered", "mean"),
    bias=("bias", "mean"), seconds=("seconds", "mean")).reset_index()
s.to_csv(src.replace("_results", "_summary"), index=False)
print("\nPotenza (scoperte corrette):")
print(s.pivot_table(index=["pop", "scenario", "tau"], columns="method", values="power")[order].round(2).to_string())
eff = df[df.tau > 0]
print("\nPotenza media sulle celle con effetto:")
print(eff.groupby(["pop", "method"]).correct.mean().unstack().loc[pops, order].round(3).to_string())
print("tutte:", eff.groupby("method").correct.mean()[order].round(3).to_dict())
pw = eff.pivot_table(index=["pop", "scenario", "tau", "rep"], columns="method", values="correct").astype(float)
comp = ["ATE", "GATES", "VT", "ENRICH", "ML2S"]
pw["BEST_COMP"] = pw[comp].mean(axis=0).idxmax()  # placeholder
print("\nDifferenze appaiate ACC60 e ACC_COND meno ogni concorrente (tutte le celle con effetto):")
for ref in ("ACC60", "ACC_COND"):
    for c in comp + ["ACC40"]:
        x = pw[ref] - pw[c]
        print(f"  {ref:8s} - {c:6s} {x.mean():+.3f} (e.s. {x.sem():.3f})")
print("\nPer popolazione, ACC_COND meno il miglior concorrente di quella popolazione:")
for p, g in pw.groupby(level=0):
    best = g[comp].mean().idxmax()
    x = g["ACC_COND"] - g[best]
    print(f"  {p:6s} miglior concorrente {best} ({g[best].mean():.3f}); ACC_COND {g['ACC_COND'].mean():.3f}; diff {x.mean():+.3f} (e.s. {x.sem():.3f})")
print("\nPrecisione media del gruppo testato (localizzazione), celle con effetto:")
print(s[s.tau > 0].pivot_table(index=["pop", "scenario", "tau"], columns="method", values="precision")[order].round(3).to_string())
nul = df[df.scenario == "nullo"]
print("\nErrore di I tipo (nullo):")
print(nul.groupby(["pop", "method"]).reject.mean().unstack().loc[pops, order].round(3).to_string())
print("\nCopertura IC:")
print(df.groupby(["pop", "method"]).covered.mean().unstack().loc[pops, order].round(3).to_string())
print("\nDistorsione media della stima nel gruppo testato, quando rifiuta (celle con effetto):")
print(eff[eff.reject].groupby("method").bias.mean()[order].round(3).to_string())
print("\nTempo medio per esecuzione (s):", df.groupby("method").seconds.mean()[order].round(2).to_dict())
