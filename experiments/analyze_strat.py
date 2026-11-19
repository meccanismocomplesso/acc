"""analyze_strat.py - Riassunto di study_strat.py."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/study_strat_results.csv"
df = pd.read_csv(src)
pd.set_option("display.width", 220)
order = ["P_rand", "BASE", "CLEAN", "STRAT"]
pops = [p for p in ["gauss", "t3", "mix"] if p in df["pop"].unique()]
print("repliche:", sorted(df.groupby(["pop", "scenario", "tau"]).rep.nunique().unique()))
s = df.groupby(["pop", "scenario", "tau", "method"]).agg(
    power=("correct", "mean"), reject=("reject", "mean"), precision=("precision", "mean"),
    covered=("covered", "mean"), in_O=("in_O", "mean"), rej_O=("reject_O", "mean")).reset_index()
s.to_csv(src.replace("_results", "_summary"), index=False)
print("\nPotenza (scoperte corrette):")
print(s.pivot_table(index=["pop", "scenario", "tau"], columns="method", values="power")[order].round(2).to_string())
print("\nQuota della regione vera nello strato esterno, e rifiuti del test sullo strato esterno (STRAT):")
st = s[s.method == "STRAT"].set_index(["pop", "scenario", "tau"])[["in_O", "rej_O"]]
print(st.round(2).to_string())
eff = df[df.tau > 0]
print("\nPotenza media sulle celle con effetto, per popolazione:")
print(eff.groupby(["pop", "method"]).correct.mean().unstack().loc[pops, order].round(3).to_string())
pw = eff.pivot_table(index=["pop", "scenario", "tau", "rep"], columns="method", values="correct").astype(float)
print("\nDifferenze appaiate con BASE:")
for p, g in pw.groupby(level=0):
    for m in ("P_rand", "CLEAN", "STRAT"):
        x = g[m] - g["BASE"]
        print(f"  {p:6s} {m:7s} {x.mean():+.3f} (e.s. {x.sem():.3f})")
nul = df[df.scenario == "nullo"]
print("\nErrore di I tipo (nullo):")
print(nul.groupby(["pop", "method"]).reject.mean().unstack().loc[pops, order].round(3).to_string())
print("\nCopertura IC:")
print(df.groupby(["pop", "method"]).covered.mean().unstack().loc[pops, order].round(3).to_string())
print("\nPrecisione media di R_hat (corpo principale per CLEAN/STRAT):")
print(s.pivot_table(index=["pop", "scenario", "tau"], columns="method", values="precision")[order].round(3).to_string())
