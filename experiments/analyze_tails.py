"""analyze_tails.py - Riassunto di study_tails.py per popolazione, scenario, effetto e metodo."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/study_tails_results.csv"
df = pd.read_csv(src)
pd.set_option("display.width", 220)
order = ["P_rand", "P_l1", "P_mix", "P_adapt"]
dists = [d for d in ["gauss", "t5", "t3"] if d in df.dist.unique()]

s = df.groupby(["dist", "scenario", "tau", "method"]).agg(
    reps=("rep", "size"), power=("reject", "mean"), s1_in_region=("s1_in_region", "mean"),
    precision=("precision", "mean"), coverage=("covered", "mean"),
    t1_max_smd=("t1_max_smd", "mean"), final_share=("final_share_cov", "mean")).reset_index()
s.to_csv(src.replace("_results", "_summary"), index=False)

print("Diagnostica per popolazione (una riga per replica):")
first = df.drop_duplicates(["dist", "scenario", "tau", "rep"])
print(first.groupby("dist")[["tail_ratio", "excess_kurtosis", "lam0", "phase0_max_smd"]]
      .agg(["mean", "min", "max"]).round(3).loc[dists].to_string())

for col, nd in (("power", 2), ("s1_in_region", 1)):
    print(f"\n{col}:")
    t = s.pivot_table(index=["scenario", "tau"], columns=["dist", "method"], values=col)
    print(t.reindex(columns=pd.MultiIndex.from_product([dists, order])).round(nd).to_string())

print("\nErrore di I tipo (scenario nullo) e copertura IC media:")
nul = df[df.scenario == "nullo"]
e = nul.groupby(["dist", "method"]).reject.agg(["mean", "size"])
e["mcse"] = np.sqrt(e["mean"] * (1 - e["mean"]) / e["size"])
print(e.round(3).unstack("method").loc[dists].to_string())
print(df.groupby(["dist", "method"]).covered.mean().unstack().loc[dists, order].round(3).to_string())

print("\nP_adapt: quota di copertura dopo il ciclo 1 e finale (lam0 = partenza):")
ad = df[df.method == "P_adapt"]
print(ad.groupby(["dist", "scenario", "tau"])[["lam0", "share1", "final_share_cov"]].mean()
      .round(3).to_string())

print("\nPotenza media sulle celle con effetto, e minimo per cella:")
eff = s[s.tau > 0]
print(eff.groupby(["dist", "method"]).power.agg(["mean", "min"]).unstack("method")
      .loc[dists].round(3).to_string())

pw = df[df.tau > 0].pivot_table(index=["dist", "scenario", "tau", "rep"], columns="method",
                                values="reject").astype(float)
print("\nP_adapt meno P_mix (appaiato, per popolazione):")
for d, g in (pw["P_adapt"] - pw["P_mix"]).groupby(level=0):
    print(f"  {d:6s} {g.mean():+.3f} (e.s. {g.sem():.3f}, {len(g)} coppie)")
print("\nBilanciamento T/C della fase 1 (max |SMD| medio):")
print(df.groupby(["dist", "method"]).t1_max_smd.mean().unstack().loc[dists, order].round(4).to_string())
