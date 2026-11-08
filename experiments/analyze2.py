"""analyze2.py - Riassunto del pilota adattivo (results2.csv -> summary2.csv).
Aggiunge 'oracolo': per ogni scenario e tau, il migliore tra P_rand e P_l1 col senno di poi."""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results2.csv")
df["bias"] = df["est"] - df["truth"]
s = df.groupby(["scenario", "tau", "method"]).agg(
    reps=("rep", "size"), s1_in_region=("s1_in_region", "mean"),
    recall=("recall", "mean"), power=("reject", "mean"), coverage=("covered", "mean"),
    bias=("bias", "mean"), mean_share_cov=("mean_share_cov", "mean"),
    final_share_cov=("final_share_cov", "mean"), lam0=("lam0", "mean")).reset_index()
s["mcse"] = np.sqrt(s.power * (1 - s.power) / s.reps)
best = (s[s.method.isin(["P_rand", "P_l1"])].sort_values("power")
        .groupby(["scenario", "tau"]).tail(1)[["scenario", "tau", "power"]]
        .rename(columns={"power": "oracle_power"}))
s = s.merge(best, on=["scenario", "tau"])
worst = (s[s.method.isin(["P_rand", "P_l1"])].groupby(["scenario", "tau"]).power.min()
         .rename("worst_fixed").reset_index())
s = s.merge(worst, on=["scenario", "tau"])
# quota recuperata della perdita tra lambda sbagliato e lambda giusto
gap = s.oracle_power - s.worst_fixed
s["recovered"] = np.where(gap > 0.05, (s.power - s.worst_fixed) / gap, np.nan)
src = sys.argv[1] if len(sys.argv) > 1 else "results2.csv"
out = src.replace("_results", "_summary") if "_results" in src else "summary.csv"
s.to_csv(out, index=False)
print("->", out)
pd.set_option("display.width", 220)
print(s.round(3).to_string(index=False))
