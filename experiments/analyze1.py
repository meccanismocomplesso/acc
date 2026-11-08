"""analyze.py - Riassunto dei risultati del pilota (results.csv -> summary.csv)."""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
df["bias"] = df["est"] - df["truth"]
g = df.groupby(["scenario", "tau", "method"])
s = g.agg(reps=("rep", "size"),
          s1_in_region=("s1_in_region", "mean"),
          precision=("precision", "mean"),
          recall=("recall", "mean"),
          reject_rate=("reject", "mean"),
          coverage=("covered", "mean"),
          mean_truth=("truth", "mean"),
          mean_est=("est", "mean"),
          bias=("bias", "mean"),
          se=("se", "mean")).reset_index()
# errore standard Monte Carlo del tasso di rifiuto
s["reject_mcse"] = np.sqrt(s.reject_rate * (1 - s.reject_rate) / s.reps)
order = {m: i for i, m in enumerate(["B1", "B3", "P_rand", "P_l0", "P_l05", "P_l1"])}
s = s.sort_values(["scenario", "tau", "method"], key=lambda c: c.map(order) if c.name == "method" else c)
src = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
out = src.replace("_results", "_summary") if "_results" in src else "summary.csv"
s.to_csv(out, index=False)
print("->", out)
pd.set_option("display.width", 220)
print(s.round(3).to_string(index=False))
