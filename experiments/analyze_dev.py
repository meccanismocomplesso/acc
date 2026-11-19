"""analyze_dev.py - Riassunto del confronto tra regole di attribuzione (dev_rules.py).
Per ogni cella: potenza, unita' di S1 nella regione, quota di copertura finale, regret
rispetto all'oracolo (il migliore tra P_rand e P_l1, col senno di poi).
Criterio di scelta (H4 del piano): regret massimo sulle celle piu' basso."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/dev_rules_results.csv"
df = pd.read_csv(src)
s = df.groupby(["scenario", "tau", "method"]).agg(
    reps=("rep", "size"), power=("reject", "mean"), s1_in_region=("s1_in_region", "mean"),
    recall=("recall", "mean"), coverage=("covered", "mean"), n_cov=("n_cov", "mean"),
    share_final=("share3", "mean")).reset_index()
s["mcse"] = np.sqrt(s.power * (1 - s.power) / s.reps)
fixed = s[s.method.isin(["P_rand", "P_l1"])]
if len(fixed):
    s = s.merge(fixed.groupby(["scenario", "tau"]).power.max().rename("oracle").reset_index())
    s["regret"] = s.oracle - s.power
out = src.replace("_results", "_summary") if "_results" in src else "dev_summary.csv"
s.to_csv(out, index=False)
pd.set_option("display.width", 220)
print(s.round(3).to_string(index=False))
if "regret" in s:
    nonnull = s[s.tau > 0]
    r = nonnull.groupby("method").agg(regret_max=("regret", "max"), regret_medio=("regret", "mean"),
                                      potenza_min=("power", "min"), potenza_media=("power", "mean"))
    print("\nRegret rispetto all'oracolo (celle con effetto):")
    print(r.sort_values("regret_max").round(3).to_string())
print("->", out)
