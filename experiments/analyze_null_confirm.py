"""analyze_null_confirm.py - Criteri preregistrati C1 e C2 (vedi eval_null_confirm.py)."""
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest

src = sys.argv[1] if len(sys.argv) > 1 else "results/eval_null_confirm_results.csv"
df = pd.read_csv(src)
cnt = df.groupby("rep").method.nunique()
df = df[df.rep.isin(cnt[cnt == cnt.max()].index)]
print("repliche complete:", df.rep.nunique())
for m in ["ACC_v03", "B5", "B1", "P_rand"]:
    g = df[df.method == m]
    k, n = int(g.reject.sum()), len(g)
    hi = beta.ppf(0.975, k + 1, n - k)
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    p = binomtest(k, n, 0.05).pvalue
    print(f"{m:8s} {k}/{n} = {k / n:.4f}  IC95 [{lo:.4f}, {hi:.4f}]  C1 {'superato' if hi <= 0.07 else 'NON superato'}"
          f"  C2 p = {p:.3f} {'(compatibile con 5%)' if p >= 0.05 else '(diverso da 5%)'}"
          f"  z medio {g.z.mean():+.3f}, sd {g.z.std():.3f}")
a = df.pivot_table(index="rep", columns="method", values="reject").astype(float)
x = a["ACC_v03"] - a["B5"]
print(f"\nACC_v03 - B5 (appaiato per popolazione): {x.mean():+.4f} (e.s. {x.sem():.4f})")
both = pd.concat([pd.read_csv("results/eval_central_results.csv").query("geom == 'nullo'")[["method", "reject"]]
                  .assign(src="valutazione"), df[["method", "reject"]].assign(src="conferma")])
print("\nValutazione (1.000) + conferma (5.000), solo come riepilogo:")
print(both[both.method.isin(["ACC_v03", "B5", "B1", "P_rand"])].groupby("method").reject.agg(["sum", "size", "mean"]).round(4))
