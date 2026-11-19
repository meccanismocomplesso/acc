"""analyze_seqlen.py - Riassunto di study_seqlen.py."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments")
from study_seqlen import apply_rules  # noqa: E402

src = sys.argv[1] if len(sys.argv) > 1 else "results/study_seqlen_results.csv"
raw = pd.read_csv(src)
full = raw.groupby(["scenario", "tau", "rep"]).n1.nunique()
raw = raw.set_index(["scenario", "tau", "rep"]).loc[full[full == 6].index].reset_index()
df = apply_rules(raw)
df.to_csv(src.replace("_results", "_rules"), index=False)
pd.set_option("display.width", 220)
rules = [f"FIX_{n}" for n in (300, 400, 500, 600, 700, 800)] + ["SIG_2.0", "SIG_2.5", "SIG_3.0"]
print("repliche:", df.groupby(["scenario", "tau"]).rep.nunique().to_dict())
s = df.groupby(["scenario", "tau", "rule"]).agg(power=("reject", "mean"), stop=("stop", "mean"),
                                                  precision=("precision", "mean"))
for col, nd in (("power", 2), ("stop", 0)):
    print(f"\n{col}:")
    print(s[col].unstack("rule")[rules].round(nd).to_string())
eff = df[df.tau > 0]
print("\nPotenza media sulle celle con effetto:")
print(eff.groupby("rule").reject.mean()[rules].round(3).to_string())
pw = eff.pivot_table(index=["scenario", "tau", "rep"], columns="rule", values="reject").astype(float)
ref = pw[["FIX_300", "FIX_400", "FIX_500", "FIX_600", "FIX_700", "FIX_800"]].mean().idxmax()
print(f"\nMiglior fissa: {ref}. Differenze appaiate:")
for r in rules:
    x = pw[r] - pw[ref]
    print(f"  {r:8s} {x.mean():+.3f} (e.s. {x.sem():.3f})")
print("\nErrore di I tipo (nullo) e punto di arresto medio:")
nul = df[df.scenario == "nullo"]
print(nul.groupby("rule").agg(reject=("reject", "mean"), stop=("stop", "mean")).loc[rules].round(3).to_string())
print("\nCopertura IC media:", df.groupby("rule").covered.mean()[rules].round(3).to_dict())
print("\nStatistica t media per prefisso (separazione segnale/rumore):")
print(raw.groupby(["scenario", "tau", "n1"]).tstat.mean().unstack().round(2).to_string())
r = raw.copy(); r["nonnull"] = r.tau > 0
print("\nQuota di prefissi con t >= 2,5 (dal ciclo 3):")
print(r.groupby(["scenario", "tau"]).tstat.apply(lambda v: (v >= 2.5).mean()).round(3).to_string())
