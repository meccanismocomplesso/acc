"""analyze_multicand.py - Riassunto di study_multicand.py."""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else "results/study_multicand_results.csv"
df = pd.read_csv(src)
pd.set_option("display.width", 220)
order = ["BASE", "BONF_2", "BONF_3", "BONF_4", "SCREEN_2", "SCREEN_3", "SCREEN_4"]
print("repliche:", sorted(df.groupby(["scenario", "tau", "n1"]).rep.nunique().unique()))
s = df.groupby(["n1", "scenario", "tau", "variant"]).agg(
    power=("correct", "mean"), reject=("reject", "mean"), false_region=("false_region", "mean"),
    found_any=("found_any", "mean"), covered=("covered", "mean")).reset_index()
s.to_csv(src.replace("_results", "_summary"), index=False)
for col in ("power", "found_any"):
    print(f"\n{col} (scoperta corretta = rifiuto in una candidata con effetto vero):" if col == "power" else f"\n{col}:")
    t = s.pivot_table(index=["n1", "scenario", "tau"], columns="variant", values=col)[order]
    print(t.round(2).to_string())
eff = df[df.tau > 0]
print("\nPotenza (scoperte corrette) media sulle celle con effetto:")
print(eff.groupby(["n1", "variant"]).correct.mean().unstack()[order].round(3).to_string())
print("\nSolo regione rara:")
rr = eff[eff.scenario == "palla_rara_1"]
print(rr.groupby(["n1", "tau", "variant"]).correct.mean().unstack()[order].round(3).to_string())
pw = eff.pivot_table(index=["n1", "scenario", "tau", "rep"], columns="variant", values="correct").astype(float)
print("\nDifferenze appaiate con BASE, per n1 (tutte le celle | solo regione rara):")
for n1, g in pw.groupby(level=0):
    for v in order[1:]:
        x = g[v] - g["BASE"]
        xr = x[x.index.get_level_values("scenario") == "palla_rara_1"]
        print(f"  n1={n1} {v:9s} {x.mean():+.3f} (e.s. {x.sem():.3f}) | {xr.mean():+.3f} (e.s. {xr.sem():.3f})")
nul = df[df.scenario == "nullo"]
print("\nErrore di I tipo (nullo, rifiuto globale):")
print(nul.groupby(["n1", "variant"]).reject.mean().unstack()[order].round(3).to_string())
print("\nRifiuti in una candidata senza effetto, celle con effetto:")
print(eff.groupby(["n1", "variant"]).false_region.mean().unstack()[order].round(3).to_string())
print("\nCopertura IC (tutte le candidate testate):")
print(df.groupby(["n1", "variant"]).covered.mean().unstack()[order].round(3).to_string())
sc = df[df.variant.str.startswith("SCREEN")]
print("\nSCREEN: quando almeno una candidata contiene la regione, quante volte viene scelta quella giusta?")
x = sc[(sc.found_any) & (sc.tau > 0)]
print(x.groupby(["n1", "variant"]).apply(lambda g: (g.picked_precision >= 0.05).mean()).unstack().round(3).to_string())
