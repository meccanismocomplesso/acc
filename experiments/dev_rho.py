"""
dev_rho.py - Quanto conta la dimensione della regione R_hat (rho) nella fase 2?

Nei pilota R_hat e' il 3% del learning set con tau_hat piu' alto (~1.000 punti) e S2 sono
600 unita' estratte a caso da R_hat. Se la regione vera e' piu' piccola di R_hat (1% nelle
geometrie rare), la precisione di R_hat non puo' superare 1/3 e l'effetto medio misurato in
fase 2 si diluisce. Qui, a parita' di fase 1, si confronta la fase 2 con:
  rho = n2/N  (R_hat = le n2 unita' con tau_hat piu' alto: la regione piu' piccola possibile)
  rho = 3%    (pilota)
  rho = 5%
Il confronto e' appaiato: stessa popolazione, stesso S1, stessi esiti di fase 1.

Semi di sviluppo 1.000+ (compatta_5, coda_1); con --check semi 3.000+ e celle libere.
Uso:  python3 experiments/dev_rho.py --reps 60 --out results/dev_rho_results.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402
import pilot2
from pilot1 import N1, N2, SCENARIOS, make_population, outcome, region_mask

METHODS = ["P_rand", "P_l1", "A_v02"]
RHOS = {"n2": None, "3%": 0.03, "5%": 0.05}


def one_rep(args):
    scenario, tau, rep = args
    base = [SCENARIOS.index(scenario), int(tau * 10), rep, 77]
    rng0 = np.random.default_rng(base)
    X = make_population(rep)
    R = region_mask(X, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    lam0 = acc.population_diagnostic(ZL)["lam0"]
    rows = []
    for i, m in enumerate(METHODS):
        rng = np.random.default_rng(base + [i])
        measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
        if m == "A_v02":
            S1, T1, Y1, _, _ = acc.adaptive_phase1(ZL, XL, measure, N1, rng, lam0=lam0,
                                                   rounds=4, rk=rk, seed=rep)
        else:
            S1 = (rng.choice(len(L), N1, replace=False) if m == "P_rand"
                  else acc.select_tempered(ZL, N1, 1.0, rng, rk=rk))
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = measure(S1, T1)
        for j, (name, rho) in enumerate(RHOS.items()):
            pilot2.RHO = (N2 + 0.5) / len(L) if rho is None else rho
            rec = dict(scenario=scenario, tau=tau, rep=rep, method=m, rho=name,
                       s1_in_region=int(RL[S1].sum()))
            rec.update(pilot2.phase2(XL, RL, S1, T1, Y1, tau, np.random.default_rng(base + [i, j]), rep))
            rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="results/dev_rho_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cells", default="")
    a = ap.parse_args()
    seed0 = 3_000 if a.check else 1_000
    cells = ([(c.split(":")[0], float(c.split(":")[1])) for c in a.cells.split(",")] if a.cells
             else [(s, t) for s in ("compatta_5", "coda_1") for t in (1.5, 3.0)])
    if not a.check:
        assert all(s in ("compatta_5", "coda_1") for s, _ in cells), "sviluppo: solo compatta_5 e coda_1"
    jobs = [(s, t, seed0 + r) for r in range(a.reps) for s, t in cells]
    t0 = time.time()
    with Pool(a.procs) as p:
        chunks = p.map(one_rep, jobs, chunksize=4)
    df = pd.DataFrame([r for c in chunks for r in c])
    df.to_csv(a.out, index=False)
    print(f"{len(jobs)} job in {time.time() - t0:.0f}s -> {a.out}")
    g = df.groupby(["scenario", "tau", "method", "rho"]).agg(
        power=("reject", "mean"), precision=("precision", "mean"), coverage=("covered", "mean"))
    pd.set_option("display.width", 200)
    print(g.power.unstack("rho").round(3).to_string())
    print(g.precision.unstack("rho").round(3).to_string())
    pw = df.pivot_table(index=["scenario", "tau", "method", "rep"], columns="rho", values="reject").astype(float)
    for r in ("n2", "5%"):
        x = pw[r] - pw["3%"]
        print(f"potenza rho={r} meno rho=3%: {x.mean():+.3f} (se {x.sem():.3f}, appaiato, {len(x)} coppie)")
