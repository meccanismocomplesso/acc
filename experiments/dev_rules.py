"""
dev_rules.py - Sviluppo della regola di attribuzione v0.3 (sez. 6 del piano di simulazione).

Confronta, a parita' di tutto il resto, le regole di `acc.adaptive_phase1`:
  A_v02       regola originale
  A_density   attribuzione normalizzata per densita'      (candidato 2)
  A_yield     rendimento per braccio                      (candidato 1)
  A_thompson  Thompson sampling sulle unita' nella regione (candidato 3)
  F_*         variante fuori piano: una quota di ogni ciclo estratta dalla regione candidata
con i riferimenti P_rand e P_l1 (l'oracolo e' il migliore dei due in ogni cella).

Come da piano: solo semi di sviluppo (1.000-1.999) e geometrie compatta_5 e coda_1.
Con --check si usano invece semi 3.000-3.999 e geometrie libere (controllo fuori sviluppo).
Ogni metodo ha un generatore proprio: aggiungere o togliere metodi non cambia gli altri.

Uso:  python3 experiments/dev_rules.py --reps 80 --out results/dev_rules_results.csv
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
from pilot1 import N1, SCENARIOS, make_population, outcome, region_mask
from pilot2 import phase2

RULES = ["v02", "density", "yield", "thompson"]
FOCUS = ["F_v02_50", "F_none_50", "F_v02_25"]      # F_<regola>_<quota focus in %>
METHODS = ["P_rand", "P_l1"] + [f"A_{r}" for r in RULES] + FOCUS


def one_rep(args):
    scenario, tau, rep, methods = args
    base = [SCENARIOS.index(scenario), int(tau * 10), rep]
    rng0 = np.random.default_rng(base)
    X = make_population(rep)
    R = region_mask(X, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    lam0 = acc.population_diagnostic(ZL)["lam0"]
    rows = []
    for m in methods:
        rng = np.random.default_rng(base + [METHODS.index(m) + 1])
        measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
        t0 = time.time()
        shares, A1 = [np.nan] * 4, None
        if m == "P_rand":
            S1 = rng.choice(len(L), N1, replace=False)
        elif m == "P_l1":
            S1 = acc.select_tempered(ZL, N1, 1.0, rng, rk=rk)
        if m.startswith(("A_", "F_")):
            rule = m.split("_")[1]
            focus = int(m.split("_")[2]) / 100 if m.startswith("F_") else 0.0
            S1, T1, Y1, A1, shares = acc.adaptive_phase1(
                ZL, XL, measure, N1, rng, lam0=lam0, rounds=4, rk=rk, seed=rep,
                rule=rule, focus=focus)
        else:
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = measure(S1, T1)
        rec = dict(scenario=scenario, tau=tau, rep=rep, method=m, s1_in_region=int(RL[S1].sum()),
                   share1=shares[1], share2=shares[2], share3=shares[3],
                   n_cov=int((A1 == 1).sum()) if A1 is not None else np.nan,
                   focus_in_region=int(RL[S1[A1 == 2]].sum()) if A1 is not None else np.nan)
        rec.update(phase2(XL, RL, S1, T1, Y1, tau, rng, rep))
        rec["seconds"] = time.time() - t0
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--out", default="results/dev_rules_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="controllo fuori sviluppo: semi 3.000+")
    ap.add_argument("--cells", default="", help="es. 'palla_rara_1:3.0,nullo:0.0'")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    seed0 = 3_000 if a.check else 1_000
    if a.cells:
        cells = [(c.split(":")[0], float(c.split(":")[1])) for c in a.cells.split(",")]
    else:
        cells = [(s, t) for s in ("compatta_5", "coda_1") for t in (1.5, 3.0)]
    if not a.check:
        assert all(s in ("compatta_5", "coda_1") for s, _ in cells), "sviluppo: solo compatta_5 e coda_1"
    methods = a.methods.split(",")
    jobs = [(s, t, seed0 + r, methods) for r in range(a.reps) for s, t in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 40 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
