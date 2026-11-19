"""
dev_explore.py - Due vie esplorative per la fase 1 (fuori dal piano congelato):
  P_proj   copertura per proiezioni (`acc.select_projected`): rarita' per singole covariate e coppie
  Z_zoom   ricerca a zoom per celle (`acc.zoom_phase1`): usa gli esiti aggregati per gruppo
a confronto con P_rand, P_l1 e P_mix. Fase 2 identica per tutti (pilot2.phase2).

Semi di sviluppo 1.000+. Include palla_rara_1, che il piano riserva alla valutazione:
i risultati su quella geometria sono solo esplorativi. Semi di valutazione non toccati.

Uso:  python3 experiments/dev_explore.py --reps 50 --out results/dev_explore_results.csv
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

METHODS = ["P_rand", "P_l1", "P_mix", "P_proj", "Z_zoom"]


def one_rep(args):
    scenario, tau, rep = args
    base = [SCENARIOS.index(scenario), int(tau * 10), rep, 99]
    rng0 = np.random.default_rng(base)
    X = make_population(rep)
    R = region_mask(X, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    rows = []
    for i, m in enumerate(METHODS):
        rng = np.random.default_rng(base + [i])
        measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
        if m == "Z_zoom":
            S1, T1, Y1, _ = acc.zoom_phase1(ZL, XL, measure, N1, rng, seed=rep)
        else:
            if m == "P_rand":
                S1 = rng.choice(len(L), N1, replace=False)
            elif m == "P_l1":
                S1 = acc.select_tempered(ZL, N1, 1.0, rng, rk=rk)
            elif m == "P_mix":
                cov = acc.select_tempered(ZL, N1 // 2, 1.0, rng, rk=rk)
                rest = np.setdiff1d(np.arange(len(L)), cov)
                S1 = np.concatenate([cov, rng.choice(rest, N1 - len(cov), replace=False)])
            else:
                S1 = acc.select_projected(ZL, N1, rng, 2)
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = measure(S1, T1)
        rec = dict(scenario=scenario, tau=tau, rep=rep, method=m, s1_in_region=int(RL[S1].sum()))
        rec.update(phase2(XL, RL, S1, T1, Y1, tau, rng, rep))
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="results/dev_explore_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cells = [(s, t) for s in ("compatta_5", "coda_1", "palla_rara_1") for t in (1.5, 3.0)]
    jobs = [(s, t, 1_000 + r) for r in range(a.reps) for s, t in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
