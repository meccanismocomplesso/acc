"""
study_budget.py - Ripartizione del budget tra fase 1 ed fase 2 (sez. del piano: n1/B = 20%, 40%, 60%).

Budget totale B = 1.000 invariato. Per ogni quota si rifanno fase 1 e fase 2 con codice v0.2
(parametri di default): P_rand, P_l1, P_mix, P_adapt. Popolazione gaussiana, scenari e esito del pilota 2.
R_hat resta il 3% del learning set con tau_hat piu' alto (~1.000 punti >= n2 in tutti i casi).

Domanda: dare piu' unita' alla fase 1 (capire la popolazione) rende piu' di darle alla fase 2
(confermare l'effetto)? Ogni replica usa la stessa popolazione per le tre quote.

Semi 5.000+ (nuovi, non di valutazione).
Uso:  python3 experiments/study_budget.py --reps 40 --out results/study_budget_results.csv
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
from pilot1 import B, SCENARIOS, make_population, outcome, region_mask

SPLITS = [0.2, 0.4, 0.6]
METHODS = ["P_rand", "P_l1", "P_mix", "P_adapt"]
SEED0 = 5_000


def one_rep(args):
    scenario, tau, rep = args
    X = make_population(rep)
    R = region_mask(X, scenario)
    base0 = [SCENARIOS.index(scenario), int(tau * 10), rep]
    rng0 = np.random.default_rng(base0)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    lam0 = acc.population_diagnostic(ZL)["lam0"]
    rows = []
    for s in SPLITS:
        n1 = int(round(s * B))
        pilot2.N2 = B - n1                         # phase2 legge N2 dal modulo pilot2
        for i, m in enumerate(METHODS):
            rng = np.random.default_rng(base0 + [int(s * 100), i])
            measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
            if m == "P_adapt":
                S1, T1, Y1, _, _ = acc.adaptive_phase1(ZL, XL, measure, n1, rng, lam0=lam0,
                                                       rounds=4, rk=rk, seed=rep)
            else:
                if m == "P_rand":
                    S1 = rng.choice(len(L), n1, replace=False)
                elif m == "P_l1":
                    S1 = acc.select_tempered(ZL, n1, 1.0, rng, rk=rk)
                else:
                    cov = acc.select_tempered(ZL, n1 // 2, 1.0, rng, rk=rk)
                    rest = np.setdiff1d(np.arange(len(L)), cov)
                    S1 = np.concatenate([cov, rng.choice(rest, n1 - len(cov), replace=False)])
                T1 = acc.anticluster(XL[S1], 2, rng)
                Y1 = measure(S1, T1)
            rec = dict(scenario=scenario, tau=tau, rep=rep, split=s, n1=n1, method=m,
                       s1_in_region=int(RL[S1].sum()))
            rec.update(pilot2.phase2(XL, RL, S1, T1, Y1, tau, rng, rep))
            rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="results/study_budget_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cells = [("nullo", 0.0)] + [(s, t) for s in SCENARIOS[1:] for t in (1.5, 3.0)]
    jobs = [(s, t, SEED0 + r) for r in range(a.reps) for s, t in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
