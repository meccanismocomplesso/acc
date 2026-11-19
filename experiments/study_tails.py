"""
study_tails.py - La v0.2 su popolazioni con code pesanti.

Domande:
  1. La diagnostica delle code (population_diagnostic) reagisce? Che lambda0 produce?
  2. Il metodo adattivo v0.2 usa bene la partenza diversa da 0,5?
  3. Bilanciamento (fase 0 e trattamento/controllo), errore di I tipo e copertura degli IC
     restano validi quando le covariate hanno code pesanti?

Popolazioni (d = 5, N = 100.000, covariate indipendenti, varianza unitaria):
  gauss   normale standard (riferimento, come nei pilota)
  t5      t di Student con 5 gradi di liberta', riscalata a varianza 1
  t3      t di Student con 3 gradi di liberta', riscalata a varianza 1 (code molto pesanti)
Scenari, esito e fase 2 identici al pilota 2 (le regioni sono definite per quantili, quindi
contengono sempre il 5% / 1% della popolazione). Codice acc invariato (v0.2, parametri di default).

Semi 4.000+: nuovi, disgiunti da pilota (0-99), sviluppo (1.000-3.199) e valutazione (>= 10.000).
Uso:  python3 experiments/study_tails.py --reps 40 --out results/study_tails_results.csv
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
import pilot1
from pilot1 import N, D, N1, SCENARIOS, region_mask, outcome
from pilot2 import phase2

DISTS = ["gauss", "t5", "t3"]
METHODS = ["P_rand", "P_l1", "P_mix", "P_adapt"]
SEED0 = 4_000


def make_population(seed: int, dist: str) -> np.ndarray:
    rng = np.random.default_rng([seed, DISTS.index(dist)])
    if dist == "gauss":
        return rng.standard_normal((N, D))
    df = int(dist[1:])
    return rng.standard_t(df, (N, D)) / np.sqrt(df / (df - 2))


def one_rep(args):
    dist, scenario, tau, rep = args
    base = [DISTS.index(dist), SCENARIOS.index(scenario), int(tau * 10), rep]
    rng0 = np.random.default_rng(base)
    X = make_population(rep, dist)
    R = region_mask(X, scenario)
    parts = acc.anticluster(X, 3, rng0)
    bal0 = acc.balance_report(X, parts)["max_abs_smd"]
    L = np.flatnonzero(parts == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    diag = acc.population_diagnostic(ZL)
    rows = []
    for i, m in enumerate(METHODS):
        rng = np.random.default_rng(base + [i])
        measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
        t0 = time.time()
        shares = None
        if m == "P_adapt":
            S1, T1, Y1, _, shares = acc.adaptive_phase1(ZL, XL, measure, N1, rng, lam0=diag["lam0"],
                                                        rounds=4, rk=rk, seed=rep)
        else:
            if m == "P_rand":
                S1 = rng.choice(len(L), N1, replace=False)
            elif m == "P_l1":
                S1 = acc.select_tempered(ZL, N1, 1.0, rng, rk=rk)
            else:
                cov = acc.select_tempered(ZL, N1 // 2, 1.0, rng, rk=rk)
                rest = np.setdiff1d(np.arange(len(L)), cov)
                S1 = np.concatenate([cov, rng.choice(rest, N1 - len(cov), replace=False)])
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = measure(S1, T1)
        rec = dict(dist=dist, scenario=scenario, tau=tau, rep=rep, method=m,
                   s1_in_region=int(RL[S1].sum()),
                   tail_ratio=diag["tail_ratio"], excess_kurtosis=diag["excess_kurtosis"],
                   lam0=diag["lam0"], phase0_max_smd=bal0,
                   t1_max_smd=acc.balance_report(XL[S1], T1)["max_abs_smd"],
                   share1=shares[1] if shares else np.nan,
                   final_share_cov=shares[-1] if shares else np.nan)
        rec.update(phase2(XL, RL, S1, T1, Y1, tau, rng, rep))
        rec["seconds"] = time.time() - t0
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="results/study_tails_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--dists", default=",".join(DISTS))
    a = ap.parse_args()
    cells = [("nullo", 0.0)] + [(s, t) for s in SCENARIOS[1:] for t in (1.5, 3.0)]
    jobs = [(dd, s, t, SEED0 + r) for r in range(a.reps) for dd in a.dists.split(",") for s, t in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
