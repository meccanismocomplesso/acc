"""
pilot2.py - Pilota della fase 1 adattiva. Stessa impostazione di pilot.py
(d = 5, N = 100.000, B = 1.000, n1 = 400, n2 = 600, R_hat = top 3% di tau_hat,
inferenza solo su S2), metodi:

  P_rand   S1 casuale (rappresentativo)
  P_l1     S1 a copertura (densita' temperata, lambda = 1)
  P_mix    mix fisso: 200 casuali + 200 copertura
  P_adapt  diagnosi della popolazione -> lambda0, poi S1 in 4 cicli da 100
           con quota di copertura aggiornata dopo ogni ciclo

Uso (dalla radice del pacchetto):  python3 experiments/pilot2.py --reps 60 --out results/pilot2_results.csv
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402
from pilot1 import B, N1, N2, RHO, SCENARIOS, TAUS, make_population, outcome, region_mask

METHODS = ["P_rand", "P_l1", "P_mix", "P_adapt"]


def phase2(XL, RL, S1, T1, Y1, tau, rng, rep):
    cate = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)
    tau_hat = cate(XL)
    tau_hat[S1] = -np.inf
    pool = np.argsort(-tau_hat)[:int(RHO * len(XL))]
    Rhat = np.zeros(len(XL), bool)
    Rhat[pool] = True
    S2 = rng.choice(pool, N2, replace=False)
    T2 = acc.anticluster(XL[S2], 2, rng)
    Y2 = outcome(XL[S2], T2, RL[S2], tau, rng)
    res = acc.ols_effect(XL[S2], T2, Y2)
    truth = tau * RL[pool].mean()
    inter = (Rhat & RL).sum()
    out = dict(precision=inter / Rhat.sum(), recall=inter / RL.sum() if RL.sum() else np.nan,
               truth=truth, **res)
    out["covered"] = bool(res["lo"] <= truth <= res["hi"])
    return out


def one_rep(args):
    scenario, tau, rep = args
    rng = np.random.default_rng(100_000 * SCENARIOS.index(scenario) + 1_000 * int(tau * 10) + rep)
    X = make_population(rep)
    R = region_mask(X, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng)
    diag = acc.population_diagnostic(ZL)
    measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)

    rows = []
    for m in METHODS:
        t0 = time.time()
        shares = None
        if m == "P_rand":
            S1 = rng.choice(len(L), N1, replace=False)
        elif m == "P_l1":
            S1 = acc.select_tempered(ZL, N1, 1.0, rng, rk=rk)
        elif m == "P_mix":
            cov = acc.select_tempered(ZL, N1 // 2, 1.0, rng, rk=rk)
            rest = np.setdiff1d(np.arange(len(L)), cov)
            S1 = np.concatenate([cov, rng.choice(rest, N1 - len(cov), replace=False)])
        if m != "P_adapt":
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = measure(S1, T1)
        else:
            S1, T1, Y1, A1, shares = acc.adaptive_phase1(
                ZL, XL, measure, N1, rng, lam0=diag["lam0"], rounds=4, rk=rk, seed=rep)
        rec = dict(scenario=scenario, tau=tau, rep=rep, method=m,
                   s1_in_region=int(RL[S1].sum()), lam0=diag["lam0"],
                   tail_ratio=diag["tail_ratio"],
                   final_share_cov=shares[-1] if shares else np.nan,
                   mean_share_cov=float(np.mean(shares)) if shares else np.nan)
        rec.update(phase2(XL, RL, S1, T1, Y1, tau, rng, rep))
        rec["seconds"] = time.time() - t0
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--out", default="results2.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [("nullo", 0.0, r) for r in range(a.reps)]
    jobs += [(s, t, r) for s in SCENARIOS[1:] for t in TAUS for r in range(a.reps)]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
