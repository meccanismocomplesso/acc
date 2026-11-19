"""
dev_null_strat.py - Errore di I tipo della stratificazione (punto 12) nello scenario nullo.

Nei punti 12 e 13 la versione stratificata sul mix ha rifiutato in 8 casi su 80 (10%) senza effetto.
Qui si separano i due test che la compongono e si stimano con molte ripetizioni:
  M  corpo principale: fase 1 adattiva (520) una volta per replica, poi la fase 2 (320 unita' nuove da R_hat)
     ripetuta `--redraws_m` volte; test a alpha/2.
  O  strato esterno (2% piu' lontano, fissato dalle covariate): 160 unita' casuali, ripetuto `--redraws_o` volte;
     test a alpha/2 con quattro varianti:
       hc1        stima attuale (OLS con covariate, errori standard HC1, assegnazione anticluster deterministica)
       hc3        errori standard HC3 (piu' prudenti con punti ad alta leva)
       jit_hc1    assegnazione anticluster con jitter = 0.05 (randomizzazione vera, punto 2), errori HC1
       jit_hc3    jitter e HC3
Il tasso globale si ottiene combinando i due test (rifiuto se almeno uno rifiuta); essendo indipendenti,
P(globale) = 1 - (1 - pM)(1 - pO).
Popolazioni mix e t3. Semi 3.200-3.599 (liberi, non di valutazione).
Uso: python3 experiments/dev_null_strat.py --reps 200
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402
from pilot1 import RHO, outcome
from study_strat import POPS, make_population

ALPHA = 0.05
SEED0 = 3_200
RM, RO = 10, 50


def ols(X, T, Y, kind="hc1"):
    D = np.column_stack([np.ones(len(Y)), T, X])
    bread = np.linalg.pinv(D.T @ D)
    beta = bread @ D.T @ Y
    e = Y - D @ beta
    n, p = D.shape
    if kind == "hc3":
        h = np.einsum("ij,jk,ik->i", D, bread, D)
        w = e ** 2 / np.clip(1 - h, 1e-6, None) ** 2
        V = bread @ (D * w[:, None]).T @ D @ bread
    else:
        V = bread @ (D * (e ** 2)[:, None]).T @ D @ bread * n / (n - p)
    return float(beta[1] / np.sqrt(V[1, 1]))


def one_rep(args):
    pop, rep = args
    base = [POPS.index(pop), rep, 31]
    rng = np.random.default_rng(base)
    X, _ = make_population(rep, pop)
    L = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)
    XL = X[L]
    RL = np.zeros(len(L), bool)
    O = acc.outer_stratum(XL, 0.02)
    XM, XO = XL[~O], XL[O]
    ZM = acc.standardize(XM)
    rk = acc.knn_radius(ZM, rng)
    lam0 = acc.population_diagnostic(ZM)["lam0"]
    measure = lambda idx, T: outcome(XM[idx], T, RL[~O][idx], 0.0, rng)
    S1, T1, Y1, _, _ = acc.adaptive_phase1(ZM, XM, measure, 520, rng, lam0=lam0, rounds=4, rk=rk, seed=rep)
    th = acc.fit_cate(XM[S1], T1, Y1, seed=rep, learner="dr", leaf=10)(XM)
    th[S1] = -np.inf
    pool = np.argsort(-th)[:int(RHO * len(XM))]
    zc = norm.ppf(1 - ALPHA / 2)
    rows = []
    for k in range(RM):
        S2 = rng.choice(pool, 320, replace=False)
        T2 = acc.anticluster(XM[S2], 2, rng)
        Y2 = outcome(XM[S2], T2, np.zeros(320, bool), 0.0, rng)
        rows.append(dict(pop=pop, rep=rep, test="M", variant="hc1", k=k, z=ols(XM[S2], T2, Y2)))
    for k in range(RO):
        S = rng.choice(len(XO), 160, replace=False)
        Y0 = outcome(XO[S], np.zeros(160, int), np.zeros(160, bool), 0.0, rng)   # esiti sotto H0
        Td = acc.anticluster(XO[S], 2, rng)
        Tj = acc.anticluster(XO[S], 2, rng, jitter=0.05)
        for v, T, kind in (("hc1", Td, "hc1"), ("hc3", Td, "hc3"), ("jit_hc1", Tj, "hc1"), ("jit_hc3", Tj, "hc3")):
            rows.append(dict(pop=pop, rep=rep, test="O", variant=v, k=k, z=ols(XO[S], T, Y0, kind)))
    for r in rows:
        r["reject"] = r["z"] > zc
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="results/dev_null_strat_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(p, SEED0 + r) for r in range(a.reps) for p in ("mix", "t3")]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
