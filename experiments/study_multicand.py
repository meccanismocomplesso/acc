"""
study_multicand.py - Fase 2 con piu' regioni candidate.

Dalla stima di tau(x) della fase 1 si prendono le unita' del learning set con tau_hat nel top 3% * K
e le si divide con k-means in K regioni candidate distinte (ciascuna ~3%, come la R_hat attuale).
Varianti di fase 2, a parita' di budget n2 = B - n1:
  BASE       fase 2 attuale (pilot2): una sola R_hat = top 3%, test unilaterale a alpha = 5%
  BONF_K     n2 diviso tra le K candidate, ognuna testata a alpha / K (Bonferroni).
             Rifiuto globale se almeno una candidata rifiuta.
  SCREEN_K   meta' di n2 divisa tra le K candidate (valutazione), si sceglie la candidata con la z
             piu' alta; l'altra meta' di n2 va a unita' nuove di quella candidata, test unico a 5%.
La selezione usa solo dati precedenti al test, quindi entrambe le varianti controllano l'errore di
I tipo al 5% (familywise per BONF).

Misure: reject (rifiuto globale); correct (rifiuto in una candidata che contiene davvero effetto);
false_region (rifiuto in una candidata con effetto vero nullo); found_any (almeno una candidata
con precisione >= 5%); covered (tutti gli IC delle candidate testate contengono il loro valore vero).

Fase 1: metodo adattivo v0.2 con n1 = 400 e 600. Popolazione gaussiana, scenari del pilota 2.
K = 2, 3, 4, quota di valutazione 50% e dimensione delle candidate fissate prima di vedere i risultati.
Semi 7.000+ (nuovi). Uso: python3 experiments/study_multicand.py --reps 80
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402
from pilot1 import B, RHO, SCENARIOS, make_population, outcome, region_mask

SPLITS = [400, 600]
KS = [2, 3, 4]
SEED0 = 7_000
ALPHA = 0.05


def test_in(XL, RL, cand, n, tau, rng, alpha, exclude=None):
    pool = cand if exclude is None else np.setdiff1d(cand, exclude)
    S = rng.choice(pool, min(n, len(pool)), replace=False)
    T = acc.anticluster(XL[S], 2, rng)
    Y = outcome(XL[S], T, RL[S], tau, rng)
    res = acc.ols_effect(XL[S], T, Y, alpha=alpha)
    truth = tau * RL[cand].mean()
    return res, truth, S


def candidates(ZL, tau_hat, K, rng):
    top = np.argsort(-tau_hat)[:int(RHO * K * len(ZL))]
    if K == 1:
        return [top]
    lab = KMeans(n_clusters=K, n_init=3, random_state=int(rng.integers(1 << 31))).fit_predict(ZL[top])
    return [top[lab == k] for k in range(K)]


def summarize(cands, tested, RL, tau):
    """tested: lista di (indice candidata, res, truth)."""
    prec = [RL[c].mean() for c in cands]
    rej = [(i, r, t) for i, r, t in tested if r["reject"]]
    return dict(reject=bool(rej), correct=any(t > 0 for _, _, t in rej),
                false_region=any(t == 0 for _, _, t in rej),
                found_any=bool(max(prec) >= 0.05), best_precision=float(max(prec)),
                covered=all(r["lo"] <= t <= r["hi"] for _, r, t in tested))


def one_rep(args):
    scenario, tau, rep = args
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
    for n1 in SPLITS:
        n2 = B - n1
        rng = np.random.default_rng(base + [n1])
        measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
        S1, T1, Y1, _, _ = acc.adaptive_phase1(ZL, XL, measure, n1, rng, lam0=lam0,
                                               rounds=4, rk=rk, seed=rep)
        tau_hat = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)(XL)
        tau_hat[S1] = -np.inf
        common = dict(scenario=scenario, tau=tau, rep=rep, n1=n1)
        # BASE
        r2 = np.random.default_rng(base + [n1, 0])
        c1 = candidates(ZL, tau_hat, 1, r2)
        res, truth, _ = test_in(XL, RL, c1[0], n2, tau, r2, ALPHA)
        rows.append(dict(common, variant="BASE", **summarize(c1, [(0, res, truth)], RL, tau)))
        for K in KS:
            rK = np.random.default_rng(base + [n1, K])
            cands = candidates(ZL, tau_hat, K, rK)
            # BONF_K
            g = np.random.default_rng(base + [n1, K, 1])
            tested = []
            for i, c in enumerate(cands):
                res, truth, _ = test_in(XL, RL, c, n2 // K, tau, g, ALPHA / K)
                tested.append((i, res, truth))
            rows.append(dict(common, variant=f"BONF_{K}", **summarize(cands, tested, RL, tau)))
            # SCREEN_K
            g = np.random.default_rng(base + [n1, K, 2])
            n_screen = (n2 // 2) // K
            zs, used = [], []
            for c in cands:
                res, _, S = test_in(XL, RL, c, n_screen, tau, g, ALPHA)
                zs.append(res["est"] / res["se"] if res["se"] > 0 else -np.inf)
                used.append(S)
            best = int(np.argmax(zs))
            res, truth, _ = test_in(XL, RL, cands[best], n2 - n_screen * K, tau, g, ALPHA,
                                    exclude=used[best])
            rec = summarize(cands, [(best, res, truth)], RL, tau)
            rec["picked_precision"] = float(RL[cands[best]].mean())
            rows.append(dict(common, variant=f"SCREEN_{K}", **rec))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--out", default="results/study_multicand_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cells = [("nullo", 0.0)] + [(s, t) for s in SCENARIOS[1:] for t in (1.5, 3.0)]
    jobs = [(s, t, SEED0 + r) for r in range(a.reps) for s, t in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 40 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
