"""
study_strat.py - acc stratificato: isolare all'inizio lo strato raro nelle covariate, studiarlo a parte,
riunire alla fine.

Strato esterno O = 2% del learning set piu' lontano dal centro (`acc.outer_stratum`, distanza robusta,
solo covariate). Corpo principale M = il resto. Budget totale B = 1.000.
Metodi:
  P_rand      fase 1 casuale (600) + fase 2 attuale (400) su tutto il learning set
  BASE        acc v0.2: fase 1 adattiva (600) + fase 2 attuale (400) su tutto il learning set
  CLEAN       acc v0.2 solo su M con tutto il budget (600 + 400); O ignorato
  STRAT       M: fase 1 adattiva (520) + fase 2 (320) su R_hat di M, test a alpha/2.
              O: 160 unita' casuali, test dell'effetto medio di O a alpha/2 (O e' fissato dalle sole
              covariate, quindi il suo test non richiede unita' nuove). Rifiuto globale se uno dei due
              rifiuta (Bonferroni, errore familywise <= 5%).
Popolazioni (d = 5, N = 100.000):
  gauss   normale standard
  t3      t di Student con 3 g.d.l., varianza 1
  mix     99% normale standard + 1% gruppo isolato N((0, 0, 3.5, 3.5, 0), 0.5^2 I)
Scenari: gauss e t3 come il pilota 2; mix: nullo, isolato (effetto nel gruppo isolato), compatta_5.
Misure: correct (rifiuto in un test con effetto vero > 0), reject, covered, in_O (quota della regione
vera che cade nello strato esterno). Semi 8.000+ (nuovi). Scelte (2%, 160 unita', 520/320) fissate prima.
Uso: python3 experiments/study_strat.py --reps 40
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
from pilot1 import B, D, N, RHO, SCENARIOS, outcome, region_mask

POPS = ["gauss", "t3", "mix"]
METHODS = ["P_rand", "BASE", "CLEAN", "STRAT"]
ALPHA = 0.05
SEED0 = 8_000
N1, N_O = 600, 160
N1_M, N2_M = 520, 320


def make_population(seed, pop):
    rng = np.random.default_rng([seed, POPS.index(pop)])
    group = np.zeros(N, bool)
    if pop == "gauss":
        X = rng.standard_normal((N, D))
    elif pop == "t3":
        X = rng.standard_t(3, (N, D)) / np.sqrt(3.0)
    else:
        X = rng.standard_normal((N, D))
        group[: N // 100] = True
        X[group] = np.array([0, 0, 3.5, 3.5, 0]) + 0.5 * rng.standard_normal((group.sum(), D))
        perm = rng.permutation(N)
        X, group = X[perm], group[perm]
    return X, group


def mask(X, group, scenario):
    return group.copy() if scenario == "isolato" else region_mask(X, scenario)


def phase1(method, ZL, XL, RL, n1, tau, rng, rk, lam0, rep):
    measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
    if method == "P_rand":
        S1 = rng.choice(len(XL), n1, replace=False)
        T1 = acc.anticluster(XL[S1], 2, rng)
        return S1, T1, measure(S1, T1)
    S1, T1, Y1, _, _ = acc.adaptive_phase1(ZL, XL, measure, n1, rng, lam0=lam0, rounds=4, rk=rk, seed=rep)
    return S1, T1, Y1


def phase2(XL, RL, S1, T1, Y1, n2, tau, rng, rep, alpha):
    tau_hat = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)(XL)
    tau_hat[S1] = -np.inf
    pool = np.argsort(-tau_hat)[:int(RHO * len(XL))]
    S2 = rng.choice(pool, n2, replace=False)
    T2 = acc.anticluster(XL[S2], 2, rng)
    Y2 = outcome(XL[S2], T2, RL[S2], tau, rng)
    res = acc.ols_effect(XL[S2], T2, Y2, alpha=alpha)
    return res, tau * RL[pool].mean(), RL[pool].mean()


def test_stratum(XO, RO, n, tau, rng, alpha):
    S = rng.choice(len(XO), min(n, len(XO)), replace=False)
    T = acc.anticluster(XO[S], 2, rng)
    Y = outcome(XO[S], T, RO[S], tau, rng)
    return acc.ols_effect(XO[S], T, Y, alpha=alpha), tau * RO.mean()


def run_acc(XL, RL, n1, n2, tau, rng, rep, alpha, method="BASE"):
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng)
    lam0 = acc.population_diagnostic(ZL)["lam0"]
    S1, T1, Y1 = phase1(method, ZL, XL, RL, n1, tau, rng, rk, lam0, rep)
    return phase2(XL, RL, S1, T1, Y1, n2, tau, rng, rep, alpha)


def one_rep(args):
    pop, scenario, tau, rep = args
    base = [POPS.index(pop), ["nullo", "compatta_5", "palla_rara_1", "coda_1", "isolato"].index(scenario),
            int(tau * 10), rep]
    rng0 = np.random.default_rng(base)
    X, group = make_population(rep, pop)
    R = mask(X, group, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, RL = X[L], R[L]
    O = acc.outer_stratum(XL, 0.02)
    in_O = float(RL[O].sum() / RL.sum()) if RL.sum() else np.nan
    rows = []
    for i, m in enumerate(METHODS):
        rng = np.random.default_rng(base + [i])
        tests = []
        if m in ("P_rand", "BASE"):
            res, truth, prec = run_acc(XL, RL, N1, B - N1, tau, rng, rep, ALPHA, m)
            tests.append((res, truth))
        elif m == "CLEAN":
            M = ~O
            res, truth, prec = run_acc(XL[M], RL[M], N1, B - N1, tau, rng, rep, ALPHA)
            tests.append((res, truth))
        else:
            M = ~O
            res, truth, prec = run_acc(XL[M], RL[M], N1_M, N2_M, tau, rng, rep, ALPHA / 2)
            tests.append((res, truth))
            resO, truthO = test_stratum(XL[O], RL[O], N_O, tau, rng, ALPHA / 2)
            tests.append((resO, truthO))
        rej = [(r, t) for r, t in tests if r["reject"]]
        rows.append(dict(pop=pop, scenario=scenario, tau=tau, rep=rep, method=m, in_O=in_O,
                         precision=prec, reject=bool(rej), correct=any(t > 0 for _, t in rej),
                         reject_main=bool(tests[0][0]["reject"]),
                         reject_O=bool(tests[1][0]["reject"]) if len(tests) > 1 else np.nan,
                         covered=all(r["lo"] <= t <= r["hi"] for r, t in tests)))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="results/study_strat_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cells = []
    for pop in ("gauss", "t3"):
        cells += [(pop, "nullo", 0.0)] + [(pop, s, t) for s in SCENARIOS[1:] for t in (1.5, 3.0)]
    cells += [("mix", "nullo", 0.0)] + [("mix", s, t) for s in ("isolato", "compatta_5") for t in (1.5, 3.0)]
    jobs = [c + (SEED0 + r,) for r in range(a.reps) for c in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
