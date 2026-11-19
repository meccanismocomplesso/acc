"""
eval_central.py - Valutazione del disegno centrale del piano (docs/PIANO_SIMULAZIONE.md, sez. 2-5)
sui semi riservati (da 10.000 in su). Codice congelato: tag v0.3-dev-frozen.

Disegno: d = 5, covariate gaussiane, N = 10^5, B = 1.000, n1/B = 0,4 (tranne ACC_v03 e P_rand_60, vedi CHANGELOG).
Geometrie: nullo; omogeneo; compatta_5; palla_rara_1; coda_1; multi_2x2; gradiente; tau = 1,5 e 3.
Repliche: 1.000 per il nullo, 300 per le altre celle. Seme della replica r della cella c: 10.000 + 1.000 * c + r
(c = indice della cella, r < 1.000), quindi tutti >= 10.000.

Metodi (sez. 2 del piano):
  B1          fase 1 casuale, fase 2 casuale, ATE globale su tutte le B unita'
  B3          fase 1 rappresentativa (densita' temperata lambda = 0), fase 2 casuale, ATE globale
  B4          enrichment classico: sottogruppi = sopra/sotto la mediana di ciascuna covariata (2d = 10);
              fase 2 dal sottogruppo con l'effetto stimato piu' alto in fase 1; test su S2
  P_rand      fase 1 casuale, R_hat = top 3% di tau_hat, test su S2
  P_l1        fase 1 densita' temperata lambda = 1, come P_rand
  P_mix       fase 1 50% casuale + 50% lambda = 1, come P_rand
  P_adapt_v02 fase 1 adattiva v0.2, come P_rand
  ACC_v03     acc.pipeline_v03 (configurazione congelata; sostituisce P_adapt_v03, vedi CHANGELOG)
  B5          oracolo di campionamento: S2 (600) estratto dalla regione vera R*, limite superiore
  P_rand_60   fuori piano: P_rand con n1/B = 0,6
Regione vera R*: per omogeneo tutta la popolazione; per gradiente (tau(x) = tau * Phi(x3)) il 3% con x3 piu' alto
(usato solo per B5, precisione e richiamo; l'effetto vero usa sempre il peso continuo).

Riprendibile: i job gia' presenti nel file di output vengono saltati.
Uso: python3 experiments/eval_central.py --out results/eval_central_results.csv
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
from pilot1 import N, D, B

GEOMS = ["nullo", "omogeneo", "compatta_5", "palla_rara_1", "coda_1", "multi_2x2", "gradiente"]
TAUS = [1.5, 3.0]
CELLS = [("nullo", 0.0)] + [(g, t) for g in GEOMS[1:] for t in TAUS]
REPS = {"nullo": 1000}
REPS_DEFAULT = 300
METHODS = ["B1", "B3", "B4", "P_rand", "P_l1", "P_mix", "P_adapt_v02", "ACC_v03", "B5", "P_rand_60"]
RHO, N1, ALPHA = 0.03, 400, 0.05
SEED0 = 10_000


def population(seed):
    return np.random.default_rng(seed).standard_normal((N, D))


def ball(X, c, q):
    dist = np.sqrt(((X - np.asarray(c)) ** 2).sum(1))
    return dist <= np.quantile(dist, q)


def weights(X, geom):
    """Peso dell'effetto w(x) in [0, 1] e regione vera R* (booleana)."""
    if geom == "nullo":
        w = np.zeros(len(X)); return w, w > 0
    if geom == "omogeneo":
        w = np.ones(len(X)); return w, w > 0
    if geom == "compatta_5":
        R = ball(X, [0, 0, 1.0, 0, 0], 0.05)
    elif geom == "palla_rara_1":
        R = ball(X, [0, 0, 0, 1.6, 1.6], 0.01)
    elif geom == "coda_1":
        R = X[:, 3] > np.quantile(X[:, 3], 0.99)
    elif geom == "multi_2x2":
        R = ball(X, [0, 0, 1.2, 1.2, 0], 0.02) | ball(X, [0, 0, -1.2, -1.2, 0], 0.02)
    else:  # gradiente: tau(x) = tau * Phi(x3), x3 = terza covariata
        w = norm.cdf(X[:, 2])
        return w, X[:, 2] > np.quantile(X[:, 2], 0.97)
    return R.astype(float), R


def outcome(X, T, w, tau, rng):
    return X[:, 0] + 0.5 * X[:, 1] + T * tau * w + rng.standard_normal(len(X))


def ate_test(XL, W, S, tau, rng):
    T = acc.anticluster(XL[S], 2, rng)
    Y = outcome(XL[S], T, W[S], tau, rng)
    return acc.ols_effect(XL[S], T, Y), tau * W.mean(), np.nan, np.nan


def confirm(XL, W, R, group_mask, S1, n2, tau, rng):
    pool = np.setdiff1d(np.flatnonzero(group_mask), S1)
    S2 = rng.choice(pool, min(n2, len(pool)), replace=False)
    T2 = acc.anticluster(XL[S2], 2, rng)
    Y2 = outcome(XL[S2], T2, W[S2], tau, rng)
    g = group_mask
    inter = (g & R).sum()
    return (acc.ols_effect(XL[S2], T2, Y2), tau * W[g].mean(),
            inter / g.sum(), inter / R.sum() if R.sum() else np.nan)


def focus(XL, W, R, S1, T1, Y1, n2, tau, rng, rep):
    th = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)(XL)
    th[S1] = -np.inf
    g = np.zeros(len(XL), bool)
    g[np.argsort(-th)[:int(RHO * len(XL))]] = True
    return confirm(XL, W, R, g, S1, n2, tau, rng)


def one_rep(args):
    ci, rep_in_cell = args
    geom, tau = CELLS[ci]
    seed = SEED0 + 1_000 * ci + rep_in_cell
    rng0 = np.random.default_rng([seed, 0])
    X = population(seed)
    Wall, Rall = weights(X, geom)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, W, R = X[L], Wall[L], Rall[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    lam0 = acc.population_diagnostic(ZL)["lam0"]
    rows = []
    for mi, m in enumerate(METHODS):
        rng = np.random.default_rng([seed, mi + 1])
        meas = lambda idx, T: outcome(XL[idx], T, W[idx], tau, rng)
        t0 = time.time()
        s1_in = np.nan
        smd = np.nan
        extra = {}
        if m == "B1":
            S = rng.choice(len(XL), B, replace=False)
            res, truth, prec, rec = ate_test(XL, W, S, tau, rng)
        elif m == "B3":
            S1 = acc.select_tempered(ZL, N1, 0.0, rng, rk=rk)
            rest = np.setdiff1d(np.arange(len(XL)), S1)
            S = np.concatenate([S1, rng.choice(rest, B - N1, replace=False)])
            res, truth, prec, rec = ate_test(XL, W, S, tau, rng)
            s1_in = R[S1].sum()
        elif m == "B4":
            S1 = rng.choice(len(XL), N1, replace=False)
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = meas(S1, T1)
            best, bz = None, -np.inf
            for j in range(XL.shape[1]):
                med = np.median(XL[:, j])
                for g in (XL[:, j] > med, XL[:, j] <= med):
                    inS = g[S1]
                    r = acc.ols_effect(XL[S1][inS], T1[inS], Y1[inS])
                    if r["est"] > bz:
                        best, bz = g, r["est"]
            res, truth, prec, rec = confirm(XL, W, R, best, S1, B - N1, tau, rng)
            s1_in = R[S1].sum()
        elif m == "B5":
            g = R if R.sum() else np.ones(len(XL), bool)
            res, truth, prec, rec = confirm(XL, W, R, g, np.array([], int), B - N1, tau, rng)
        elif m == "ACC_v03":
            out = acc.pipeline_v03(XL, meas, B, rng, seed=seed)
            tests = out["tests"]
            rej = [t for t in tests if t["res"]["reject"]]
            pick = max(rej, key=lambda t: t["res"]["est"] / t["res"]["se"]) if rej else tests[0]
            g = np.zeros(len(XL), bool); g[pick["group"]] = True
            res = pick["res"]
            truth = tau * W[g].mean()
            inter = (g & R).sum()
            prec, rec = inter / g.sum(), inter / R.sum() if R.sum() else np.nan
            s1_in = R[np.concatenate([t["S1"] for t in tests])].sum()
            extra = dict(stratified=out["stratified"], n_tests=len(tests),
                         reject_any=out["reject"],
                         covered_all=all(t["res"]["lo"] <= tau * W[t["group"]].mean() <= t["res"]["hi"]
                                         for t in tests))
        else:
            n1 = 600 if m == "P_rand_60" else N1
            if m in ("P_rand", "P_rand_60"):
                S1 = rng.choice(len(XL), n1, replace=False)
            elif m == "P_l1":
                S1 = acc.select_tempered(ZL, n1, 1.0, rng, rk=rk)
            elif m == "P_mix":
                cov = acc.select_tempered(ZL, n1 // 2, 1.0, rng, rk=rk)
                rest = np.setdiff1d(np.arange(len(XL)), cov)
                S1 = np.concatenate([cov, rng.choice(rest, n1 - len(cov), replace=False)])
            if m == "P_adapt_v02":
                S1, T1, Y1, _, _ = acc.adaptive_phase1(ZL, XL, meas, n1, rng, lam0=lam0, rounds=4,
                                                       rk=rk, seed=seed)
            else:
                T1 = acc.anticluster(XL[S1], 2, rng)
                Y1 = meas(S1, T1)
            smd = acc.balance_report(XL[S1], T1)["max_abs_smd"]
            res, truth, prec, rec = focus(XL, W, R, S1, T1, Y1, B - n1, tau, rng, seed)
            s1_in = R[S1].sum()
        reject = extra.get("reject_any", res["reject"])
        rows.append(dict(cell=ci, geom=geom, tau=tau, rep=rep_in_cell, seed=seed, method=m,
                         reject=bool(reject), correct=bool(reject and truth > 0),
                         est=res["est"], se=res["se"], truth=truth,
                         covered=extra.get("covered_all", bool(res["lo"] <= truth <= res["hi"])),
                         precision=prec, recall=rec, s1_in_region=s1_in, t1_max_smd=smd,
                         stratified=extra.get("stratified", np.nan), seconds=time.time() - t0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eval_central_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="solo per prove: primi N job")
    a = ap.parse_args()
    jobs = [(ci, r) for r in range(max(REPS.get(g, REPS_DEFAULT) for g, _ in CELLS))
            for ci, (g, _) in enumerate(CELLS) if r < REPS.get(g, REPS_DEFAULT)]
    done_keys = set()
    if Path(a.out).exists():
        prev = pd.read_csv(a.out, usecols=["cell", "rep", "method"])
        cnt = prev.groupby(["cell", "rep"]).method.nunique()
        done_keys = set(cnt[cnt == len(METHODS)].index)
    jobs = [j for j in jobs if j not in done_keys]
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} job da eseguire ({len(done_keys)} gia' fatti)", flush=True)
    t0, done = time.time(), 0
    header = not Path(a.out).exists()
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs, chunksize=2):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=header)
            header = False
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"{done}/{len(jobs)} job, {el:.0f}s, mancano circa {el / done * (len(jobs) - done) / 60:.0f} min",
                      flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
