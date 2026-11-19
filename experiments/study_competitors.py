"""
study_competitors.py - acc a confronto con metodi esistenti, a parita' di budget (B = 1.000).

Metodi concorrenti (implementazioni semplificate, con gli stessi stimatori di base di acc dove possibile):
  ATE        esperimento unico non focalizzato: 1.000 unita' casuali, test dell'effetto medio.
  GATES      esperimento unico con divisione del campione (Chernozhukov, Demirer, Duflo, Fernandez-Val,
             "Generic machine learning inference on heterogeneous treatment effects", 2018):
             1.000 unita' casuali; meta' A stima tau(x) (foresta DR), sull'altra meta' B si testa
             l'effetto nel quinto superiore di tau_hat. Una sola divisione (l'articolo ne usa molte).
  VT         virtual twins (Foster, Taylor, Ruberg, Statistics in Medicine 2011), in due stadi:
             600 unita' casuali; T-learner (una foresta per braccio), differenza z(x); albero di regressione
             (profondita' 3, foglie >= 30) di z su x; il sottogruppo e' la foglia con z media piu' alta;
             400 unita' nuove da quella foglia e test.
  ENRICH     arricchimento adattivo a soglia su un biomarcatore (nello spirito di Simon & Simon,
             Biometrics 2013): 600 unita' casuali; sottogruppi candidati {x_j >= quantile} e {x_j <= quantile}
             per ogni covariata e quote 5, 10, 20, 30, 50%, piu' la popolazione intera; si sceglie quello
             con la z piu' alta; 400 unita' nuove da quel sottogruppo e test.
  ML2S       due stadi generico con ML: 600 casuali, foresta DR, top 3%, 400 nuove (come acc senza
             campionamento guidato in fase 1).
Versioni di acc:
  ACC40      acc v0.2 pubblicato (fase 1 adattiva 400 + fase 2 600)
  ACC60      acc v0.2 con fase 1 al 60% (punto 9 del registro)
  ACC_COND   ACC60, ma se `outer_ratio` supera il valore atteso per una normale di oltre il 10%
             si usa la stratificazione del punto 12 (corpo principale 520 + 320, strato esterno 160).
Tutti i test sono unilaterali a 5% (familywise per ACC_COND stratificato) e usano unita' nuove rispetto
alla selezione, oppure gruppi definiti dalle sole covariate.

Misure: correct (rifiuto in un gruppo che contiene effetto vero), reject, precision (quota del gruppo
testato nella regione vera), covered, bias (stima - valore vero nel gruppo testato).
Popolazioni e scenari come study_strat.py. Semi 9.000+ (nuovi).
Uso: python3 experiments/study_competitors.py --reps 40
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402
from pilot1 import B, RHO, SCENARIOS, outcome
from study_strat import POPS, make_population, mask, run_acc, test_stratum

METHODS = ["ATE", "GATES", "VT", "ENRICH", "ML2S", "ACC40", "ACC60", "ACC_COND"]
ALPHA = 0.05
SEED0 = 9_000
N1, N2 = 600, 400


def experiment(XL, RL, S, tau, rng, alpha=ALPHA):
    T = acc.anticluster(XL[S], 2, rng)
    Y = outcome(XL[S], T, RL[S], tau, rng)
    return T, Y, acc.ols_effect(XL[S], T, Y, alpha=alpha)


def rec_from(res, truth, prec):
    return dict(reject=res["reject"], correct=bool(res["reject"] and truth > 0), precision=prec,
                covered=bool(res["lo"] <= truth <= res["hi"]), bias=res["est"] - truth)


def confirm(XL, RL, group, tau, rng, exclude):
    pool = np.setdiff1d(np.flatnonzero(group), exclude)
    S2 = rng.choice(pool, min(N2, len(pool)), replace=False)
    _, _, res = experiment(XL, RL, S2, tau, rng)
    return rec_from(res, tau * RL[group].mean(), RL[group].mean())


def m_ate(XL, RL, tau, rng, rep):
    S = rng.choice(len(XL), B, replace=False)
    _, _, res = experiment(XL, RL, S, tau, rng)
    return rec_from(res, tau * RL.mean(), RL.mean())


def m_gates(XL, RL, tau, rng, rep):
    S = rng.choice(len(XL), B, replace=False)
    T = acc.anticluster(XL[S], 2, rng)
    Y = outcome(XL[S], T, RL[S], tau, rng)
    half = np.zeros(B, bool)
    for a in (0, 1):
        idx = np.flatnonzero(T == a)
        half[rng.permutation(idx)[: len(idx) // 2]] = True
    th = acc.fit_cate(XL[S[half]], T[half], Y[half], seed=rep, learner="dr", leaf=10)(XL[S[~half]])
    top = np.flatnonzero(~half)[th >= np.quantile(th, 0.8)]
    res = acc.ols_effect(XL[S[top]], T[top], Y[top])
    return rec_from(res, tau * RL[S[top]].mean(), RL[S[top]].mean())


def m_vt(XL, RL, tau, rng, rep):
    S1 = rng.choice(len(XL), N1, replace=False)
    T1, Y1, _ = experiment(XL, RL, S1, tau, rng)
    kw = dict(n_estimators=200, min_samples_leaf=5, n_jobs=1, random_state=rep)
    m1 = RandomForestRegressor(**kw).fit(XL[S1][T1 == 1], Y1[T1 == 1])
    m0 = RandomForestRegressor(**kw).fit(XL[S1][T1 == 0], Y1[T1 == 0])
    z = m1.predict(XL[S1]) - m0.predict(XL[S1])
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=rep).fit(XL[S1], z)
    leafL = tree.apply(XL)
    leaves = np.unique(tree.apply(XL[S1]))
    vals = {lf: tree.tree_.value[lf][0][0] for lf in leaves}
    order = sorted(leaves, key=lambda lf: -vals[lf])
    best = next((lf for lf in order if (leafL == lf).sum() - np.isin(np.flatnonzero(leafL == lf), S1).sum() >= N2),
                order[0])
    return confirm(XL, RL, leafL == best, tau, rng, S1)


def m_enrich(XL, RL, tau, rng, rep):
    S1 = rng.choice(len(XL), N1, replace=False)
    T1, Y1, _ = experiment(XL, RL, S1, tau, rng)
    cands = [np.ones(len(XL), bool)]
    for j in range(XL.shape[1]):
        for q in (0.05, 0.1, 0.2, 0.3, 0.5):
            cands.append(XL[:, j] >= np.quantile(XL[:, j], 1 - q))
            cands.append(XL[:, j] <= np.quantile(XL[:, j], q))
    best, bz = cands[0], -np.inf
    for g in cands:
        inS = g[S1]
        if min((T1[inS] == 1).sum(), (T1[inS] == 0).sum()) < 10 or (g.sum() - inS.sum()) < N2:
            continue
        r = acc.ols_effect(XL[S1][inS], T1[inS], Y1[inS])
        z = r["est"] / r["se"] if r["se"] > 0 else -np.inf
        if z > bz:
            best, bz = g, z
    return confirm(XL, RL, best, tau, rng, S1)


def m_ml2s(XL, RL, tau, rng, rep):
    S1 = rng.choice(len(XL), N1, replace=False)
    T1, Y1, _ = experiment(XL, RL, S1, tau, rng)
    th = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)(XL)
    th[S1] = -np.inf
    g = np.zeros(len(XL), bool)
    g[np.argsort(-th)[:int(RHO * len(XL))]] = True
    return confirm(XL, RL, g, tau, rng, S1)


def m_acc(XL, RL, tau, rng, rep, n1):
    res, truth, prec = run_acc(XL, RL, n1, B - n1, tau, rng, rep, ALPHA)
    return rec_from(res, truth, prec)


def m_acc_cond(XL, RL, tau, rng, rep):
    if acc.outer_ratio(XL)["relative"] <= 1.10:
        out = m_acc(XL, RL, tau, rng, rep, N1)
        out["stratified"] = False
        return out
    O = acc.outer_stratum(XL, 0.02)
    M = ~O
    res, truth, prec = run_acc(XL[M], RL[M], 520, 320, tau, rng, rep, ALPHA / 2)
    resO, truthO = test_stratum(XL[O], RL[O], 160, tau, rng, ALPHA / 2)
    tests = [(res, truth, prec), (resO, truthO, RL[O].mean())]
    rej = [t for t in tests if t[0]["reject"]]
    pick = max(rej, key=lambda t: t[0]["est"] / t[0]["se"]) if rej else tests[0]
    return dict(reject=bool(rej), correct=any(t[1] > 0 for t in rej), precision=pick[2],
                covered=all(r["lo"] <= tr <= r["hi"] for r, tr, _ in tests),
                bias=pick[0]["est"] - pick[1], stratified=True)


def one_rep(args):
    pop, scenario, tau, rep = args
    base = [POPS.index(pop), ["nullo", "compatta_5", "palla_rara_1", "coda_1", "isolato"].index(scenario),
            int(tau * 10), rep]
    rng0 = np.random.default_rng(base)
    X, group = make_population(rep, pop)
    R = mask(X, group, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, RL = X[L], R[L]
    rows = []
    for i, m in enumerate(METHODS):
        rng = np.random.default_rng(base + [i])
        t0 = time.time()
        if m == "ATE":
            out = m_ate(XL, RL, tau, rng, rep)
        elif m == "GATES":
            out = m_gates(XL, RL, tau, rng, rep)
        elif m == "VT":
            out = m_vt(XL, RL, tau, rng, rep)
        elif m == "ENRICH":
            out = m_enrich(XL, RL, tau, rng, rep)
        elif m == "ML2S":
            out = m_ml2s(XL, RL, tau, rng, rep)
        elif m == "ACC40":
            out = m_acc(XL, RL, tau, rng, rep, 400)
        elif m == "ACC60":
            out = m_acc(XL, RL, tau, rng, rep, 600)
        else:
            out = m_acc_cond(XL, RL, tau, rng, rep)
        rows.append(dict(pop=pop, scenario=scenario, tau=tau, rep=rep, method=m,
                         seconds=time.time() - t0, **out))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="results/study_competitors_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cells = []
    for pop in ("gauss", "t3"):
        cells += [(pop, "nullo", 0.0)] + [(pop, s, t) for s in SCENARIOS[1:] for t in (1.5, 3.0)]
    cells += [("mix", "nullo", 0.0)] + [("mix", "isolato", t) for t in (1.5, 3.0)]
    jobs = [c + (SEED0 + r,) for r in range(a.reps) for c in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
