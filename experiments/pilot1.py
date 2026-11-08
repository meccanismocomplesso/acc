"""
pilot.py - Studio pilota: d = 5, N = 100.000, budget B = 1.000 (n1 = 400, n2 = 600).

Scenari:   nullo | compatta_5 (regione con il 5% di P, densita' media)
           | rara_1_coda (regione con l'1% di P, in coda alla distribuzione)
Metodi:    B1  casuale, nessuna focalizzazione (ATE globale su tutte le B unita')
           B3  rappresentativo (lambda=0) in fase 1, poi casuale, nessuna focalizzazione
           P_rand  S1 casuale + focalizzazione (isola il valore della focalizzazione)
           P_l0 / P_l05 / P_l1  S1 ibrida con lambda = 0 / 0.5 / 1 + focalizzazione
In tutti i metodi trattamento/controllo sono assegnati con anticlustering a 2 gruppi.
Inferenza dei metodi P: solo su S2 (sample splitting), quindi valida dopo la selezione.

Uso (dalla radice del pacchetto):  python3 experiments/pilot1.py --reps 60 --out results/pilot1_results.csv
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

N, D, B, N1 = 100_000, 5, 1_000, 400
N2 = B - N1
RHO = 0.03          # R_hat = il 3% del learning set con tau_hat piu' alto (~1.000 punti)
METHODS = ["B1", "B3", "P_rand", "P_l0", "P_l05", "P_l1"]
SCENARIOS = ["nullo", "compatta_5", "palla_rara_1", "coda_1"]
TAUS = [1.5, 3.0]   # effetto nella regione (rumore sd = 1)


def make_population(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D))
    return X


def region_mask(X: np.ndarray, scenario: str) -> np.ndarray:
    if scenario == "nullo":
        return np.zeros(len(X), bool)
    if scenario == "coda_1":  # valori estremi di una covariata: poca massa, tanto volume
        return X[:, 3] > np.quantile(X[:, 3], 0.99)
    if scenario == "compatta_5":  # palla nel nucleo denso
        c, q = np.array([0, 0, 1.0, 0, 0]), 0.05
    else:  # palla_rara_1: palla piccola in zona a densita' medio-bassa
        c, q = np.array([0, 0, 0, 1.6, 1.6]), 0.01
    dist = np.sqrt(((X - c) ** 2).sum(1))
    return dist <= np.quantile(dist, q)


def outcome(X, T, inR, tau, rng):
    mu = X[:, 0] + 0.5 * X[:, 1]
    return mu + T * tau * inR + rng.standard_normal(len(X))


def one_rep(args):
    scenario, tau, rep = args
    rng = np.random.default_rng(100_000 * SCENARIOS.index(scenario) + 1_000 * int(tau * 10) + rep)
    X = make_population(rep)
    Rstar_all = region_mask(X, scenario)

    # Fase 0: learning / test / holdout con anticlustering; si lavora sul learning set
    parts = acc.anticluster(X, 3, rng)
    L = np.flatnonzero(parts == 0)
    XL, RL = X[L], Rstar_all[L]
    ZL = acc.standardize(XL)
    true_ate = tau * RL.mean()

    # Fase 1: S1 condivisi (B1 e P_rand usano lo stesso S1, cosi' B3 e P_l0):
    # le coppie differiscono solo per la fase 2, confronto appaiato.
    rk = acc.knn_radius(ZL, rng)
    S1s = {"rand": rng.choice(len(L), N1, replace=False)}
    for lam, key in ((0.0, "l0"), (0.5, "l05"), (1.0, "l1")):
        S1s[key] = acc.select_tempered(ZL, N1, lam, rng, rk=rk)
    phase1 = {}
    for key, S1 in S1s.items():
        T1 = acc.anticluster(XL[S1], 2, rng)
        phase1[key] = (S1, T1, outcome(XL[S1], T1, RL[S1], tau, rng))
    use = {"B1": "rand", "P_rand": "rand", "B3": "l0", "P_l0": "l0", "P_l05": "l05", "P_l1": "l1"}

    rows = []
    for m in METHODS:
        t0 = time.time()
        S1, T1, Y1 = phase1[use[m]]
        rec = dict(scenario=scenario, tau=tau, rep=rep, method=m, s1_in_region=int(RL[S1].sum()))

        rest = np.setdiff1d(np.arange(len(L)), S1)
        if m in ("B1", "B3"):
            # nessuna focalizzazione: altre N2 unita' casuali, ATE globale su tutte
            S2 = rng.choice(rest, N2, replace=False)
            T2 = acc.anticluster(XL[S2], 2, rng)
            Y2 = outcome(XL[S2], T2, RL[S2], tau, rng)
            res = acc.ols_effect(np.vstack([XL[S1], XL[S2]]), np.concatenate([T1, T2]),
                                 np.concatenate([Y1, Y2]))
            rec.update(focused=False, region_size=len(L), jaccard=np.nan,
                       precision=np.nan, recall=np.nan, n_used=B, truth=true_ate)
        else:
            cate = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)
            tau_hat = cate(XL)
            tau_hat[S1] = -np.inf                      # S1 non si riusa in fase 2
            pool = np.argsort(-tau_hat)[:int(RHO * len(L))]
            Rhat = np.zeros(len(L), bool)
            Rhat[pool] = True
            inter = (Rhat & RL).sum()
            union = (Rhat | RL).sum()
            rec.update(region_size=int(Rhat.sum()),
                       jaccard=inter / union if union else np.nan,
                       precision=inter / Rhat.sum() if Rhat.sum() else np.nan,
                       recall=inter / RL.sum() if RL.sum() else np.nan)
            S2 = rng.choice(pool, N2, replace=False)
            truth = tau * RL[pool].mean()  # effetto medio vero nella regione R_hat
            rec["focused"] = True
            T2 = acc.anticluster(XL[S2], 2, rng)
            Y2 = outcome(XL[S2], T2, RL[S2], tau, rng)
            res = acc.ols_effect(XL[S2], T2, Y2)  # solo S2: inferenza valida
            rec.update(n_used=N1 + len(S2), truth=truth)
        rec.update(res)
        rec["covered"] = bool(res["lo"] <= rec["truth"] <= res["hi"])
        rec["coverage_radius_S1"] = acc.coverage_radius(ZL, S1)
        rec["seconds"] = time.time() - t0
        rows.append(rec)
    return rows


def phase0_check(reps: int = 20) -> pd.DataFrame:
    """Bilanciamento della fase 0: anticlustering vs split casuale (K = 3)."""
    out = []
    for r in range(reps):
        rng = np.random.default_rng(99_000 + r)
        X = make_population(r)
        t0 = time.time()
        lab = acc.anticluster(X, 3, rng)
        sec = time.time() - t0
        out.append(dict(rep=r, split="anticlustering", seconds=sec, **acc.balance_report(X, lab)))
        lab_r = rng.permutation(np.arange(N) % 3)
        out.append(dict(rep=r, split="casuale", seconds=np.nan, **acc.balance_report(X, lab_r)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [("nullo", 0.0, r) for r in range(a.reps)]
    jobs += [(s, t, r) for s in SCENARIOS[1:] for t in TAUS for r in range(a.reps)]
    t0 = time.time()
    done = 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
