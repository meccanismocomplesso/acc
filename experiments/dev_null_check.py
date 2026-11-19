"""
dev_null_check.py - Controllo dell'errore di I tipo nello scenario nullo (limite noto della v0.2:
P_adapt rifiuta in 5/60 = 8,3% dei casi nel pilota 2).

Idea: sotto H0 il test di fase 2 dipende dalla fase 1 solo attraverso l'insieme R_hat da cui
si estrae S2. Per ogni replica si esegue quindi la fase 1 vera (P_adapt v0.2 e P_rand) una volta,
e poi si ripete la sola fase 2 `--redraws` volte (nuovo S2, nuovo T2, nuovo rumore).
Questo misura il tasso di rifiuto su regioni R_hat realistiche con un errore Monte Carlo
molto piu' piccolo di quello di 60 repliche singole.

Semi: 2.000-2.999, disgiunti da pilota (0-99), sviluppo (1.000-1.999) e valutazione (>= 10.000).

Uso:  python3 experiments/dev_null_check.py --reps 150 --redraws 50 --out results/dev_null_check.csv
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
from pilot1 import N1, N2, RHO, make_population, outcome

SEED0 = 2_000
REDRAWS = 50


def one_rep(rep):
    rng = np.random.default_rng(rep)
    X = make_population(rep)
    L = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)
    XL = X[L]
    RL = np.zeros(len(L), bool)
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng)
    lam0 = acc.population_diagnostic(ZL)["lam0"]
    measure = lambda idx, T: outcome(XL[idx], T, RL[idx], 0.0, rng)
    rows = []
    for m in ("P_rand", "P_adapt"):
        if m == "P_rand":
            S1 = rng.choice(len(L), N1, replace=False)
            T1 = acc.anticluster(XL[S1], 2, rng)
            Y1 = measure(S1, T1)
        else:
            S1, T1, Y1, _, _ = acc.adaptive_phase1(ZL, XL, measure, N1, rng, lam0=lam0,
                                                   rounds=4, rk=rk, seed=rep)
        tau_hat = acc.fit_cate(XL[S1], T1, Y1, seed=rep, learner="dr", leaf=10)(XL)
        tau_hat[S1] = -np.inf
        pool = np.argsort(-tau_hat)[:int(RHO * len(XL))]
        for k in range(REDRAWS):
            S2 = rng.choice(pool, N2, replace=False)
            T2 = acc.anticluster(XL[S2], 2, rng)
            Y2 = outcome(XL[S2], T2, RL[S2], 0.0, rng)
            res = acc.ols_effect(XL[S2], T2, Y2)
            rows.append(dict(rep=rep, method=m, redraw=k, est=res["est"], se=res["se"],
                             reject=res["reject"], covered=bool(res["lo"] <= 0 <= res["hi"])))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--redraws", type=int, default=50)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out", default="results/dev_null_check.csv")
    a = ap.parse_args()
    REDRAWS = a.redraws
    t0 = time.time()
    with Pool(a.procs) as p:
        chunks = p.map(one_rep, range(SEED0, SEED0 + a.reps), chunksize=5)
    df = pd.DataFrame([r for c in chunks for r in c])
    df.to_csv(a.out, index=False)
    print(f"{a.reps} repliche x {a.redraws} ripetizioni in {time.time() - t0:.0f}s -> {a.out}")
    for m, g in df.groupby("method"):
        per_rep = g.groupby("rep").reject.mean()          # errore standard a cluster (per replica)
        first = g[g.redraw == 0].reject                    # una sola fase 2 per replica, come nel pilota
        print(f"{m:8s} rifiuto {per_rep.mean():.4f} (se {per_rep.std(ddof=1) / np.sqrt(len(per_rep)):.4f}, "
              f"{len(g)} test)  |  solo prima estrazione {first.mean():.3f} ({int(first.sum())}/{len(first)})"
              f"  |  copertura IC {g.covered.mean():.4f}")
