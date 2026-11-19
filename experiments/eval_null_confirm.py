"""
eval_null_confirm.py - Nullo di conferma per H1 (registro, punto 15c). Preregistrato prima dell'esecuzione.

Motivo: nella valutazione del disegno centrale ACC_v03 ha rifiutato 62 volte su 1.000 nello scenario nullo (6,2%,
limite superiore IC 95% 7,9%), e non ha superato il criterio H1 del piano (limite superiore <= 7%).

Disegno: scenario nullo del disegno centrale (d = 5, covariate gaussiane, N = 10^5, B = 1.000), 5.000 repliche
indipendenti, una popolazione nuova per replica. Semi 30.000-34.999: riservati (>= 10.000), mai usati prima
(la valutazione ha usato 10.000-22.999). Codice congelato invariato (tag v0.3-dev-frozen), stesse funzioni di
eval_central.py.
Metodi: ACC_v03 (oggetto della verifica); B5 (oracolo: esperimento casuale con lo stesso test finale, per vedere
se un eventuale eccesso e' comune); B1 e P_rand (riferimenti economici).

Criteri, fissati prima di vedere i risultati:
  C1 (piano, H1)   limite superiore Clopper-Pearson al 95% del tasso di rifiuto <= 7%.
  C2 (aggiuntivo)  test binomiale bilaterale contro il 5%, alpha = 0,05. Con 5.000 repliche rileva uno scostamento
                   di circa 0,9 punti con potenza dell'80%.
ACC_v03 si considera valido se supera C1. C2 si riporta per tutti i metodi come informazione sulla calibrazione:
se lo fallisce, si indica se l'eccesso e' presente anche in B5 (test finale comune) oppure solo in ACC_v03.

Riprendibile. Uso: python3 experiments/eval_null_confirm.py --out results/eval_null_confirm_results.csv
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
from eval_central import B, N1, RHO, ate_test, confirm, focus, outcome, population, weights

SEED0, REPS = 30_000, 5_000
METHODS = ["B1", "P_rand", "ACC_v03", "B5"]


def one_rep(r):
    seed = SEED0 + r
    rng0 = np.random.default_rng([seed, 0])
    X = population(seed)
    Wall, Rall = weights(X, "nullo")
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, W, R = X[L], Wall[L], Rall[L]
    rows = []
    for mi, m in enumerate(METHODS):
        rng = np.random.default_rng([seed, mi + 1])
        meas = lambda idx, T: outcome(XL[idx], T, W[idx], 0.0, rng)
        t0 = time.time()
        if m == "B1":
            res = ate_test(XL, W, rng.choice(len(XL), B, replace=False), 0.0, rng)[0]
            rej = res["reject"]
        elif m == "P_rand":
            S1 = rng.choice(len(XL), N1, replace=False)
            T1 = acc.anticluster(XL[S1], 2, rng)
            res = focus(XL, W, R, S1, T1, meas(S1, T1), B - N1, 0.0, rng, seed)[0]
            rej = res["reject"]
        elif m == "B5":
            res = confirm(XL, W, R, np.ones(len(XL), bool), np.array([], int), B - N1, 0.0, rng)[0]
            rej = res["reject"]
        else:
            out = acc.pipeline_v03(XL, meas, B, rng, seed=seed)
            res, rej = out["tests"][0]["res"], out["reject"]
        rows.append(dict(rep=r, seed=seed, method=m, reject=bool(rej), z=res["est"] / res["se"],
                         seconds=time.time() - t0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eval_null_confirm_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    todo = list(range(REPS))
    if Path(a.out).exists():
        prev = pd.read_csv(a.out, usecols=["rep", "method"])
        cnt = prev.groupby("rep").method.nunique()
        done = set(cnt[cnt == len(METHODS)].index)
        todo = [r for r in todo if r not in done]
    print(f"{len(todo)} repliche da eseguire", flush=True)
    t0, k = time.time(), 0
    header = not Path(a.out).exists()
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, todo, chunksize=4):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=header)
            header = False
            k += 1
            if k % 250 == 0:
                el = time.time() - t0
                print(f"{k}/{len(todo)}, {el:.0f}s, mancano circa {el / k * (len(todo) - k) / 60:.0f} min", flush=True)
    print(f"{k} repliche in {time.time() - t0:.0f}s -> {a.out}", flush=True)
