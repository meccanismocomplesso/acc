"""
eval_regret60.py - Regret massimo a parita' di budget di esplorazione (n1/B = 0,6 per tutti).
Preregistrato prima dell'esecuzione (tag v0.3-dev-regret60).

Domanda: scegliendo ACC prima di sapere se e dove sia l'effetto, quanta potenza si rischia di perdere rispetto
alla strategia che si sarebbe scelta conoscendone la posizione? Nel punto 15 del registro il regret di ACC_v03
(n1 = 600) era calcolato rispetto a strategie con n1 = 400; qui tutte hanno n1 = 600.

Disegno: le 12 celle con effetto del disegno centrale (omogeneo, compatta_5, palla_rara_1, coda_1, multi_2x2,
gradiente; tau = 1,5 e 3), d = 5, covariate gaussiane, N = 10^5, B = 1.000, 300 repliche per cella.
Semi riservati nuovi: 40.000 + 1.000 * c + r (c = indice della cella in eval_central.CELLS, 1..12; r < 300),
cioe' 41.000-52.299, mai usati prima. Stessa popolazione per tutti i metodi di una replica (confronti appaiati).
Metodi, tutti con fase 1 di 600 unita' e fase 2 di 400 sul top 3% di tau_hat (stesse funzioni di eval_central.py):
  P_rand_60   fase 1 casuale
  P_l1_60     fase 1 densita' temperata lambda = 1 (copertura)
  P_mix_60    fase 1 50% casuale + 50% lambda = 1
  ACC_v03     acc.pipeline_v03, configurazione congelata (in popolazioni gaussiane non stratifica)

Analisi preregistrata (analyze_regret60.py):
  Oracolo per cella O = max(P_rand_60, P_l1_60, P_mix_60): la migliore strategia fissa, scelta conoscendo la geometria.
  Regret di un metodo in una cella = O - potenza del metodo; regret massimo = massimo sulle 12 celle.
  Incertezza: bootstrap appaiato (5.000 ricampionamenti delle repliche dentro ogni cella, tutti i metodi insieme),
  IC 95% percentili per il regret massimo di ogni metodo e per le differenze ACC_v03 - alternativa.
  Affermazione principale: "ACC_v03 ha regret massimo minore di P_rand_60, P_l1_60 e P_mix_60" e' sostenuta se,
  per ciascuna delle tre, il limite superiore dell'IC 95% della differenza e' < 0.
  Analisi secondarie: oracolo che include anche ACC_v03; regret medio; regret per cella; versione con oracolo
  scelto su meta' delle repliche e regret misurato sull'altra meta' (riduce la distorsione del massimo di stime).
Limite noto: solo popolazioni gaussiane (disegno centrale); la stratificazione di ACC non e' messa alla prova qui.

Riprendibile. Uso: python3 experiments/eval_regret60.py --out results/eval_regret60_results.csv
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
from eval_central import B, CELLS, focus, outcome, population, weights

SEED0, REPS, N1 = 40_000, 300, 600
METHODS = ["P_rand_60", "P_l1_60", "P_mix_60", "ACC_v03"]
CELL_IDS = list(range(1, len(CELLS)))          # 12 celle con effetto


def one_rep(args):
    ci, r = args
    geom, tau = CELLS[ci]
    seed = SEED0 + 1_000 * ci + r
    rng0 = np.random.default_rng([seed, 0])
    X = population(seed)
    Wall, Rall = weights(X, geom)
    L = np.flatnonzero(acc.anticluster(X, 3, rng0) == 0)
    XL, W, R = X[L], Wall[L], Rall[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng0)
    rows = []
    for mi, m in enumerate(METHODS):
        rng = np.random.default_rng([seed, mi + 1])
        meas = lambda idx, T: outcome(XL[idx], T, W[idx], tau, rng)
        t0 = time.time()
        if m == "ACC_v03":
            out = acc.pipeline_v03(XL, meas, B, rng, seed=seed)
            t = out["tests"][0]
            g = np.zeros(len(XL), bool); g[t["group"]] = True
            res, truth = t["res"], tau * W[g].mean()
            reject, prec = out["reject"], (g & R).sum() / g.sum()
            strat = out["stratified"]
        else:
            if m == "P_rand_60":
                S1 = rng.choice(len(XL), N1, replace=False)
            elif m == "P_l1_60":
                S1 = acc.select_tempered(ZL, N1, 1.0, rng, rk=rk)
            else:
                cov = acc.select_tempered(ZL, N1 // 2, 1.0, rng, rk=rk)
                rest = np.setdiff1d(np.arange(len(XL)), cov)
                S1 = np.concatenate([cov, rng.choice(rest, N1 - len(cov), replace=False)])
            T1 = acc.anticluster(XL[S1], 2, rng)
            res, truth, prec, _ = focus(XL, W, R, S1, T1, meas(S1, T1), B - N1, tau, rng, seed)
            reject, strat = res["reject"], False
        rows.append(dict(cell=ci, geom=geom, tau=tau, rep=r, seed=seed, method=m, reject=bool(reject),
                         correct=bool(reject and truth > 0), precision=prec, stratified=strat,
                         seconds=time.time() - t0))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/eval_regret60_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    jobs = [(ci, r) for r in range(REPS) for ci in CELL_IDS]
    if Path(a.out).exists():
        prev = pd.read_csv(a.out, usecols=["cell", "rep", "method"])
        cnt = prev.groupby(["cell", "rep"]).method.nunique()
        done = set(cnt[cnt == len(METHODS)].index)
        jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} job da eseguire", flush=True)
    t0, k = time.time(), 0
    header = not Path(a.out).exists()
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs, chunksize=2):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=header)
            header = False
            k += 1
            if k % 200 == 0:
                el = time.time() - t0
                print(f"{k}/{len(jobs)}, {el:.0f}s, mancano circa {el / k * (len(jobs) - k) / 60:.0f} min", flush=True)
    print(f"{k} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
