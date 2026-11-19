"""
study_seqlen.py - Fase 1 a lunghezza variabile contro ripartizioni fisse.

Per ogni replica si genera una sola sequenza di fase 1 fino a 800 unita' (`acc.sequential_phase1`,
cicli da 100: 50 copertura + 50 casuali) registrando dopo ogni ciclo la statistica di eterogeneita'.
Poiche' il campionamento non dipende dall'arresto, ogni regola corrisponde a un prefisso della sequenza:
per ogni prefisso n1 = 300, ..., 800 si esegue la fase 2 con le B - n1 unita' restanti (pilot2.phase2).
Regole confrontate:
  FIX_n     fase 1 fissa a n unita' (300 ... 800)
  SIG_c     arresto al primo ciclo >= 3 con statistica t >= c (c = 2, 2.5, 3), altrimenti 800
Popolazione gaussiana, scenari e esito del pilota 2. Semi 6.000+ (6.900+ usati per la calibrazione).
Uso:  python3 experiments/study_seqlen.py --reps 40 --out results/study_seqlen_results.csv
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
import pilot2
from pilot1 import B, SCENARIOS, make_population, outcome, region_mask

PREFIXES = [300, 400, 500, 600, 700, 800]
THRESHOLDS = [2.0, 2.5, 3.0]
SEED0 = 6_000


def one_rep(args):
    scenario, tau, rep = args
    base = [SCENARIOS.index(scenario), int(tau * 10), rep]
    rng = np.random.default_rng(base)
    X = make_population(rep)
    R = region_mask(X, scenario)
    L = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)
    XL, RL = X[L], R[L]
    ZL = acc.standardize(XL)
    rk = acc.knn_radius(ZL, rng)
    measure = lambda idx, T: outcome(XL[idx], T, RL[idx], tau, rng)
    S, T, Y, hist = acc.sequential_phase1(ZL, XL, measure, rng, round_size=100, max_n=800,
                                          threshold=np.inf, rk=rk, seed=rep)
    tstats = {h["n"]: h["tstat"] for h in hist}
    rows = []
    for n1 in PREFIXES:
        pilot2.N2 = B - n1
        rec = dict(scenario=scenario, tau=tau, rep=rep, n1=n1, tstat=tstats[n1],
                   s1_in_region=int(RL[S[:n1]].sum()))
        rec.update(pilot2.phase2(XL, RL, S[:n1], T[:n1], Y[:n1], tau,
                                 np.random.default_rng(base + [n1]), rep))
        rows.append(rec)
    return rows


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Da una riga per prefisso a una riga per regola."""
    out = []
    for key, g in df.groupby(["scenario", "tau", "rep"]):
        g = g.set_index("n1").sort_index()
        for n1 in PREFIXES:
            out.append(dict(zip(["scenario", "tau", "rep"], key), rule=f"FIX_{n1}", stop=n1,
                            **g.loc[n1, ["reject", "precision", "covered", "s1_in_region"]].to_dict()))
        for c in THRESHOLDS:
            hit = [n for n in PREFIXES if g.loc[n, "tstat"] >= c]
            stop = hit[0] if hit else PREFIXES[-1]
            out.append(dict(zip(["scenario", "tau", "rep"], key), rule=f"SIG_{c}", stop=stop,
                            **g.loc[stop, ["reject", "precision", "covered", "s1_in_region"]].to_dict()))
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="results/study_seqlen_results.csv")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    cells = [("nullo", 0.0)] + [(s, t) for s in SCENARIOS[1:] for t in (1.5, 3.0)]
    jobs = [(s, t, SEED0 + r) for r in range(a.reps) for s, t in cells]
    t0, done = time.time(), 0
    with Pool(a.procs) as p:
        for chunk in p.imap_unordered(one_rep, jobs):
            pd.DataFrame(chunk).to_csv(a.out, mode="a", index=False, header=(done == 0))
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(jobs)} job, {time.time() - t0:.0f}s", flush=True)
    print(f"{done} job in {time.time() - t0:.0f}s -> {a.out}", flush=True)
