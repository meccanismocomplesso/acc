"""
dev_final_test.py - Il test finale comune (OLS con covariate, unilaterale) e' calibrato?
Analisi a posteriori della valutazione (H1): nello scenario nullo diversi metodi, oracolo B5 incluso, rifiutano
un po' sopra il 5%. Qui si isola il test finale: campione casuale di n unita', assegnazione, esito sotto H0
(mu = x1 + 0,5 x2 + rumore N(0, 1)), test a 5%. Popolazioni di sviluppo (semi 3.700-3.899), non di valutazione.
"""
import sys
from pathlib import Path

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402

REPS_PER_POP = 100


def run(seed):
    rng = np.random.default_rng([seed, 5])
    X = rng.standard_normal((33_333, 5))
    out = []
    for k in range(REPS_PER_POP):
        for n in (400, 600):
            S = rng.choice(len(X), n, replace=False)
            Y = X[S, 0] + 0.5 * X[S, 1] + rng.standard_normal(n)
            for assign in ("anticluster", "jitter", "casuale"):
                if assign == "anticluster":
                    T = acc.anticluster(X[S], 2, rng)
                elif assign == "jitter":
                    T = acc.anticluster(X[S], 2, rng, jitter=0.05)
                else:
                    T = rng.permutation(np.arange(n) % 2)
                for se in ("hc1", "hc3"):
                    out.append((n, assign, se, acc.ols_effect(X[S], T, Y, se_type=se)["reject"]))
    return out


if __name__ == "__main__":
    import pandas as pd
    with Pool(2) as p:
        rows = [r for c in p.map(run, range(3_700, 3_900)) for r in c]
    df = pd.DataFrame(rows, columns=["n", "assign", "se", "reject"])
    df.to_csv("results/dev_final_test_results.csv", index=False)
    g = df.groupby(["n", "assign", "se"]).reject.agg(["mean", "size"])
    g["e.s."] = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    print(g.round(4).to_string())
    print("\ncomplessivo:", round(df.reject.mean(), 4), "su", len(df), "test")
