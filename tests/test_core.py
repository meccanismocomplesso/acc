"""Test di base per acc.core. Esecuzione: python3 -m pytest -q (dalla radice)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acc  # noqa: E402


def _X(n=3000, d=5, seed=0):
    return np.random.default_rng(seed).standard_normal((n, d))


def test_anticluster_sizes_and_balance():
    X = _X()
    lab = acc.anticluster(X, 3, np.random.default_rng(1))
    sizes = np.bincount(lab)
    assert sizes.max() - sizes.min() <= 1
    rng = np.random.default_rng(2)
    rand = rng.permutation(np.arange(len(X)) % 3)
    assert acc.balance_report(X, lab)["max_abs_smd"] < acc.balance_report(X, rand)["max_abs_smd"]


def test_anticluster_odd_n():
    X = _X(n=101)
    lab = acc.anticluster(X, 2, np.random.default_rng(0))
    assert sorted(np.bincount(lab)) == [50, 51]


def test_select_tempered_size_and_unique():
    Z = acc.standardize(_X())
    rng = np.random.default_rng(0)
    for lam in (0.0, 0.5, 1.0):
        S = acc.select_tempered(Z, 100, lam, rng)
        assert len(S) == 100 and len(np.unique(S)) == 100


def test_tempered_lambda_moves_to_tails():
    Z = acc.standardize(_X(n=20000))
    rng = np.random.default_rng(0)
    r0 = np.linalg.norm(Z[acc.select_tempered(Z, 200, 0.0, rng)], axis=1).mean()
    r1 = np.linalg.norm(Z[acc.select_tempered(Z, 200, 1.0, rng)], axis=1).mean()
    assert r1 > r0


def test_population_diagnostic_gaussian_is_neutral():
    d = acc.population_diagnostic(acc.standardize(_X(n=50000)))
    assert 0.4 < d["lam0"] < 0.6


def test_ols_effect_recovers_effect():
    rng = np.random.default_rng(0)
    X = _X(n=4000)
    T = acc.anticluster(X, 2, rng)
    Y = X[:, 0] + 2.0 * T + rng.standard_normal(len(X))
    res = acc.ols_effect(X, T, Y)
    assert res["lo"] < 2.0 < res["hi"] and res["reject"]


def test_adaptive_phase1_budget():
    X = _X(n=5000)
    Z = acc.standardize(X)
    rng = np.random.default_rng(0)
    measure = lambda idx, T: X[idx, 0] + rng.standard_normal(len(idx))
    S, T, Y, A, shares = acc.adaptive_phase1(Z, X, measure, 200, rng, rounds=4)
    assert len(S) == 200 and len(np.unique(S)) == 200 and len(shares) == 4
    assert all(0.15 <= s <= 0.85 for s in shares[1:])
