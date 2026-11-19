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


def test_anticluster_default_is_deterministic_up_to_labels():
    # v0.2: con jitter = 0 due esecuzioni danno la stessa partizione (a meno delle etichette)
    X = _X(n=600)
    a = acc.anticluster(X, 2, np.random.default_rng(1))
    b = acc.anticluster(X, 2, np.random.default_rng(2))
    assert max((a == b).mean(), (a != b).mean()) == 1.0


def test_anticluster_jitter_randomizes_and_keeps_balance():
    X = _X(n=600)
    a = acc.anticluster(X, 2, np.random.default_rng(1), jitter=0.05)
    b = acc.anticluster(X, 2, np.random.default_rng(2), jitter=0.05)
    assert max((a == b).mean(), (a != b).mean()) < 0.7          # partizioni davvero diverse
    assert sorted(np.bincount(a)) == [300, 300]
    rand = np.random.default_rng(3).permutation(np.arange(600) % 2)
    for lab in (a, b):
        assert acc.balance_report(X, lab)["max_abs_smd"] < 0.2 * acc.balance_report(X, rand)["max_abs_smd"]


def _adaptive(rule="v02", focus=0.0, n=5000, n1=200):
    X = _X(n=n)
    Z = acc.standardize(X)
    rng = np.random.default_rng(0)
    measure = lambda idx, T: X[idx, 0] + 2.0 * T * (X[idx, 1] > 1) + rng.standard_normal(len(idx))
    return acc.adaptive_phase1(Z, X, measure, n1, rng, rounds=4, rule=rule, focus=focus)


def test_adaptive_phase1_rules_respect_budget():
    for rule in acc.ATTRIBUTION_RULES:
        S, T, Y, A, shares = _adaptive(rule)
        assert len(S) == 200 and len(np.unique(S)) == 200 and len(shares) == 4
        assert all(0.15 <= s <= 0.85 for s in shares[1:])
        assert set(np.unique(A)) <= {0, 1}
    S, T, Y, A, shares = _adaptive("none")
    assert len(set(shares)) == 1                                # senza attribuzione la quota non cambia


def test_adaptive_phase1_focus():
    S, T, Y, A, shares = _adaptive("v02", focus=0.5)
    assert len(S) == 200 and len(np.unique(S)) == 200
    assert (A[:50] < 2).all()                                   # primo ciclo: solo esplorazione
    assert (A[50:] == 2).sum() == 75                            # poi meta' di ogni ciclo da 50
    assert abs(T.mean() - 0.5) < 0.02


def test_adaptive_phase1_unknown_rule():
    import pytest
    with pytest.raises(ValueError):
        _adaptive("boh")


def test_select_projected_size_and_unique():
    Z = acc.standardize(_X(n=6000))
    S = acc.select_projected(Z, 100, np.random.default_rng(0), 2)
    assert len(S) == 100 and len(np.unique(S)) == 100


def test_zoom_phase1_budget_and_narrowing():
    X = _X(n=6000)
    Z = acc.standardize(X)
    rng = np.random.default_rng(0)
    measure = lambda idx, T: X[idx, 0] + 2.0 * T * (X[idx, 1] > 1) + rng.standard_normal(len(idx))
    S, T, Y, hist = acc.zoom_phase1(Z, X, measure, 200, rng, K=8, rounds=4)
    assert len(S) == 200 and len(np.unique(S)) == 200 and len(hist) == 4
    assert abs(T.mean() - 0.5) < 0.02


def test_sequential_phase1_budget_and_stop():
    X = _X(n=6000)
    Z = acc.standardize(X)
    rng = np.random.default_rng(0)
    measure = lambda idx, T: X[idx, 0] + rng.standard_normal(len(idx))
    S, T, Y, hist = acc.sequential_phase1(Z, X, measure, rng, round_size=50, max_n=200,
                                          threshold=np.inf, seed=0)
    assert len(S) == 200 and len(np.unique(S)) == 200 and len(hist) == 4
    assert [h["n"] for h in hist] == [50, 100, 150, 200]
    S2, _, _, h2 = acc.sequential_phase1(Z, X, measure, np.random.default_rng(0), round_size=50,
                                         max_n=200, min_rounds=1, threshold=-np.inf, seed=0)
    assert len(S2) == 50 and len(h2) == 1                    # soglia sempre superata: si ferma subito


def test_heterogeneity_t_finite():
    X = _X(n=400)
    rng = np.random.default_rng(1)
    T = acc.anticluster(X, 2, rng)
    Y = X[:, 0] + 3.0 * T * (X[:, 1] > 0.5) + rng.standard_normal(400)
    assert np.isfinite(acc.heterogeneity_t(X, T, Y, seed=0))


def test_outer_stratum_and_ratio():
    X = _X(n=20000)
    O = acc.outer_stratum(X, 0.02)
    assert abs(O.mean() - 0.02) < 0.002
    assert abs(acc.outer_ratio(X)["relative"] - 1.0) < 0.05          # normale: nessun segnale
    Y = X.copy()
    Y[:200] = 6.0 + 0.3 * Y[:200]                                      # 1% di gruppo isolato
    assert acc.outer_ratio(Y)["relative"] > 1.15
    assert acc.outer_stratum(Y, 0.02)[:200].mean() > 0.95


def test_ols_effect_hc3_matches_reference():
    import sys
    sys.path.insert(0, "experiments")
    from dev_null_strat import ols
    rng = np.random.default_rng(5)
    X = rng.standard_t(3, (160, 5))
    T = acc.anticluster(X, 2, rng)
    Y = X[:, 0] + rng.standard_normal(160)
    for kind in ("hc1", "hc3"):
        r = acc.ols_effect(X, T, Y, se_type=kind)
        assert abs(r["est"] / r["se"] - ols(X, T, Y, kind)) < 1e-8
    assert acc.ols_effect(X, T, Y, se_type="hc3")["se"] >= acc.ols_effect(X, T, Y)["se"] * 0.9


def test_pipeline_v03_budget_and_branches():
    rng = np.random.default_rng(0)
    X = _X(n=20000)
    used = []

    def measure(idx, T):
        used.append(np.asarray(idx))
        return X[idx, 0] + rng.standard_normal(len(idx))

    out = acc.pipeline_v03(X, measure, 500, rng, seed=0)
    assert not out["stratified"] and len(out["tests"]) == 1
    u = np.concatenate(used)
    assert len(u) == 500 and len(np.unique(u)) == 500
    Y = X.copy()
    Y[:200] = 6.0 + 0.3 * Y[:200]
    used.clear()

    def measure2(idx, T):
        used.append(np.asarray(idx))
        return Y[idx, 0] + rng.standard_normal(len(idx))

    out = acc.pipeline_v03(Y, measure2, 500, rng, seed=0)
    assert out["stratified"] and len(out["tests"]) == 2
    assert all(t["alpha"] == 0.025 for t in out["tests"])
    u = np.concatenate(used)
    assert len(u) == 500 and len(np.unique(u)) == 500
