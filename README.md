# acc — Anticlustering-Clustering for focused experiments on large populations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24051299.svg)](https://doi.org/10.5281/zenodo.24051299)

Version **0.2.0** · November 7, 2026 · author: Fabio Nelli · research prototype.

`acc` implements a multi-phase experimental design for finding and measuring heterogeneous
effects in very large populations on a limited budget:

1. **Phase 0, representative split**: anticlustering splits the population into learning, test and
   holdout sets. Each part is a smaller copy of the population.
2. **Diagnostic**: covariates alone are used to estimate how heavy the tails are, which gives a starting λ0.
3. **Phase 1, adaptive exploration**: S1 is drawn in rounds, mixing representative selection and
   coverage-oriented selection (tempered density). The mix is updated according to where the signal appears.
   Treatment/control arms are balanced with anticlustering.
4. **Phase 2, focusing**: τ(x) is estimated, the region R̂ with the highest τ̂ is selected, and a new
   experiment is run on fresh units from R̂. Inference uses only these units, so it remains valid after selection.

## Structure

```
acc/core.py              method functions (numpy, scipy, scikit-learn only)
experiments/pilot1.py    pilot 1: fixed λ vs. unfocused experiment
experiments/pilot2.py    pilot 2: adaptive phase 1 vs. fixed λ and 50/50 mix
experiments/analyze*.py  result summaries
results/                 raw and summary results of both pilots
docs/PIANO_SIMULAZIONE.md  plan for the full simulation study, frozen at v0.2 (in Italian)
tests/test_core.py       basic tests
```

## Installation and usage

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, pandas
python3 -m pytest -q                     # 7 tests, about 3 s

# reproduce the pilots (2 processes; about 12 and 14 minutes on 2 cores)
export OMP_NUM_THREADS=1
python3 experiments/pilot1.py --reps 60 --out results/pilot1_results.csv
python3 experiments/analyze1.py results/pilot1_results.csv
python3 experiments/pilot2.py --reps 60 --out results/pilot2_results.csv
python3 experiments/analyze2.py results/pilot2_results.csv
```

CSV files are written in append mode: delete the file before re-running.
With the same seeds, results are reproduced exactly.

## Main API (`import acc`)

| Function | Role |
|---|---|
| `anticluster(X, K, rng)` | partition into K mutually similar groups (assignment-based heuristic, ABA-style) |
| `balance_report(X, labels)` | maximum \|SMD\| and log variance ratio across groups |
| `population_diagnostic(Z)` | tail ratio, kurtosis, λ0 |
| `select_tempered(Z, n, lam, rng)` | S1 representing q(x) ∝ p(x)^(1−λ) |
| `adaptive_phase1(Z, X, measure, n1, rng, lam0)` | multi-round phase 1 with adaptive coverage share |
| `fit_cate(X, T, Y)` | τ̂(x) via doubly robust pseudo-outcome (or T-learner) |
| `ols_effect(X, T, Y)` | covariate-adjusted effect, HC1 standard errors, one-sided test |
| `select_kcenter`, `select_hybrid`, `coverage_radius`, `mmd` | v0.1 variants and diagnostics |

Minimal example:

```python
import numpy as np, acc
rng = np.random.default_rng(0)
X = rng.standard_normal((100_000, 5))
learn = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)      # phase 0
Z = acc.standardize(X[learn])
lam0 = acc.population_diagnostic(Z)["lam0"]                  # diagnostic
measure = lambda idx, T: ...                                 # your experiment: returns Y
S1, T1, Y1, arm, shares = acc.adaptive_phase1(Z, X[learn], measure, 400, rng, lam0=lam0)
tau_hat = acc.fit_cate(X[learn][S1], T1, Y1)(X[learn])       # phase 2: region R̂ = top 3% of tau_hat
```

## Pilot results (d = 5, N = 10^5, B = 1,000, 60 replications per cell)

- Phase 0: maximum |SMD| across parts 0.000007 vs. 0.008 for a random split; 0.28 s for 10^5 rows
  (3.5 s for 10^6 rows with d = 20).
- With τ = 3, focusing raises power from 70% to 98% (dense region) and from 8% to 100% (tail of one covariate).
- The best λ depends on where the effect lies. The adaptive phase 1 recovers about 85% of the power lost
  with the wrong λ, and has the best worst case: 47% power vs. 27–35% for fixed λ.

## Known limitations (v0.2)

- Null scenario: the adaptive method rejects in 8.3% of cases (5/60). Not significant, but needs checking.
- The attribution rule favours the random arm: in the tail scenario the coverage share does not increase.
- Small rare region (1%): no method exceeds 50% power.
- Tested only with d = 5 and Gaussian covariates.

## References

Papenberg & Klau (Psychological Methods), R package `anticlust`; Baumann et al. (2026), ABA, arXiv:2601.06351;
Thompson (1990), adaptive cluster sampling; literature on adaptive enrichment designs.

## How to cite

See `CITATION.cff` or use the DOI: [10.5281/zenodo.24051299](https://doi.org/10.5281/zenodo.24051299).

## License

MIT, see `LICENSE`.
