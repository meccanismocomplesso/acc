# acc — Anticlustering-Clustering for focused experiments on large populations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24051299.svg)](https://doi.org/10.5281/zenodo.24051299)

Version **0.3.0** · November 20, 2026 · author: Fabio Nelli · research prototype.

**What's new in 0.3.0** (details in `CHANGELOG.md`, full research log in `docs/registro_tentativi.md`, summary in `docs/RISULTATI_V03.md`):

- Recommended configuration `acc.pipeline_v03`: the v0.2 adaptive phase 1 with **60% of the budget for exploration**
  (40% in v0.2) and **conditional stratification** of rare groups in the covariates.
- Evaluated on reserved seeds with a pre-registered protocol: higher power than v0.2 (0.75 vs. 0.71), type I error
  confirmed at 4.6% on 5,000 null replications, lower worst-case regret than pure strategies.
- All library defaults are unchanged: every function returns exactly what it returned in v0.2 unless a new option is set.

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

In 0.3.0 the recommended pipeline adds a step before phase 1: if a covariate-only diagnostic (`outer_ratio`) detects
extreme values or isolated groups, the outer 2% of the population (`outer_stratum`) is set aside, studied with its own
small experiment, and the two results are recombined with a Bonferroni split of alpha.

## Structure

```
acc/core.py                     method functions (numpy, scipy, scikit-learn only)
experiments/pilot1.py, pilot2.py   v0.2 pilots
experiments/dev_*.py, study_*.py   development studies on development seeds (< 10,000), see the research log
experiments/eval_central.py        evaluation of the central design of the plan on reserved seeds (>= 10,000)
experiments/eval_null_confirm.py   pre-registered null confirmation (5,000 replications)
experiments/eval_regret60.py       pre-registered worst-case regret study at equal exploration budget
experiments/analyze*.py            result summaries and hypothesis checks
results/                           raw and summary results of all studies
docs/PIANO_SIMULAZIONE.md          simulation plan, frozen at v0.2 (in Italian)
docs/registro_tentativi.md         research log of every attempt, positive and negative (in Italian)
docs/RISULTATI_V03.md              summary of the 0.3.0 results and programme for 0.4 (in Italian)
tests/test_core.py                 tests
```

## Installation and usage

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, pandas
python3 -m pytest -q                     # 19 tests, about 25 s

# reproduce the pilots (2 processes; about 12 and 14 minutes on 2 cores)
export OMP_NUM_THREADS=1
python3 experiments/pilot1.py --reps 60 --out results/pilot1_results.csv
python3 experiments/analyze1.py results/pilot1_results.csv
python3 experiments/pilot2.py --reps 60 --out results/pilot2_results.csv
python3 experiments/analyze2.py results/pilot2_results.csv
```

CSV files are written in append mode: delete the file before re-running.
With the same seeds, results are reproduced exactly.

Reproduce the 0.3.0 evaluation (2 cores; about 6 h, 2.5 h and 2.5 h). The scripts resume where they stopped:

```bash
export OMP_NUM_THREADS=1
python3 experiments/eval_central.py      && python3 experiments/analyze_eval.py
python3 experiments/eval_null_confirm.py && python3 experiments/analyze_null_confirm.py
python3 experiments/eval_regret60.py     && python3 experiments/analyze_regret60.py
```

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

New in 0.3.0 (all optional; defaults reproduce v0.2):

| Function / option | Role |
|---|---|
| `pipeline_v03(X, measure, B, rng)`, `V03_CONFIG` | **recommended configuration**: phases 1–2 with n1/B = 0.6 and conditional stratification |
| `outer_ratio(X)`, `outer_stratum(X, q)` | covariate-only diagnostic and outer stratum (robust distance) |
| `anticluster(..., jitter=0.05)` | randomized assignment: with `jitter = 0` (default) the partition is a deterministic function of X |
| `ols_effect(..., se_type="hc3")` | HC3 standard errors, used for the outer stratum |
| `adaptive_phase1(..., rule=..., focus=...)` | alternative attribution rules and focused sampling (exploratory; none beats v0.2) |
| `heterogeneity_t`, `sequential_phase1`, `select_projected`, `zoom_phase1` | exploratory tools from the development studies |

Using the recommended configuration:

```python
import numpy as np, acc
rng = np.random.default_rng(0)
X = rng.standard_normal((100_000, 5))
learn = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)      # phase 0
measure = lambda idx, T: ...                                 # your experiment on X[learn][idx]: returns Y
out = acc.pipeline_v03(X[learn], measure, 1_000, rng)
out["reject"], out["stratified"], [t["res"] for t in out["tests"]]
```

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

## Results of 0.3.0 on reserved seeds

Central design of the plan (`docs/PIANO_SIMULAZIONE.md`): d = 5, Gaussian covariates, N = 10^5, B = 1,000,
12 cells with an effect (300 replications each) plus the null (1,000). Seeds >= 10,000, never used during development.

| Method | Mean power (12 cells) |
|---|---|
| B1, single unfocused experiment | 0.52 |
| B4, classical enrichment on median splits | 0.50 |
| v0.2 (adaptive phase 1, n1/B = 0.4) | 0.71 |
| random phase 1 with n1/B = 0.6 | 0.74 |
| **`pipeline_v03`** | **0.75** |
| B5, sampling oracle | 1.00 |

- Paired difference `pipeline_v03` − v0.2: +0.046 (s.e. 0.008).
- Type I error: 6.2% on the first 1,000 null replications (above the plan's H1 threshold); pre-registered confirmation on
  5,000 new reserved seeds: **4.58%** (95% CI 4.02–5.20%), H1 met. Both results are reported.
- Worst-case regret at **equal** exploration budget (all strategies with n1/B = 0.6; oracle = best fixed strategy per cell):
  `pipeline_v03` 0.18 (95% CI 0.14–0.25), fixed 50/50 mix 0.19, random 0.63, coverage 0.63. The adaptive rule gives the
  same protection as a fixed 50/50 mix; it does not add to it.
- Coverage of confidence intervals: 95–96% on average. Plan hypotheses: H1 met (with confirmation), H4 met; H2 and H3 not
  met as written (criteria too strict for 300 replications, ties at 100% power), see the research log, point 15.

## Pilot results of v0.2 (d = 5, N = 10^5, B = 1,000, 60 replications per cell)

- Phase 0: maximum |SMD| across parts 0.000007 vs. 0.008 for a random split; 0.28 s for 10^5 rows
  (3.5 s for 10^6 rows with d = 20).
- With τ = 3, focusing raises power from 70% to 98% (dense region) and from 8% to 100% (tail of one covariate).
- The best λ depends on where the effect lies. The adaptive phase 1 recovers about 85% of the power lost
  with the wrong λ, and has the best worst case: 47% power vs. 27–35% for fixed λ.

## Known limitations (0.3.0)

- In Gaussian populations `pipeline_v03` is v0.2 with a better budget split: the gain in power comes from n1/B = 0.6.
- The adaptive attribution rule behaves like a fixed 50/50 mix (no candidate rule improves on it).
- Conditional stratification was validated only on development seeds (heavy tails, isolated 1% group); it never
  activates in the central design and has not yet been evaluated on reserved seeds.
- `pipeline_v03` uses the default deterministic anticlustering assignment; `jitter=0.05` (true randomization) is available
  but not part of the frozen configuration.
- Small rare region in the effect (1%): no method, including competitors, exceeds about 55% power with B = 1,000.
- Dense regions: a random phase 1 is better than any coverage-based phase 1.
- Evaluated only with d = 5; competitor methods implemented in simplified form (`experiments/study_competitors.py`).
- Limitations of v0.2 resolved: the 8.3% null rejection of pilot 2 was chance (4.86% on 10,000 tests).

## References

Papenberg & Klau (Psychological Methods), R package `anticlust`; Baumann et al. (2026), ABA, arXiv:2601.06351;
Thompson (1990), adaptive cluster sampling; literature on adaptive enrichment designs.
Competitors in 0.3.0: Chernozhukov, Demirer, Duflo & Fernández-Val (2018), GATES; Foster, Taylor & Ruberg (2011),
virtual twins, Statistics in Medicine; Simon & Simon (2013), adaptive enrichment designs, Biometrics.

## How to cite

See `CITATION.cff` or use the DOI: [10.5281/zenodo.24051299](https://doi.org/10.5281/zenodo.24051299).

## License

MIT, see `LICENSE`.
