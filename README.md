# acc — Anticlustering-Clustering per la focalizzazione su grandi popolazioni

Versione **0.2.0** · 7 novembre 2026 · autore: Fabio Nelli · prototipo di ricerca, non ancora pubblicato.

`acc` implementa un disegno sperimentale in più fasi per trovare e misurare effetti eterogenei
in popolazioni molto grandi con un budget limitato:

1. **Fase 0, split rappresentativo**: l'anticlustering divide la popolazione in learning, test e holdout.
   Ogni parte è una copia in piccolo della popolazione.
2. **Diagnosi**: dalle sole covariate si stima quanto sono pesanti le code, e da qui un λ0 di partenza.
3. **Fase 1, esplorazione adattiva**: S1 si estrae in cicli, mescolando selezione rappresentativa e
   selezione a copertura (densità temperata). La quota si aggiorna in base a dove compare il segnale.
   I bracci trattamento/controllo si bilanciano con anticlustering.
4. **Fase 2, focalizzazione**: si stima τ(x), si prende la regione R̂ con τ̂ più alto e si fa un
   esperimento su unità nuove di R̂. L'inferenza usa solo queste unità, quindi resta valida dopo la selezione.

## Struttura

```
acc/core.py              funzioni del metodo (solo numpy, scipy, scikit-learn)
experiments/pilot1.py    pilota 1: λ fissi contro esperimento non focalizzato
experiments/pilot2.py    pilota 2: fase 1 adattiva contro λ fissi e mix 50/50
experiments/analyze*.py  riassunti dei risultati
results/                 risultati grezzi e riassunti dei due pilota
docs/PIANO_SIMULAZIONE.md  piano della simulazione completa, congelato alla v0.2
tests/test_core.py       test di base
```

## Installazione e uso

```bash
pip install -r requirements.txt          # numpy, scipy, scikit-learn, pandas
python3 -m pytest -q                     # 7 test, circa 3 s

# riprodurre i pilota (2 processi; circa 12 e 14 minuti su 2 core)
export OMP_NUM_THREADS=1
python3 experiments/pilot1.py --reps 60 --out results/pilot1_results.csv
python3 experiments/analyze1.py results/pilot1_results.csv
python3 experiments/pilot2.py --reps 60 --out results/pilot2_results.csv
python3 experiments/analyze2.py results/pilot2_results.csv
```

I CSV si scrivono in coda (append): cancellare il file prima di rilanciare.
Con gli stessi semi i risultati si riproducono esattamente.

## API principale (`import acc`)

| Funzione | Ruolo |
|---|---|
| `anticluster(X, K, rng)` | partizione in K gruppi simili tra loro (euristica assignment-based, stile ABA) |
| `balance_report(X, labels)` | \|SMD\| massimo e log-rapporto di varianze tra gruppi |
| `population_diagnostic(Z)` | tail ratio, curtosi, λ0 |
| `select_tempered(Z, n, lam, rng)` | S1 che rappresenta q(x) ∝ p(x)^(1−λ) |
| `adaptive_phase1(Z, X, measure, n1, rng, lam0)` | fase 1 a cicli con quota di copertura adattiva |
| `fit_cate(X, T, Y)` | τ̂(x) con pseudo-outcome doubly robust (o T-learner) |
| `ols_effect(X, T, Y)` | effetto aggiustato per covariate, SE HC1, test unilaterale |
| `select_kcenter`, `select_hybrid`, `coverage_radius`, `mmd` | varianti e diagnostica della v0.1 |

Esempio minimo:

```python
import numpy as np, acc
rng = np.random.default_rng(0)
X = rng.standard_normal((100_000, 5))
learn = np.flatnonzero(acc.anticluster(X, 3, rng) == 0)      # fase 0
Z = acc.standardize(X[learn])
lam0 = acc.population_diagnostic(Z)["lam0"]                  # diagnosi
measure = lambda idx, T: ...                                 # il vostro esperimento: restituisce Y
S1, T1, Y1, arm, shares = acc.adaptive_phase1(Z, X[learn], measure, 400, rng, lam0=lam0)
tau_hat = acc.fit_cate(X[learn][S1], T1, Y1)(X[learn])       # fase 2: regione R̂ = top 3% di tau_hat
```

## Risultati dei pilota (d = 5, N = 10^5, B = 1.000, 60 repliche per cella)

- Fase 0: |SMD| massimo tra le parti 0,000007 contro 0,008 dello split casuale; 0,28 s per 10^5 righe.
- Con τ = 3, la focalizzazione porta la potenza dal 70% al 98% (regione densa) e dall'8% al 100% (coda di una covariata).
- Il λ migliore dipende da dove sta l'effetto. La fase 1 adattiva recupera circa l'85% della potenza persa
  con un λ sbagliato. Ha anche il caso peggiore migliore: 47% di potenza contro il 27–35% dei λ fissi.

## Limiti noti (v0.2)

- Scenario nullo: il metodo adattivo rifiuta nell'8,3% dei casi (5/60). Non è significativo, ma va verificato.
- La regola di attribuzione favorisce il braccio casuale: nella coda la quota di copertura non sale.
- Piccola regione rara (1%): nessun metodo supera il 50% di potenza.
- Testato solo con d = 5 e covariate gaussiane.

## Riferimenti

Papenberg & Klau (Psychological Methods), pacchetto R `anticlust`; Baumann et al. (2026), ABA, arXiv:2601.06351;
Thompson (1990), adaptive cluster sampling; letteratura sugli adaptive enrichment design.
Bibliografia completa nel documento di ricerca.

## Licenza

MIT, vedi `LICENSE`.
