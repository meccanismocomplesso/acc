# Changelog

## 0.2.0 — 7 novembre 2026 (versione congelata)

- Aggiunta la diagnosi della popolazione (`population_diagnostic`) e la fase 1 adattiva a cicli (`adaptive_phase1`).
- Pilota 2: P_rand, P_l1, P_mix, P_adapt (`experiments/pilot2.py`).
- Codice riorganizzato in pacchetto (`acc/`), test di base, README.
- Congelato il piano della simulazione completa (`docs/PIANO_SIMULAZIONE.md`).
- Licenza MIT (`LICENSE`).

## 0.1.0 — 7 novembre 2026

- Prototipo: anticlustering assignment-based, selezione rappresentativa, k-center, ibrida, densità temperata.
- CATE con pseudo-outcome DR; regione R̂ per ranking (top 3%); inferenza solo su S2.
- Pilota 1 (`experiments/pilot1.py`).
- Modifiche nate dalla taratura, prima del pilota 1:
  - copertura tramite densità temperata invece del k-center, che sceglieva soprattutto outlier;
  - regione per ranking invece della soglia τ̂ > δ, che restituiva regioni vuote.
