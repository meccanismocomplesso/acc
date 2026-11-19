# Changelog

## 0.3-dev, congelata il 19 novembre 2026 (tag `v0.3-dev-frozen`)

Configurazione da valutare sui semi riservati, `acc.pipeline_v03` / `acc.V03_CONFIG`:
fase 1 adattiva v0.2 (regola invariata) con n1/B = 0,6; stratificazione condizionata (strato esterno 2%,
attiva se `outer_ratio` supera di oltre il 10% il valore atteso per una normale; strato con il 16% del budget,
test HC3 a alpha/2; corpo principale con fase 1 al 52% del budget, test a alpha/2); fase 2 invariata (top 3%, HC1).
Motivazioni e risultati sui semi di sviluppo: `acc_registro_tentativi.md`, punti 9, 12, 13, 14.

**Deviazioni dal piano congelato v0.2 (`docs/PIANO_SIMULAZIONE.md`)**, registrate prima di lanciare i semi di valutazione:
1. Sez. 6: nessuno dei tre candidati per la nuova regola di attribuzione batte la v0.2 (registro, punto 3).
   Il metodo `P_adapt_v03` del piano è quindi sostituito da `ACC_v03` = `pipeline_v03`.
2. `ACC_v03` usa n1/B = 0,6, mentre il disegno centrale del piano fissa 0,4 per tutti i metodi. Gli altri metodi
   restano a 0,4 come da piano; si aggiunge, fuori piano, `P_rand_60` (fase 1 casuale al 60%), il concorrente più forte
   nel punto 13 del registro, per separare l'effetto della ripartizione da quello del campionamento guidato.
3. Si esegue solo il disegno centrale (13 celle), non le 54 celle di variazione, per limiti di calcolo.
   Le geometrie `omogeneo`, `multi_2x2` e `gradiente` e i riferimenti B3, B4, B5 sono implementati per la prima volta
   in `experiments/eval_central.py`, secondo la descrizione del piano.
4. Errore standard HC3 (invece di HC1) solo per il test dello strato esterno, che nel disegno centrale gaussiano
   non si attiva.

## Non rilasciato — proposta di lavoro del 15 novembre 2026 (sviluppo verso la 0.3)

Nessun comportamento predefinito cambia: con i parametri di default `acc.core` dà risultati
identici a quelli del codice v0.2 (verificato sul pilota 2). I semi di valutazione (da 10.000 in su) non sono stati usati.

### Aggiunte al codice
- `anticluster(..., jitter=0.0)`: con `jitter = 0` la partizione è una funzione deterministica di X
  (due esecuzioni coincidono al 100%, cambia solo l'etichetta dei gruppi), quindi usata come
  assegnazione trattamento/controllo non è una randomizzazione. Con `jitter = 0.05` le esecuzioni
  sono indipendenti (concordanza 52%) e il bilanciamento resta quasi uguale
  (max |SMD| 0,0038 contro 0,0034; assegnazione casuale 0,126; n = 600, d = 5).
- `adaptive_phase1(..., rule=...)`: i tre candidati della sez. 6 del piano (`"density"`, `"yield"`,
  `"thompson"`), più `"none"` (mix fisso). Default `"v02"`.
- `adaptive_phase1(..., focus=0.0)`: quota di ogni ciclo estratta dalla regione candidata corrente.
  **Fuori dal piano congelato** (non è una regola di attribuzione): solo esplorativo.
- `select_projected` e `zoom_phase1`: due selettori esplorativi per la fase 1, fuori dal piano.
- 7 nuovi test (14 in totale).

### Risultati (dettagli in `results/dev_*`)
- **Errore di I tipo** (`experiments/dev_null_check.py`, semi 2.000–2.199): 200 regioni selezionate
  da P_adapt v0.2 sotto H0, fase 2 ripetuta 50 volte su ciascuna. Rifiuto 4,86% (e.s. 0,21) su
  10.000 test; P_rand 5,38% (e.s. 0,25). L'8,3% del pilota 2 (5/60) è compatibile con il caso.
  Non sostituisce H1 del piano (1.000 repliche sui semi di valutazione).
- **Regola di attribuzione** (`experiments/dev_rules.py`, semi 1.000–1.042, compatta_5 e coda_1):
  nessun candidato batte la v0.2. Regret massimo: v0.2 0,16 · density 0,21 · yield 0,26 · thompson 0,28.
  I candidati correggono la deriva verso il braccio casuale (quota di copertura finale in coda_1
  circa 0,70 contro 0,43) ma perdono altrettanto in compatta_5. La 0.3 non ha quindi una regola nuova da congelare.
- **Focus** (semi 1.000–1.070): le unità di S1 nella regione passano da circa 12 a 17–32, ma la
  potenza non aumenta (media 0,74 contro 0,76 della v0.2).
- **Dimensione di R̂** (`experiments/dev_rho.py`, semi 1.000–1.059, confronto appaiato): con R̂ = le n2
  unità con τ̂ più alto la precisione sale (coda_1, τ = 3: da 0,29 a 0,46 per P_l1) ma la potenza non
  cambia (−1,3 punti, e.s. 1,4).
- **Due vie esplorative per la fase 1** (`experiments/dev_explore.py`, semi 1.000–1.049, 50 repliche,
  anche su palla_rara_1: solo esplorativo). `select_projected` (copertura per singole covariate e coppie)
  è una copertura più equilibrata di λ = 1 (coda_1 0,64/0,98, compatta_5 0,30/0,70 a τ = 1,5/3) ma non
  batte il mix 50/50 (−3 punti in media, e.s. 3) e non aiuta su palla_rara_1. `zoom_phase1` (ricerca per
  celle con esiti aggregati) si comporta come il campionamento casuale: bene su compatta_5, male su coda_1;
  su palla_rara_1 0,20/0,50 contro 0,10/0,34 del mix, ma pari al casuale (0,24/0,40). Potenza media sulle
  6 celle: mix 0,55 · zoom 0,53 · proiezioni 0,51 · casuale 0,49 · λ = 1 0,45.
- **Diagnosi comune**: la potenza coincide quasi con la probabilità che la fase 1 individui la regione.
  Se R̂ contiene almeno il 5% di regione vera il test rifiuta nel 92% dei casi, altrimenti nel 10%.
  Il collo di bottiglia è la stima di τ(x) con pochi dati, non il campionamento successivo.

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
