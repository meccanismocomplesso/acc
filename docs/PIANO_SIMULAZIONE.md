# Piano della simulazione completa — congelato alla versione 0.2

Data di congelamento: 7 novembre 2026. Questo piano si fissa **prima** di vedere i risultati
della simulazione completa. Ogni modifica successiva va registrata in `CHANGELOG.md`
con data e motivo, e i risultati vanno presentati sia con il piano originale sia con quello modificato.

## 1. Domande e ipotesi

| Ipotesi | Enunciato | Criterio di successo |
|---|---|---|
| H1 Validità | I metodi P controllano l'errore di I tipo | Nello scenario nullo, limite superiore dell'IC 95% del tasso di rifiuto ≤ 7% per ogni metodo P (1.000 repliche) |
| H2 Copertura | Gli IC al 95% della fase 2 sono corretti | Copertura tra 93% e 97% in ogni cella |
| H3 Focalizzazione | La pipeline a due fasi è più potente dell'esperimento non focalizzato | Potenza di P_adapt > potenza di B1 in ≥ 80% delle celle non nulle |
| H4 Adattività | La fase 1 adattiva ha il regret massimo più basso | max sugli scenari di (potenza oracolo − potenza metodo) minore per P_adapt che per P_rand, P_l1 e P_mix |
| H5 Scalabilità | La fase 0 scala | anticlustering di N = 10^6, K = 3, d = 20 in meno di 60 s su un core |

L'**oracolo** è il migliore tra P_rand e P_l1 in ciascuna cella, scelto col senno di poi.

## 2. Metodi (congelati)

| Codice | Fase 1 | Fase 2 |
|---|---|---|
| B1 | casuale | casuale, ATE globale su tutte le B unità |
| B3 | rappresentativo (densità temperata, λ = 0) | casuale, ATE globale |
| B4 | casuale | enrichment classico: sottogruppi predefiniti = mediana di ciascuna covariata (2d sottogruppi); in fase 2 si campiona il sottogruppo con l'effetto stimato più alto in fase 1 |
| P_rand | casuale | R̂ = top ρ di τ̂, inferenza solo su S2 |
| P_l1 | densità temperata λ = 1 | come P_rand |
| P_mix | 50% casuale + 50% λ = 1 | come P_rand |
| P_adapt_v02 | regola adattiva della v0.2 (`acc.adaptive_phase1`, senza modifiche) | come P_rand |
| P_adapt_v03 | nuova regola di attribuzione (sez. 6) | come P_rand |
| B5 | oracolo di campionamento: S2 estratto da R* | limite superiore |

Parametri comuni: ρ = 3%, CATE con pseudo-outcome DR ed ExtraTrees (150 alberi, foglia minima 10),
bracci bilanciati con anticlustering a 2 gruppi, test unilaterale α = 5%, SE robusti HC1.

## 3. Disegno

**Disegno centrale** (tutte le combinazioni): d = 5, covariate gaussiane, N = 10^5, B = 1.000, n1/B = 0,4.

| Geometria dell'effetto | Descrizione | τ |
|---|---|---|
| nullo | τ(x) = 0 | 0 |
| omogeneo | τ(x) = τ ovunque | 1,5 · 3 |
| compatta_5 | palla con il 5% di P, nucleo denso | 1,5 · 3 |
| palla_rara_1 | palla con l'1% di P, densità medio-bassa | 1,5 · 3 |
| coda_1 | x4 oltre il 99° percentile | 1,5 · 3 |
| multi_2x2 | due palle separate, 2% ciascuna | 1,5 · 3 |
| gradiente | τ(x) = τ · Φ(x3), senza confini netti | 1,5 · 3 |

**Variazioni una alla volta**, sulle geometrie compatta_5, coda_1 e palla_rara_1, entrambi i τ:

| Fattore | Livelli oltre al centrale |
|---|---|
| Dimensione d | 20, 100 (selezione e CATE su 10 componenti principali) |
| Distribuzione delle covariate | t di Student con 3 gradi di libertà; mistura di 4 cluster; mista 3 continue + 2 categoriche (distanza di Gower) |
| Budget B | 500, 2.000 |
| Quota di esplorazione n1/B | 0,2 · 0,6 |
| N (solo tempi e una cella per geometria) | 10^6 |

Celle totali: 13 centrali + 54 di variazione = 67.

## 4. Repliche, semi e calcolo

- Scenario nullo: 1.000 repliche (errore Monte Carlo sul tasso di rifiuto circa 0,7 punti a 5%).
- Altre celle: 300 repliche (errore Monte Carlo sulla potenza ≤ 2,9 punti).
- **Semi di valutazione da 10.000 in su**, disgiunti da quelli dei pilota (0–99) e da quelli di sviluppo (sez. 6).
- Stima del calcolo: circa 21.000 job da circa 6 s, cioè circa 35 ore-CPU (circa 1 ora su 32 core).

## 5. Metriche

Primarie: potenza, errore di I tipo, copertura degli IC, regret rispetto all'oracolo.
Secondarie: precisione e richiamo di R̂, bias, unità di S1 nella regione, |SMD| massimo tra i bracci,
tempo di calcolo, costo per scoperta (B / tasso di scoperta).
Ogni tasso si riporta con il suo errore Monte Carlo.

## 6. Sviluppo della regola v0.3 (prima della valutazione)

La nuova regola di attribuzione si sviluppa **solo** sui semi di sviluppo 1.000–1.999 e sulle geometrie
compatta_5 e coda_1. Candidati da confrontare:

1. **Rendimento per braccio**: media del pseudo-outcome DR delle unità di ciascun braccio tra quelle con τ̂ più alto.
2. **Attribuzione normalizzata per densità**: attribuzione al vicino più prossimo pesata per l'inverso della quota del braccio nella zona.
3. **Thompson sampling** su una ricompensa = numero di unità del braccio nella regione candidata.

La regola scelta si congela in `acc.core` con un tag di versione **prima** di lanciare i semi di valutazione.

## 7. Regole di decisione

- Se H1 fallisce per un metodo P, quel metodo non si presenta come valido. Si analizza la causa prima di qualsiasi pubblicazione.
- Se H4 fallisce, il contributo si riformula come "mix robusto" invece che "adattivo".
- Le conclusioni sulle dimensioni elevate (d = 100) si riportano come esplorative.
