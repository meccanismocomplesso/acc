# acc — Registro dei tentativi (sviluppo verso la 0.3)

Lavoro svolto il 15 novembre 2026 sul repository `meccanismocomplesso/acc` (v0.2.0, commit 3e583c4).
Scopo: affrontare i limiti noti dichiarati nel README e capire dove si trova il collo di bottiglia del metodo.

## Regole seguite

- **Semi.** Solo semi di sviluppo (1.000–3.199). I semi di valutazione (da 10.000 in su) previsti dal piano congelato **non sono stati usati**.
- **Geometrie.** Lo sviluppo delle regole usa solo `compatta_5` e `coda_1`, come da piano. `palla_rara_1` compare solo nelle prove esplorative (sez. 7) e quei numeri non valgono come valutazione.
- **Default invariato.** Con i parametri predefiniti il codice modificato dà risultati identici al v0.2. È stato verificato sulle righe del pilota 2 e con un confronto diretto contro il codice originale.
- **Precisione.** Con 43–71 repliche per cella l'errore Monte Carlo sulla potenza è di 6–7 punti. I confronti "nessun miglioramento" significano *nessuna evidenza di miglioramento*, non *impossibile*.

Notazione: potenza = frequenza di rifiuto nel test di fase 2. "Regione" = regione vera in cui agisce l'effetto. τ = 1,5 effetto debole, τ = 3 effetto forte.

---

## 1. Errore di I tipo nello scenario nullo — **risolto**

**Problema.** Nel pilota 2 il metodo adattivo rifiutava H0 in 5 casi su 60 (8,3%).

**Metodo** (`experiments/dev_null_check.py`, semi 2.000–2.199). Si eseguono 200 fasi 1 reali senza effetto. Per ognuna la fase 2 viene ripetuta 50 volte, con nuovo campione, nuova assegnazione e nuovo rumore. Si ottengono così 10.000 test su regioni selezionate davvero dal metodo, con errore standard calcolato per replica.

**Risultato.**

| Metodo | Rifiuto | e.s. |
|---|---|---|
| P_adapt (v0.2) | 4,86% | 0,21 |
| P_rand | 5,38% | 0,25 |

**Conclusione.** L'8,3% del pilota era una fluttuazione. L'inferenza dopo la selezione è valida, come da costruzione del disegno (fase 2 su unità fresche).
**Valore di ricerca:** alto. Toglie il principale dubbio sulla validità del metodo. Non sostituisce la verifica H1 del piano (1.000 repliche sui semi di valutazione).

---

## 2. Assegnazione trattamento/controllo non casuale — **difetto trovato e corretto**

**Problema** (non presente tra i limiti noti). `anticluster(X, 2, rng)` produce una partizione che è funzione deterministica di X. Due esecuzioni con semi diversi danno gli stessi due gruppi nel 100% dei casi, e il seme cambia solo quale gruppo riceve l'etichetta "trattato". Usata come assegnazione, quindi, non è una randomizzazione: rimane solo una moneta sull'etichetta.

**Correzione.** Nuovo parametro `anticluster(..., jitter=0.0)`. Con `jitter > 0` l'ordine di elaborazione viene perturbato con rumore gaussiano.

| Assegnazione (n = 600, d = 5) | max \|SMD\| | Concordanza tra esecuzioni |
|---|---|---|
| anticluster, jitter = 0 (v0.2) | 0,0034 | 100% |
| anticluster, jitter = 0,05 | 0,0038 | 52% |
| casuale | 0,126 | ~50% |

**Conclusione.** Con `jitter = 0.05` si ottiene una randomizzazione vera, con bilanciamento praticamente invariato.
**Valore di ricerca:** alto sul piano metodologico. Un revisore lo noterebbe, e giustifica formalmente l'uso dell'inferenza basata sulla randomizzazione.
**Da fare:** decidere se rendere `jitter = 0.05` il default nella fase di assegnazione (cambia i risultati riproducibili del pilota).

---

## 3. Regola di attribuzione della fase 1 adattiva — **nessun miglioramento**

**Problema.** La regola v0.2 scivola verso il braccio casuale anche quando l'effetto è nella coda.

**Metodo** (`experiments/dev_rules.py`, semi 1.000–1.042, 43 repliche complete). Sono stati implementati i tre candidati della sez. 6 del piano: `density` (attribuzione normalizzata per densità), `yield` (rendimento per braccio) e `thompson` (Thompson sampling sulle unità nella regione stimata), più `none` (mix fisso).

**Risultato** (celle con effetto; regret = scarto dal migliore tra P_rand e P_l1 col senno di poi).

| Regola | Regret massimo | Regret medio | Potenza minima | Potenza media |
|---|---|---|---|---|
| **v0.2** | **0,163** | **0,093** | **0,558** | **0,756** |
| density | 0,209 | 0,157 | 0,488 | 0,692 |
| thompson | 0,279 | 0,140 | 0,419 | 0,709 |
| yield | 0,256 | 0,163 | 0,442 | 0,686 |

**Osservazione.** I candidati correggono la deriva: in `coda_1` la quota di copertura finale sale da circa 0,43 a circa 0,70. Però perdono altrettanto in `compatta_5`.
**Conclusione.** La 0.3 non ha una regola nuova da congelare: la v0.2 resta la migliore tra quelle previste.
**Valore di ricerca:** medio. È un risultato negativo pulito, che evita di spendere le simulazioni finali su regole peggiori.

---

## 4. Campionamento diretto nella regione candidata ("focus") — **nessun miglioramento**

**Idea** (fuori piano). Da un ciclo all'altro una quota del lotto viene estratta direttamente dalla regione dove τ̂ è più alto, invece che dai due bracci ciechi. Parametro `adaptive_phase1(..., focus=...)`.

**Metodo.** Semi 1.000–1.070 (71 repliche per le varianti focus), con un confronto appaiato su 43 repliche contro la v0.2.

**Risultato.**

| Variante | Unità di S1 nella regione | Potenza media | Differenza vs v0.2 |
|---|---|---|---|
| v0.2 | ~12 | 0,756 | — |
| focus 50%, quota fissa | 17–32 | 0,733 | −0,023 (e.s. 0,044) |
| focus 50%, regola v0.2 | 13–34 | 0,698 | −0,058 (e.s. 0,043) |
| focus 25%, regola v0.2 | 14–25 | 0,709 | −0,047 (e.s. 0,046) |

**Conclusione.** Raddoppiare le unità utili non aumenta la potenza. Per costruzione il focus rinforza la partenza: chi parte bene migliora, chi parte male concentra il campione sul rumore.
**Valore di ricerca:** medio. Mostra che il problema non è *quante* unità finiscono nella regione dopo che è stata trovata.

---

## 5. Dimensione della regione selezionata R̂ — **nessun effetto sulla potenza**

**Idea.** Se la regione vera è piccola (1%), una R̂ al 3% ha precisione al massimo pari a 1/3 e diluisce l'effetto.

**Metodo** (`experiments/dev_rho.py`, semi 1.000–1.059). Confronto appaiato a parità di fase 1 tra tre dimensioni di R̂: le n2 unità con τ̂ più alto, il 3% (pilota) e il 5%.

**Risultato.**
- La precisione sale con R̂ più piccola: in `coda_1`, τ = 3, P_l1 passa da 0,29 a 0,46.
- La potenza non cambia: n2 rispetto a 3% dà −0,013 (e.s. 0,014); 5% rispetto a 3% dà −0,017 (e.s. 0,015), su 720 coppie.

**Conclusione.** La scelta ρ = 3% non è un collo di bottiglia.

---

## 6. Diagnosi comune — **risultato principale**

Unendo le prove 3–5 (800 osservazioni) emerge questo quadro.

| La fase 1 ha individuato la regione? (precisione di R̂ ≥ 5%) | Casi | Potenza |
|---|---|---|
| Sì | 559 | 91,6% |
| No | 241 | 9,5% |

Correlazione tra "regione trovata" e "rifiuto": 0,80.

**Conclusione.** La potenza è quasi tutto-o-niente e dipende da un solo evento: se la stima di τ(x) con circa 400 osservazioni intercetta la regione. Il collo di bottiglia è il **rilevamento in fase 1**, non il campionamento successivo, la regola adattiva o la dimensione di R̂.
**Valore di ricerca:** alto. Indica dove concentrare il lavoro e dove non insistere.

---

## 7. Vie esplorative sul campionamento della fase 1 — **nessuna supera il mix 50/50**

`experiments/dev_explore.py`, semi 1.000–1.049, 50 repliche per cella. Include `palla_rara_1`, quindi questi risultati sono **solo esplorativi**.

### 7a. Peso di rarità per combinazioni di covariate (`select_projected`)

**Idea.** Pesare la rarità non sulla densità congiunta, che porta il campione alle code estreme, ma su ogni singola covariata e su ogni coppia. In questo modo si favoriscono le zone "insolite per una combinazione" che restano ordinarie nelle altre dimensioni.

**Motivazione dal pilota 1** (unità di S1 nella regione, su 400): il peso di rarità congiunto peggiora la regione rara intermedia.

| Scenario | Casuale | λ = 0,5 | λ = 1 |
|---|---|---|---|
| coda_1 | 3,6 | 10,7 | 21,0 |
| compatta_5 | 20,3 | 5,3 | 1,5 |
| palla_rara_1 | 4,3 | 2,9 | 1,6 |

### 7b. Ricerca a zoom per celle (`zoom_phase1`)

**Idea.** Si divide la popolazione in 16 celle (k-means) e si stima l'effetto medio per cella, che è meno rumoroso del dato singolo. Si tiene la metà più promettente, la si divide in due e si ripete. Si ispira al campionamento adattivo a grappoli (Thompson, 1990), applicato però a gruppi, perché l'effetto individuale non è osservabile.

### Risultati (potenza, τ = 1,5 / τ = 3)

| Scenario | Casuale | Copertura λ = 1 | Mix 50/50 | Proiezioni | Zoom |
|---|---|---|---|---|---|
| coda_1 | 16 / 38 | 82 / 100 | 52 / 86 | 64 / 98 | 24 / 58 |
| compatta_5 | 76 / 100 | 18 / 30 | 52 / 94 | 30 / 70 | 64 / 100 |
| palla_rara_1 | 24 / 40 | 14 / 28 | 10 / 34 | 18 / 28 | 20 / 50 |
| **Media** | 49 | 45 | **55** | 51 | 53 |

Confronti appaiati con il mix:
- **Proiezioni:** −3,3 punti (e.s. 3,4); +12 in coda_1, −23 in compatta_5, +1 in palla_rara_1.
- **Zoom:** −2,0 punti (e.s. 3,5); −28 in coda_1, +9 in compatta_5, +13 in palla_rara_1.

**Conclusioni.**
- Le **proiezioni** sono una copertura più equilibrata di λ = 1 (nel nucleo denso perdono molto meno), ma da sole non battono il mix e non aiutano la regione rara.
- Lo **zoom** si comporta come il campionamento casuale, perché le celle iniziali seguono la densità. Sulla regione rara con effetto forte è il migliore (50%), ma il vantaggio sul casuale (40%) è entro l'errore.
- Sette varianti di campionamento si fermano tutte tra il 10% e il 50% sulla regione rara intermedia. Con circa 4 unità su 400 nella regione il limite è la quantità di informazione, non l'algoritmo di selezione.

**Valore di ricerca:** medio. Il mix 50/50 si conferma come scelta robusta, ed è coerente con il ripiego "mix robusto" già previsto dal piano.

---

## 8. La v0.2 su popolazioni con code pesanti — **punto debole nuovo trovato**

Studio del 18 novembre 2026 (`experiments/study_tails.py`, `analyze_tails.py`). Codice v0.2 invariato, parametri di default.
Popolazioni: gaussiana (riferimento), t di Student con 5 e con 3 gradi di libertà, riscalate a varianza 1. d = 5, N = 100.000.
Gli scenari, l'esito e la fase 2 sono quelli del pilota 2; 40 repliche per cella; semi 4.000–4.039 (nuovi, non di valutazione).
Errore Monte Carlo sulla potenza: circa 7–8 punti per cella.

### 8a. La diagnostica satura

| Popolazione | Rapporto code | Curtosi in eccesso (media; max) | λ0 |
|---|---|---|---|
| Gaussiana | 1,00 | 0,0 | 0,50 |
| t5 | 1,83 | 5,6; 44 | 0,80 |
| t3 | 1,81 | 244; 3.522 | 0,80 |

Il termine di curtosi esplode con code pesanti (per t3 la curtosi teorica è infinita), quindi λ0 va sempre al limite di 0,8.
La diagnostica distingue solo "normale" da "non normale". Anche il rapporto delle code non separa t5 da t3 (1,83 contro 1,81).

### 8b. Il metodo adattivo ignora la partenza

Con λ0 = 0,8 la quota di copertura scende a circa 0,65 dopo il primo ciclo e a 0,49–0,66 alla fine, **anche nello scenario nullo** (0,54–0,57).
Con la gaussiana, partendo da 0,5, scende a 0,26–0,45. La regola tende a ridurre la quota indipendentemente dal segnale,
quindi l'informazione della diagnostica viene in buona parte cancellata.

### 8c. Potenza (τ = 1,5 / τ = 3)

| Scenario | Popolazione | Casuale | Copertura λ = 1 | Mix 50/50 | Adattivo v0.2 |
|---|---|---|---|---|---|
| Coda | gauss | 20 / 42 | 88 / 100 | 42 / 88 | 45 / 88 |
| | t5 | 20 / 40 | 85 / 100 | 60 / 98 | 72 / 98 |
| | t3 | 22 / 35 | 55 / 100 | 60 / 92 | 48 / 98 |
| Nucleo denso | gauss | 65 / 100 | 10 / 35 | 38 / 92 | 60 / 92 |
| | t5 | 68 / 100 | 18 / 18 | 20 / 72 | 30 / 57 |
| | t3 | 55 / 98 | 10 / 30 | 10 / 35 | 22 / 40 |
| Regione rara | gauss | 12 / 38 | 2 / 10 | 30 / 42 | 15 / 40 |
| | t5 | 18 / 40 | 5 / 25 | 5 / 25 | 5 / 10 |
| | t3 | 8 / 30 | 10 / 8 | 2 / 12 | 15 / 18 |

Potenza media sulle celle con effetto:

| Popolazione | Casuale | Copertura | Mix | Adattivo |
|---|---|---|---|---|
| gauss | 0,46 | 0,41 | 0,55 | 0,57 |
| t5 | 0,48 | 0,42 | 0,47 | 0,45 |
| t3 | 0,41 | 0,35 | 0,35 | 0,40 |

- Con code pesanti **il vantaggio di mix e adattivo sul campionamento casuale sparisce**: in t3 il casuale è alla pari o meglio.
- Nel nucleo denso il crollo è netto: il mix passa da 92% a 35% (τ = 3) pur mettendo nella regione lo stesso numero di unità (10,2 contro 11,2 nella gaussiana).
- Nella coda le code pesanti aiutano la copertura a trovare la regione (39–40 unità contro 21), ma con effetto debole in t3 la potenza della copertura scende comunque (55% contro 88%).

### 8d. Causa (diagnosi su 15 repliche, nucleo denso, mix 50/50, τ = 3)

| | Gaussiana | t3 |
|---|---|---|
| Valore assoluto massimo di una covariata in S1 | 4,1 | 47,7 |
| Quota di S1 nel 5% più esterno della popolazione | 15% | 43% |
| Quota di R̂ nel 5% più esterno | 2% | 22% |
| Precisione di R̂ | 0,40 | 0,03 |
| Precisione togliendo il trend lineare noto dall'esito | 0,42 | 0,11 |

Il braccio di copertura, per costruzione, va a prendere i punti più isolati, e con code pesanti questi sono outlier estremi (fino a 48 deviazioni standard).
Due effetti si sommano:
1. **Spreco di budget.** Quasi metà della fase 1 finisce su punti estremi che non dicono nulla sulla maggior parte della popolazione.
2. **Stima di τ(x) contaminata.** Sugli outlier il modello dell'esito (foresta) non riesce a estrapolare il trend di base: i residui e quindi gli pseudo-outcome diventano enormi, e la regione stimata R̂ viene attratta verso le zone estreme.
   Togliere il trend lineare noto recupera solo in parte (da 0,03 a 0,11): il resto è dovuto allo spreco di budget e alla geometria distorta.

Il campionamento casuale non ne soffre perché pesca raramente gli outlier.

### 8e. Cosa resta valido

- **Errore di I tipo**: tra 0% e 12,5% con 40 repliche, tutti compatibili con il 5% (e.s. 2,5–5 punti).
- **Copertura degli IC**: 92–97% in tutte le popolazioni.
- **Bilanciamento**: fase 0 con max |SMD| < 0,00002; trattamento/controllo < 0,01 in tutti i casi.

La validità dell'inferenza regge; è la capacità di trovare la regione che si rompe.

### Valore di ricerca: **alto**

È un limite non documentato che colpisce proprio il caso per cui la diagnostica esiste (code pesanti). Interventi possibili, non ancora provati:
- **Diagnostica robusta**: sostituire la curtosi con una misura stabile (per esempio basata su quantili) o limitarla.
- **Standardizzazione robusta**: mediana e scarto interquartile invece di media e deviazione standard, così gli outlier non schiacciano il resto.
- **Copertura limitata**: escludere dal braccio di copertura il percentile più estremo, oppure temperare la densità entro un raggio massimo.
- **Modello dell'esito più robusto** in `fit_cate`: una componente lineare più la foresta sui residui, per estrapolare meglio sugli outlier.

---

## 9. Ripartizione del budget tra fase 1 e fase 2 — **primo miglioramento netto**

Studio del 18 novembre 2026 (`experiments/study_budget.py`, `analyze_budget.py`). Budget totale B = 1.000 invariato;
la fase 1 riceve il 20%, il 40% (default dei pilota) o il 60%. Codice v0.2, popolazione gaussiana, scenari del pilota 2.
40 repliche per cella, semi 5.000–5.039 (nuovi, non di valutazione). Ogni replica usa la stessa popolazione per le
tre ripartizioni, quindi i confronti sono appaiati.

### Potenza media sulle celle con effetto

| Metodo | 20% | 40% | 60% | 60% − 40% (appaiato) |
|---|---|---|---|---|
| Casuale | 0,39 | 0,46 | 0,58 | +0,12 (e.s. 0,04) |
| Copertura λ = 1 | 0,32 | 0,48 | 0,51 | +0,04 (e.s. 0,04) |
| Mix 50/50 | 0,33 | 0,55 | 0,62 | +0,08 (e.s. 0,04) |
| Adattivo v0.2 | 0,38 | 0,57 | **0,65** | +0,08 (e.s. 0,04) |

Ridurre la fase 1 al 20% costa da 7 a 22 punti in tutti i metodi.

### Potenza per cella (τ = 1,5 / τ = 3), adattivo v0.2

| Scenario | 20% | 40% | 60% |
|---|---|---|---|
| Coda | 28 / 70 | 42 / 98 | 57 / 100 |
| Nucleo denso | 40 / 62 | 60 / 92 | 82 / 100 |
| Regione rara | 5 / 22 | 15 / 35 | 12 / 35 |

La regione rara resta sotto il 40% con qualsiasi ripartizione e metodo, tranne il casuale al 60% con τ = 3 (65%).

### Perché funziona

| Quota fase 1 | Casi in cui R̂ contiene almeno il 5% di regione vera | Potenza se trovata | Potenza se non trovata |
|---|---|---|---|
| 20% | 34% | 87% | 9% |
| 40% | 52% | 89% | 11% |
| 60% | 63% | 89% | 9% |

Più budget in fase 1 aumenta la probabilità di trovare la regione. Una volta trovata, 400 unità di conferma
bastano quanto 600 o 800: la potenza condizionata resta all'89%. La fase 2 era sovradimensionata.

### Validità

- Errore di I tipo tra 0% e 15% (casuale al 20%: 6 su 40, e.s. 5,6 punti); gli altri valori sono compatibili con il 5%.
- Copertura degli IC: 95–96% in tutte le ripartizioni.

### Valore di ricerca: **alto**

È il primo intervento che aumenta la potenza rispetto alla configurazione pubblicata: adattivo da 0,57 a 0,65.
Conferma la diagnosi del punto 6: il collo di bottiglia è trovare la regione, e il budget speso a capire la popolazione rende più di quello speso a confermare.
Non è stato provato oltre il 60%, quindi non si sa dove si trovi l'ottimo.

---

## 10. Fase 1 a lunghezza variabile — **nessun guadagno, ma due risultati utili**

Studio del 18 novembre 2026 (`experiments/study_seqlen.py`, `analyze_seqlen.py`; funzioni `acc.sequential_phase1`
e `acc.heterogeneity_t`). Fase 1 a cicli da 100 unità (50 dalla copertura, 50 a caso). Dopo ogni ciclo l'algoritmo
decide se fermarsi e passare alla conferma con il budget restante (B = 1.000). Arresto tra 300 e 800 unità.
Il campionamento non dipende dalla regola di arresto, quindi tutte le regole e tutte le ripartizioni fisse sono
confrontate sulla stessa sequenza (confronto appaiato). Popolazione gaussiana, 40 repliche per cella,
semi 6.000–6.039 (6.900–6.907 solo per la calibrazione).

### 10a. Due criteri di arresto, entrambi deboli

- **Stabilità della regione stimata** (Jaccard tra R̂ di cicli consecutivi): scartata in calibrazione.
  Cresce anche senza effetto (da circa 0,2 a 0,5), perché da un ciclo all'altro i dati coincidono quasi del tutto.
  Misura la sovrapposizione dei dati, non la presenza del segnale.
- **Test di eterogeneità** (`heterogeneity_t`): la stima di τ(x) fatta su metà delle unità deve predire chi
  risponde di più nell'altra metà. Una prima versione era distorta: l'esito di un'unità entrava sia nella stima sia
  nella verifica, e senza effetto dava in media −1,7. La versione corretta separa del tutto le due metà ed è centrata
  sullo zero senza effetto. Ma separa poco: anche con effetto forte la statistica media resta tra 0,5 e 1,2.

| Scenario | t medio a 300 unità | t medio a 800 unità | Prefissi con t ≥ 2,5 |
|---|---|---|---|
| Nullo | −0,02 | 0,43 | 2,9% |
| Coda, τ = 3 | 0,55 | 0,82 | 10,0% |
| Nucleo denso, τ = 3 | 0,06 | 1,16 | 12,1% |
| Regione rara, τ = 3 | 0,01 | 0,47 | 3,8% |

### 10b. Risultati

| Regola | Arresto medio | Potenza media | Differenza con fissa 800 (appaiata) |
|---|---|---|---|
| Fissa 300 | 300 | 0,39 | −0,15 (e.s. 0,03) |
| Fissa 400 | 400 | 0,45 | −0,09 (e.s. 0,03) |
| Fissa 500 | 500 | 0,46 | −0,08 (e.s. 0,03) |
| Fissa 600 | 600 | 0,50 | −0,04 (e.s. 0,03) |
| Fissa 700 | 700 | 0,52 | −0,03 (e.s. 0,03) |
| **Fissa 800** | 800 | **0,54** | — |
| Variabile, t ≥ 2 | 620–740 | 0,53 | −0,02 (e.s. 0,01) |
| Variabile, t ≥ 2,5 | 650–790 | 0,54 | −0,00 (e.s. 0,01) |
| Variabile, t ≥ 3 | 760–800 | 0,55 | +0,00 (e.s. 0,01) |

- **La lunghezza variabile non aggiunge niente.** Il test si attiva di rado, quindi le regole arrivano quasi sempre
  al massimo e si comportano come la fase 1 fissa a 800.
- **Più esplorazione continua a rendere fino all'80%.** Con questo campionatore la potenza cresce in modo regolare
  da 300 a 800 unità; la conferma con 200 unità basta ancora. L'ottimo della ripartizione fissa è quindi almeno all'80%,
  oltre il 60% del punto 9.
- **Regione rara:** 30–40% con τ = 3, e 0–15% con τ = 1,5, con qualsiasi regola o ripartizione.
- **Validità:** errore di I tipo 5–7,5%, copertura IC 93–97%.

I valori assoluti non sono confrontabili con il punto 9 (campionatore a cicli invece che globale, semi diversi):
vanno letti solo all'interno di questo studio.

### 10c. Perché non funziona, e cosa significa

La fase 1 non riesce a capire da sola se ha trovato la regione. Un test di eterogeneità sulle unità della fase 1 guarda
tutta la popolazione campionata, e un effetto concentrato nel 5% o nell'1% si diluisce. Il test finale della fase 2
invece concentra unità nuove proprio nella regione candidata, ed è molto più sensibile.

Ne segue che la fase 2 è anche il miglior strumento di verifica che l'algoritmo ha. Questo suggerisce di cambiare la
struttura invece della durata dell'esplorazione: spendere il budget di conferma per verificare più regioni candidate
invece di una sola (vedi "Idee non ancora provate").

### Valore di ricerca: **medio-alto**

Smentisce l'idea più naturale (esplorare finché non si è sicuri) e ne spiega il motivo. Estende il risultato del punto 9
(ottimo almeno all'80%). E introduce un test di eterogeneità corretto e riutilizzabile, con la lezione sulla distorsione
della prima versione.

---

## 11. Fase 2 con più regioni candidate — **la regione rara viene catturata più spesso, ma la potenza non sale**

Studio del 19 novembre 2026 (`experiments/study_multicand.py`, `analyze_multicand.py`). Fase 1 con il metodo adattivo
v0.2, n1 = 400 o 600 su B = 1.000. Dalla stima di τ(x) si prendono le unità con τ̂ nel top 3% × K e le si divide con
k-means in K candidate distinte, ciascuna grande circa quanto la R̂ attuale (3%). Varianti, a parità di budget:

- **BASE**: fase 2 attuale, una sola R̂.
- **BONF_K**: il budget di conferma è diviso tra le K candidate, ognuna testata a α/K.
- **SCREEN_K**: metà del budget valuta tutte le candidate; si sceglie quella con la z più alta e la si conferma con
  l'altra metà su unità nuove, con un solo test a α = 5%.

K = 2, 3, 4, quota di valutazione 50% e dimensione delle candidate sono stati fissati prima di vedere i risultati.
Popolazione gaussiana, 80 repliche per cella, semi 7.000–7.079. Errore Monte Carlo sulla potenza: circa 5 punti per cella.
"Potenza" qui conta solo le scoperte corrette, cioè i rifiuti in una candidata che contiene davvero effetto.

### 11a. Catturare la regione: funziona

Casi in cui almeno una candidata contiene la regione vera (precisione ≥ 5%):

| Scenario | n1 | BASE | K = 2 | K = 3 | K = 4 |
|---|---|---|---|---|---|
| Regione rara, τ = 1,5 | 400 | 9% | 18% | 25% | 36% |
| | 600 | 18% | 34% | 39% | 50% |
| Regione rara, τ = 3 | 400 | 30% | 45% | 56% | 56% |
| | 600 | 44% | 55% | 62% | 70% |
| Coda, τ = 1,5 | 600 | 76% | 85% | 86% | 91% |
| Nucleo denso, τ = 1,5 | 600 | 78% | 86% | 92% | 96% |

Con più candidate la regione rara viene catturata fino a più del doppio delle volte.
Nello screening la candidata giusta viene scelta nel 77–93% dei casi in cui è presente.

### 11b. Potenza: non migliora

Potenza media sulle celle con effetto:

| n1 | BASE | BONF_2 | BONF_3 | BONF_4 | SCREEN_2 | SCREEN_3 | SCREEN_4 |
|---|---|---|---|---|---|---|---|
| 400 | **0,53** | 0,52 | 0,48 | 0,44 | 0,52 | 0,52 | 0,51 |
| 600 | **0,61** | 0,57 | 0,55 | 0,49 | 0,58 | 0,59 | 0,57 |

Solo regione rara, differenza appaiata con BASE: tra −0,11 e +0,04. Nessun guadagno significativo (il migliore è SCREEN_3 con n1 = 400: +0,04, e.s. 0,03); BONF_4 con n1 = 600 perde in modo netto (−0,11, e.s. 0,04).
La correzione di Bonferroni perde sempre, e di più al crescere di K. Lo screening è circa neutro.

### 11c. Perché: la regione catturata è diluita

Regione rara, solo i casi in cui la regione è catturata:

| Variante | n1 | Casi su 160 | Precisione media della candidata | Potenza |
|---|---|---|---|---|
| BASE | 400 | 31 | 0,10 | 71% |
| SCREEN_3 | 400 | 65 | 0,12 | 54% |
| BASE | 600 | 49 | 0,11 | 69% |
| SCREEN_3 | 600 | 81 | 0,13 | 33% |

La regione rara occupa l'1% della popolazione e ogni candidata il 3%: anche quando la contiene, solo circa il 12% delle
unità della candidata riceve davvero l'effetto. Con τ = 3 l'effetto medio misurato è quindi circa 0,35, e la conferma
dispone della metà del budget. I casi in più catturati sono quelli più difficili. Per BASE la potenza condizionata è più
alta anche perché i casi catturati da una sola regione sono i più facili.

### 11d. Validità

- Errore di I tipo (rifiuto globale): 0–10%, compatibile con il 5% (e.s. circa 2,4 punti).
- Rifiuti in una candidata senza effetto, quando l'effetto c'è altrove: 0,2–2,7%.
- Copertura degli IC: 94–96%.

### Valore di ricerca: **alto**

Sposta il problema della regione rara: con più candidate la regione viene localizzata molto più spesso, ma resta diluita
dentro una candidata troppo grande. Il collo di bottiglia per la regione rara non è più "trovarla" ma "isolarla".
Il passo successivo naturale è restringere la candidata scelta prima della conferma (vedi "Idee non ancora provate").

---

## 12. Isolare all'inizio lo strato raro nelle covariate — **utile quando c'è davvero qualcosa da isolare**

Studio del 19 novembre 2026 (`experiments/study_strat.py`, `analyze_strat.py`; funzioni `acc.outer_stratum` e
`acc.outer_ratio`). Idea: separare all'inizio, con una regola sulle sole covariate, il gruppo raro nelle covariate,
studiarlo a parte e riunire i risultati alla fine. Non si applica alla rarità nell'effetto, che si vede solo dagli esiti.

**Regola.** Strato esterno O = il 2% del learning set più lontano dal centro, con distanza robusta (mediana e scarto
interquartile). Corpo principale M = il resto. Budget B = 1.000. Metodi:
- **P_rand**: fase 1 casuale (600) + fase 2 (400) su tutto.
- **BASE**: acc v0.2, fase 1 adattiva (600) + fase 2 (400) su tutto.
- **CLEAN**: acc v0.2 solo su M con tutto il budget; O ignorato.
- **STRAT**: M con acc v0.2 (520 + 320, test a α/2); O con 160 unità proprie e test dell'effetto medio di O a α/2.
  Rifiuto globale se uno dei due rifiuta. O è fissato dalle sole covariate, quindi il suo test non richiede unità nuove.

**Popolazioni**: gaussiana; t di Student con 3 g.d.l. (code molto pesanti); "mix" = 99% normale + 1% gruppo isolato
centrato a 3,5 deviazioni standard in due covariate. Per mix gli scenari sono nullo, effetto nel gruppo isolato e nucleo
denso. 40 repliche per cella, semi 8.000–8.039; 2%, 160 unità e 520/320 fissati prima di vedere i risultati.
Errore Monte Carlo: circa 7–8 punti per cella.

### 12a. La regola isola bene

Nel mix lo strato esterno contiene il 99% del gruppo isolato. Nella coda (gauss e t3) contiene il 22% della regione.
Nelle regioni interne (nucleo denso, regione rara) non ne contiene nulla.

### 12b. Potenza media sulle celle con effetto

| Popolazione | Casuale | BASE (v0.2) | CLEAN | STRAT |
|---|---|---|---|---|
| Gaussiana | 0,54 | **0,61** | 0,55 | 0,49 |
| Code pesanti (t3) | 0,49 | 0,47 | **0,57** | 0,52 |
| Gruppo isolato (mix) | 0,74 | 0,85 | 0,45 | **0,86** |

Differenze appaiate con BASE:

| Popolazione | CLEAN | STRAT |
|---|---|---|
| Gaussiana | −0,06 (e.s. 0,03) | −0,12 (e.s. 0,04) |
| t3 | +0,10 (e.s. 0,04) | +0,05 (e.s. 0,03) |
| mix | −0,40 (e.s. 0,05) | +0,01 (e.s. 0,03) |

Celle significative:
- **Gruppo isolato, τ = 1,5**: STRAT 98% contro BASE 78% e casuale 38%. Il test sullo strato esterno rifiuta nel 97,5% dei casi.
  CLEAN 0%: escludere lo strato senza studiarlo cancella proprio il gruppo dove sta l'effetto.
- **t3, nucleo denso, τ = 3**: BASE 62%, CLEAN 85%, STRAT 82%. Togliere gli estremi ripara il problema del punto 8.
- **Gaussiana, nucleo denso, τ = 1,5**: BASE 65%, STRAT 40%. Senza nulla da isolare la stratificazione costa
  (budget e soglia divisi in due, coda in parte tolta al corpo principale).

### 12c. Una diagnostica dalle sole covariate decide quando stratificare

Rapporto tra il quantile 99,5% e il 90% della distanza robusta (`acc.outer_ratio`); per una normale in 5 dimensioni vale 1,35.

| Popolazione | Minimo | Media | Massimo |
|---|---|---|---|
| Gaussiana | 1,34 | 1,35 | 1,36 |
| Gruppo isolato | 1,61 | 1,62 | 1,64 |
| t3 | 2,44 | 2,55 | 2,62 |

Separazione completa nelle 40 repliche di ogni popolazione. Una regola condizionata, "stratifica se il rapporto supera di
oltre il 10% il valore atteso", sceglie BASE su tutte le gaussiane e STRAT su tutte le altre. Ricostruita dai dati
dello studio (la scelta dipende solo dalle covariate, quindi è legittimo):

| Popolazione | BASE | Regola condizionata |
|---|---|---|
| Gaussiana | 0,61 | 0,61 |
| t3 | 0,47 | 0,52 |
| mix | 0,85 | 0,86 |
| Complessiva (appaiata) | — | +0,02 (e.s. 0,015) |

La soglia del 10% è stata scelta dopo aver visto le distribuzioni del rapporto, ma a partire dal valore atteso teorico;
le gaussiane stanno tutte entro l'1%. Va verificata su forme di popolazione diverse da queste tre.

### 12d. Validità

- Errore di I tipo: 0–7,5%, compatibile con il 5%.
- Copertura degli IC: 93,5–98,5%.

### Valore di ricerca: **alto**

È il primo intervento che aggiunge una capacità nuova: un gruppo raro nelle covariate viene studiato a parte invece di
essere trattato come rumore o scartato. Mostra anche che isolare senza studiare (CLEAN) è pericoloso, e che stratificare
quando non c'è nulla da isolare costa. Con la diagnostica la stratificazione si attiva solo quando serve.
CLEAN resta la migliore sulle code pesanti pure (t3), ma solo STRAT è sicura in tutti e tre i casi.

---

## 13. Confronto con metodi esistenti — **acc migliorato batte tutti; la v0.2 pubblicata no**

Studio del 19 novembre 2026 (`experiments/study_competitors.py`, `analyze_competitors.py`). Budget B = 1.000 per tutti,
test unilaterali al 5% su unità nuove rispetto alla selezione (oppure su gruppi definiti dalle sole covariate).
Popolazioni: gaussiana, t3 e mix (gruppo isolato dell'1%). 17 celle, 40 repliche ciascuna, semi 9.000–9.039.

### Metodi

Concorrenti (implementazioni semplificate):
- **ATE**: esperimento unico non focalizzato, test dell'effetto medio.
- **GATES** (Chernozhukov, Demirer, Duflo e Fernández-Val, "Generic machine learning inference on heterogeneous treatment
  effects in randomized experiments", 2018): esperimento unico, metà stima τ(x), l'altra metà testa il quinto superiore.
  Una sola divisione del campione (l'articolo ne combina molte).
- **VT**, virtual twins (Foster, Taylor e Ruberg, *Statistics in Medicine*, 2011): 600 unità casuali, una foresta per braccio,
  albero di profondità 3 sulle differenze, conferma su 400 unità nuove nella foglia migliore.
- **ENRICH**, arricchimento adattivo a soglia su un biomarcatore (nello spirito di Simon e Simon, *Biometrics*, 2013):
  600 casuali, si sceglie il sottogruppo {x_j sopra o sotto un quantile} con la z più alta, conferma su 400 nuove.
- **ML2S**: due stadi generico, 600 casuali + foresta DR + top 3% + 400 nuove. È acc senza il campionamento guidato.

Versioni di acc:
- **ACC40**: v0.2 pubblicata.
- **ACC60**: v0.2 con fase 1 al 60% (punto 9).
- **ACC_COND**: ACC60 più la stratificazione condizionata (punto 12). In questo studio si è attivata su tutte le popolazioni
  t3 e mix e su nessuna gaussiana.

### 13a. Potenza media sulle celle con effetto

| Popolazione | ATE | GATES | VT | ENRICH | ML2S | ACC40 | ACC60 | ACC_COND |
|---|---|---|---|---|---|---|---|---|
| Gaussiana | 0,22 | 0,22 | 0,29 | 0,19 | 0,54 | 0,51 | **0,63** | **0,63** |
| t3 | 0,25 | 0,20 | 0,25 | 0,28 | 0,51 | 0,38 | 0,49 | **0,54** |
| Gruppo isolato | 0,08 | 0,14 | 0,19 | 0,26 | 0,66 | 0,73 | 0,90 | **1,00** |
| **Tutte** | 0,21 | 0,20 | 0,26 | 0,24 | 0,54 | 0,48 | 0,61 | **0,65** |

Differenze appaiate su tutte le celle con effetto:

| | ATE | GATES | VT | ENRICH | ML2S | ACC40 |
|---|---|---|---|---|---|---|
| ACC60 meno… | +0,40 | +0,41 | +0,35 | +0,37 | +0,07 (e.s. 0,03) | +0,13 (e.s. 0,02) |
| ACC_COND meno… | +0,43 | +0,45 | +0,39 | +0,41 | +0,10 (e.s. 0,03) | +0,16 (e.s. 0,02) |

ACC_COND contro il concorrente migliore di ogni popolazione (sempre ML2S): gaussiana +0,10 (e.s. 0,04),
t3 +0,03 (e.s. 0,04), gruppo isolato +0,34 (e.s. 0,05).

### 13b. Dove vince e dove perde

- **Coda e gruppo isolato:** acc domina. Coda gaussiana con τ = 1,5: ML2S 32%, ACC60 72%. Gruppo isolato con τ = 1,5:
  ML2S 40%, ACC_COND 100%. È il campo per cui il braccio di copertura e la stratificazione sono pensati.
- **Nucleo denso:** vince ML2S, il campionamento casuale. Gaussiana con τ = 1,5: ML2S 82%, ACC60 72%.
  t3 con τ = 3: ML2S 98%, ACC_COND 65%. La copertura sottrae unità dove la popolazione è densa.
- **Regione rara:** nessun metodo supera il 57%. Su t3 ML2S è il migliore (45% con τ = 3, acc 22–25%).
- **Metodi classici** (VT, ENRICH, GATES) restano tra il 19% e il 29%, poco sopra l'esperimento unico. Motivi plausibili:
  - ENRICH usa soglie su una sola covariata, con quota minima del 5%: diluisce regioni dell'1% e non coglie regioni sferiche;
  - VT ha foglie da almeno 30 unità su 600 (circa il 5% della popolazione);
  - GATES testa il quinto superiore, quindi il 20%, contro regioni del 1–5%.
  Sono pensati per sottogruppi ampi, tipicamente definiti da biomarcatori, non per regioni piccole.

### 13c. La v0.2 pubblicata non batte il confronto più semplice

ACC40 (0,48) sta sotto ML2S (0,54): con la configurazione pubblicata, il campionamento guidato della fase 1 non compensa
il budget di esplorazione ridotto. È la ripartizione al 60% (+0,13) e poi la stratificazione (+0,03) a portare acc sopra
tutti i concorrenti.

### 13d. Localizzazione, distorsione, validità

- **Precisione del gruppo testato:** acc più precisa di tutti nella coda e nel gruppo isolato (ACC_COND 0,49 contro 0,11–0,26
  di ML2S); ML2S più precisa nel nucleo denso.
- **Distorsione della stima quando il test rifiuta:** 0,01–0,07 per tutti, tranne GATES (0,23), che testa gruppi di sole 100 unità.
  È l'effetto del filtro di significatività: condizionando al rifiuto, i gruppi piccoli sovrastimano di più.
- **Copertura degli IC:** 92–98%.
- **Errore di I tipo:** quasi tutto compatibile con il 5%. Da controllare: ACC_COND sul mix al 12,5% (5 su 40). Unito al 7,5%
  del punto 12 fa 8 su 80, il 10%, a circa 2 errori standard dal 5%. Il disegno è valido per costruzione (test su strato fissato
  dalle covariate e su unità nuove, soglia divisa), ma va verificato con più repliche.

### 13e. Rumore

Nella popolazione gaussiana ACC_COND e ACC60 sono lo stesso algoritmo con semi diversi: media identica (0,633 contro 0,633),
ma nella regione rara con τ = 3 una cella dà 18% e l'altra 57%. Le singole celle della regione rara oscillano molto più
dell'errore binomiale nominale (6–8 punti). Vanno lette le medie, non le celle isolate.

### Valore di ricerca: **alto**

È il risultato che serve per presentare il metodo: acc, con le modifiche dei punti 9 e 12, supera i metodi classici di 35–45
punti e il due stadi generico con ML di 7–10. Mostra anche, onestamente, che la configurazione pubblicata non basta, e che nel
nucleo denso il campionamento casuale resta preferibile.
Implementazioni semplificate: un confronto definitivo richiederebbe le versioni originali (per esempio GATES con divisioni
ripetute, causal forest onesta, SIDES).

---

## 14. Falsi positivi della stratificazione — **risolto: era rumore**

Verifica del 19 novembre 2026 (`experiments/dev_null_strat.py`). Nei punti 12 e 13 la versione stratificata sul mix aveva
rifiutato in 8 casi su 80 senza effetto (10%). Sotto un tasso vero del 5%, almeno 8 su 80 capita con probabilità 4,7%;
su una trentina di celle nulle controllate in questi studi, un caso così è atteso.

**Metodo.** Scenario nullo, popolazioni mix e t3, 200 repliche ciascuna (semi 3.200–3.399, non di valutazione).
Per ogni replica la fase 1 del corpo principale viene eseguita davvero, poi la conferma viene ripetuta 10 volte; il test sullo
strato esterno viene ripetuto 50 volte. Totale: 2.000 test sul corpo principale e 10.000 sullo strato esterno per popolazione,
con errore standard calcolato per replica. Ogni test è a α/2 = 2,5%.

| Popolazione | Test | Variante | Rifiuto | e.s. |
|---|---|---|---|---|
| mix | corpo principale | HC1 | 2,30% | 0,35 |
| mix | strato esterno | HC1 (attuale) | 2,82% | 0,16 |
| mix | strato esterno | HC3 | 2,48% | 0,16 |
| mix | strato esterno | jitter + HC1 | 2,69% | 0,15 |
| mix | strato esterno | jitter + HC3 | 2,30% | 0,14 |
| t3 | corpo principale | HC1 | 1,90% | 0,30 |
| t3 | strato esterno | HC1 (attuale) | 2,68% | 0,16 |
| t3 | strato esterno | HC3 | 2,43% | 0,15 |
| t3 | strato esterno | jitter + HC1 | 2,60% | 0,17 |
| t3 | strato esterno | jitter + HC3 | 2,29% | 0,16 |

**Tasso globale** (rifiuto se almeno uno dei due test rifiuta):

| Popolazione | Con HC1 (attuale) | Con HC3 nello strato esterno |
|---|---|---|
| mix | 5,06% | 4,72% |
| t3 | 4,53% | 4,28% |

**Conclusioni.**
- La versione stratificata controlla l'errore di I tipo al 5%: il 10% dei punti 12 e 13 era una fluttuazione.
- Il test sullo strato esterno con HC1 è appena troppo permissivo (2,7–2,8% invece di 2,5%, circa 2 errori standard),
  come atteso con pochi punti ad alta leva. Con HC3 torna esattamente al livello nominale. L'effetto sul totale è trascurabile,
  ma HC3 non costa nulla: è la scelta consigliata per lo strato esterno.
- Aggiunta l'opzione `acc.ols_effect(..., se_type="hc3")`. Il default resta HC1, quindi tutti i risultati precedenti si riproducono
  invariati. 18 test superati.

### Valore di ricerca: **medio**

Chiude il dubbio di validità sulla stratificazione e indica una piccola correzione prudente.

---

## 15. Valutazione del disegno centrale sui semi riservati — **acc v0.3 il più potente; H1 non superata, probabilmente per il criterio**

Valutazione del 19 novembre 2026 (`experiments/eval_central.py`, analisi `experiments/analyze_eval.py`). Codice congelato prima
dell'esecuzione: tag `v0.3-dev-frozen` (configurazione), `v0.3-dev-eval` (script); anche l'analisi è stata registrata nel repository
prima di vedere i risultati. Deviazioni dal piano nel `CHANGELOG.md`.

**Disegno** (piano, sez. 3): d = 5, covariate gaussiane, N = 10^5, B = 1.000, 13 celle: nullo, omogeneo, compatta_5, palla_rara_1,
coda_1, multi_2x2, gradiente, con τ = 1,5 e 3. Repliche: 1.000 per il nullo, 300 per le altre. Semi 10.000–22.999, mai usati prima.
Metodi del piano (B1, B3, B4, P_rand, P_l1, P_mix, P_adapt_v02, B5) con n1/B = 0,4; `ACC_v03` = `pipeline_v03` (n1/B = 0,6) al posto
di P_adapt_v03; fuori piano `P_rand_60`. Nel disegno gaussiano la stratificazione non si è mai attivata (0 su 4.600), come previsto:
qui ACC_v03 coincide con P_adapt_v02 al 60%.

### 15a. Potenza

| Metodo | Potenza media (12 celle con effetto) | Costo per scoperta |
|---|---|---|
| B1, esperimento unico | 0,52 | 1.931 |
| B3 | 0,50 | 2.007 |
| B4, enrichment classico | 0,50 | 2.008 |
| P_rand | 0,68 | 1.463 |
| P_l1 | 0,59 | 1.710 |
| P_mix | 0,69 | 1.454 |
| P_adapt_v02 (pubblicata) | 0,71 | 1.413 |
| P_rand_60 (fuori piano) | 0,74 | 1.361 |
| **ACC_v03** | **0,75** | **1.327** |
| B5, oracolo | 1,00 | 1.000 |

Le celle omogeneo e gradiente sono al 100% per tutti (anche B1): le altre 8 celle fanno le differenze. Errore Monte Carlo per cella ≤ 2,9 punti.

Differenze appaiate sulle 12 celle con effetto:
- ACC_v03 − P_adapt_v02: **+0,046** (e.s. 0,008): il guadagno della v0.3 sulla versione pubblicata, sui semi riservati.
- ACC_v03 − P_rand_60: **+0,019** (e.s. 0,008), ma a favore solo in 2 celle su 12: tutto il vantaggio viene dalla coda
  (coda_1: 64% contro 21% con τ = 1,5; 98% contro 59% con τ = 3). P_rand_60 è migliore in compatta_5, multi_2x2 e palla_rara_1.
- ACC_v03 − B1: +0,24; ACC_v03 − B4: +0,26.

Regione rara (palla_rara_1): ACC_v03 14% e 38%; P_rand_60 15% e 54%; nessun metodo non oracolo supera il 54%.

### 15b. Ipotesi del piano

| Ipotesi | Criterio | P_adapt_v02 | ACC_v03 |
|---|---|---|---|
| H1 Validità | limite sup. IC 95% del rifiuto nel nullo ≤ 7% | 5,3% [4,0; 6,9] **sì** | 6,2% [4,8; 7,9] **no** |
| H2 Copertura | 93–97% in ogni cella | 3 celle fuori su 13 **no** | 1 cella fuori (97,3%) **no** |
| H3 Focalizzazione | potenza > B1 in ≥ 80% delle celle con effetto | 8/12 **no** | 8/12 **no** |
| H4 Adattività | regret massimo < P_rand, P_l1, P_mix | 0,22 **sì** | 0,11 **sì** |

Altri metodi nel nullo: B1 4,2%, B3 5,5%, B4 3,0%, P_rand 5,6%, P_l1 4,7%, P_mix 4,6%, P_rand_60 5,9%, B5 (oracolo) 6,5%.

### 15c. Lettura delle ipotesi

- **H1.** Il criterio fallisce già con 55 rifiuti su 1.000: un metodo perfettamente calibrato al 5% lo fallisce nel 25% dei casi, e con
  7 metodi P più l'oracolo la probabilità che almeno uno fallisca è dell'87%. Fallisce anche B5, che è un semplice esperimento casuale
  con lo stesso test. Analisi a posteriori della causa (`experiments/dev_final_test.py`, semi di sviluppo): il test finale da solo è
  calibrato, 4,94% su 240.000 test (5,0% con n = 400 e 5,2% con n = 600). Gli studi di sviluppo sulla stessa pipeline davano 4,86% (punto 1)
  e 5,06% (punto 14). Sotto un tasso vero del 5%, 62 rifiuti su 1.000 hanno probabilità 5,1%. L'aggregato dei metodi P e di B5 (5,54% su
  7.000) ha probabilità 2,1%: un lieve eccesso comune non si può escludere del tutto, ma riguarda anche l'oracolo e non acc in particolare.
  Secondo la regola del piano (sez. 7) ACC_v03 **non si presenta come valido** finché una verifica dedicata non lo conferma.
- **H2.** Con 300 repliche l'errore sulla copertura è 1,3 punti, quindi 93–97% è ±1,6 errori standard: anche con IC perfetti ogni cella
  esce nell'11% dei casi, e su 13 celle se ne attendono circa 1,4 fuori. ACC_v03 ne ha 1 (97,3%, quindi prudente), gli altri metodi P 3–4.
  La copertura media è del 95–96% per tutti: gli IC sono corretti; il criterio è troppo stretto per 300 repliche.
- **H3.** Fallisce per pareggi al 100% (omogeneo e gradiente, 4 celle) con il criterio "maggiore stretto". In tutte le 8 celle dove la
  potenza non è al massimo, i metodi focalizzati battono B1. Il criterio non prevedeva i pareggi.
- **H4.** Superata da entrambi; ACC_v03 dimezza il regret massimo della v0.2 (0,11 contro 0,22) e ha regret medio 0,02. Nota: l'oracolo
  del piano usa P_rand e P_l1 al 40%, mentre ACC_v03 usa il 60%, quindi il confronto lo favorisce.

### 15d. Altre metriche

- Distorsione media della stima nel gruppo testato: tra −0,005 e +0,003 per tutti i metodi.
- Localizzazione: ACC_v03 ha la precisione di R̂ più alta o vicina alla più alta nella coda (0,17–0,29) e nel nucleo denso (0,20–0,46);
  P_rand_60 è più preciso nel nucleo denso, in multi_2x2 e nella regione rara.
- Tempo: ACC_v03 3,0 s per esecuzione, P_rand_60 0,5 s.

### Valore di ricerca: **alto**

Conferma sui semi riservati i risultati di sviluppo: la v0.3 è la più potente (+4,6 punti sulla versione pubblicata, +23,6 sull'esperimento
unico) e ha il regret massimo più basso. Il suo vantaggio sul due stadi casuale al 60% è piccolo e concentrato nella coda.
Il piano congelato aveva criteri H1–H3 troppo stretti per il numero di repliche previsto: è un risultato da segnalare all'autore.
Prima di dichiarare la validità serve un nullo dedicato con più repliche su semi riservati nuovi.

---

## Sintesi

| # | Tentativo | Esito | Valore |
|---|---|---|---|
| 1 | Verifica dell'errore di I tipo | Risolto: 4,86% | Alto |
| 2 | Randomizzazione dell'assegnazione | Difetto trovato e corretto | Alto |
| 3 | Regole di attribuzione del piano | Nessun miglioramento | Medio |
| 4 | Focus sulla regione candidata | Nessun miglioramento | Medio |
| 5 | Dimensione di R̂ | Nessun effetto | Basso–medio |
| 6 | Diagnosi: rilevamento in fase 1 | Collo di bottiglia individuato | Alto |
| 7a | Rarità per proiezioni | Non batte il mix | Medio |
| 7b | Zoom per celle | Non batte il mix | Medio |
| 8 | v0.2 con code pesanti | Diagnostica satura; la copertura insegue gli outlier e il vantaggio sul casuale sparisce | Alto |
| 9 | Più budget alla fase 1 (60%) | Potenza adattivo da 0,57 a 0,65 | Alto |
| 10 | Fase 1 a lunghezza variabile | Nessun guadagno; la fase 1 non riconosce il proprio successo; l'ottimo fisso è almeno all'80% | Medio-alto |
| 11 | Più regioni candidate in fase 2 | La regione rara è catturata fino al doppio delle volte, ma resta diluita; potenza invariata o peggiore | Alto |
| 12 | Strato raro nelle covariate isolato e studiato a parte | Gruppo isolato da 78% a 98% (τ = 1,5); code pesanti +5 punti; costa sulle gaussiane, ma una diagnostica decide quando attivarlo | Alto |
| 13 | Confronto con metodi esistenti | acc migliorato 0,65 contro 0,20–0,26 dei metodi classici e 0,54 del due stadi con ML; la v0.2 pubblicata (0,48) non batte quest'ultimo | Alto |
| 14 | Falsi positivi della stratificazione | 5,06% e 4,53% su 12.000 test per popolazione: era rumore; HC3 consigliato nello strato esterno | Medio |
| 15 | Valutazione del disegno centrale sui semi riservati | v0.3 la più potente (0,75 contro 0,71 della v0.2); H4 superata; H1–H3 falliscono per criteri troppo stretti; validità da confermare | Alto |

Sulla potenza i passi avanti rispetto alla v0.2 pubblicata sono due: la ripartizione del budget (9), con il 60% alla fase 1 l'adattivo passa da 0,57 a 0,65; e la stratificazione condizionata (12), che isola e studia a parte i gruppi rari nelle covariate. Con entrambe acc supera tutti i metodi concorrenti provati (13), cosa che la v0.2 pubblicata non fa; la valutazione sui semi riservati lo conferma (15), con un vantaggio piccolo sul due stadi casuale al 60%. Gli altri passi avanti sono sulla validità (1, 2) e sulla comprensione del metodo (6, 8). Il punto 8 mostra che con covariate a code pesanti il metodo perde il suo vantaggio.

## Idee non ancora provate

0. **Nullo di conferma** per H1 (15c): almeno 5.000 repliche dello scenario nullo su semi riservati nuovi, per ACC_v03 e B5.
   **Criteri del piano**: proporre all'autore criteri H1–H3 coerenti con il numero di repliche (per esempio H1 sul limite superiore
   con più repliche, H2 sulla copertura media o con correzione per il numero di celle, H3 con "maggiore o uguale" o escludendo le celle al 100%).

1. **Ripartizione oltre l'80%** e conferma del risultato con il metodo adattivo v0.2 (vedi 9 e 10). La fase 1 a lunghezza variabile è stata provata (10) e non aiuta.
   **Più regioni candidate in fase 2**: provato (11). Cattura la regione rara più spesso, ma non aumenta la potenza.
   **Zoom nella candidata scelta**: dopo lo screening, dividere la candidata migliore in sottoregioni più piccole (per esempio
   1%) e confermare solo la più promettente, su unità nuove. Affronta la diluizione emersa al punto 11c. Va dosato il budget
   tra valutazione e conferma, perché ogni livello di zoom ne consuma una parte.
2. **Stimatore di τ(x) più sensibile con pochi dati.** Oggi è una foresta con foglie da 10 unità su 400 osservazioni. È la leva più diretta sul collo di bottiglia individuato al punto 6.
3. **Proiezioni dentro il mix.** Sostituire la copertura λ = 1 con `select_projected` nel mix 50/50.
4. **Informazione esterna.** Dati storici, oppure un esito indiretto meno costoso, per orientare la fase 1.
5. **Robustezza alle code pesanti** (vedi 8e): in parte affrontata dalla stratificazione (12). Restano da provare la standardizzazione robusta dentro il corpo principale e il modello dell'esito con componente lineare.
   **Stratificazione condizionata** (12c): verificare la diagnostica su altre forme di popolazione (asimmetriche, discrete, con più gruppi isolati) e studiare lo strato esterno con una ricerca della regione invece di un solo test sull'effetto medio.
6. **Banco di prova più ampio.** Più di 5 covariate, covariate asimmetriche o discrete, effetti sfumati o su più regioni.
7. **Confronti con metodi concorrenti**: fatti in forma semplificata (13). Restano le versioni originali (GATES con divisioni ripetute, causal forest onesta, SIDES). La verifica dell'errore di I tipo della stratificazione è fatta (14).
8. **Nucleo denso:** ridurre la quota di copertura quando la fase 1 suggerisce un effetto nella zona densa, l'unico caso in cui il campionamento casuale batte acc (13b).

## File prodotti

Tutto il materiale è in `acc_proposta_v03dev.zip`; la patch del solo codice è in `acc_proposta_v03dev.patch`.

- Codice: `acc/core.py` (parametri `jitter`, `rule`, `focus`; funzioni `select_projected`, `zoom_phase1`, `sequential_phase1`, `heterogeneity_t`, `outer_stratum`, `outer_ratio`) e `acc/__init__.py`.
- Test: `tests/test_core.py`, 19 test, tutti superati.
- Script: `experiments/dev_null_check.py`, `dev_rules.py`, `dev_rho.py`, `dev_explore.py`, `analyze_dev.py`, `study_tails.py`, `analyze_tails.py`, `study_budget.py`, `analyze_budget.py`, `study_seqlen.py`, `analyze_seqlen.py`, `study_multicand.py`, `analyze_multicand.py`, `study_strat.py`, `analyze_strat.py`, `study_competitors.py`, `analyze_competitors.py`, `dev_null_strat.py`, `eval_central.py`, `analyze_eval.py`, `dev_final_test.py`.
- Risultati: `results/dev_*_results.csv` e `results/dev_*_summary.csv`.
- Le stesse informazioni in forma breve sono nel `CHANGELOG.md`, nella sezione "Non rilasciato".
