# acc 0.3.0 — Sintesi dei risultati e programma per la 0.4

20 novembre 2026. Questo documento riassume cosa cambia nella 0.3.0, che cosa è stato dimostrato, che cosa no, e che cosa
resta da fare. Ogni affermazione rimanda a una sezione di `docs/registro_tentativi.md` (il registro completo di tutti i
tentativi, positivi e negativi) e a script e risultati in `experiments/` e `results/`.

## 1. In una frase

La 0.3.0 è una v0.2 ricalibrata e validata meglio: più potenza, falsi positivi confermati al livello nominale e minore perdita
nel caso peggiore rispetto alle strategie pure. Introduce in forma sperimentale la stratificazione dei gruppi rari nelle
covariate. Nel disegno gaussiano il guadagno viene dalla ripartizione del budget, e la regola adattiva equivale a un mix fisso.

## 2. Che cosa cambia

**Configurazione consigliata** `acc.pipeline_v03`, congelata prima della valutazione (tag `v0.3-dev-frozen`):

- **Fase 1 al 60% del budget** (era il 40%). Più esplorazione aumenta la probabilità di trovare la regione; una volta trovata,
  400 unità di conferma bastano (registro, punti 6, 9 e 10).
- **Stratificazione condizionata.** Se una diagnostica sulle sole covariate (`outer_ratio`) supera di oltre il 10% il valore
  atteso per una normale, il 2% più esterno della popolazione (`outer_stratum`, distanza robusta) viene separato all'inizio,
  studiato con il 16% del budget e un test proprio (HC3), e riunito alla fine con la soglia divisa (punti 12 e 14).
- Fase 2 invariata.

**Difetto della v0.2 documentato**: l'assegnazione trattamento/controllo con `anticluster` è deterministica data X. Con
`jitter = 0,05` diventa una randomizzazione vera senza perdere bilanciamento (punto 2). L'opzione è disponibile ma non fa parte
della configurazione congelata.

Tutte le funzioni esistenti mantengono il comportamento della v0.2 con i parametri di default (verificato sul pilota 2).

## 3. Che cosa è stato dimostrato

Valutazione sui semi riservati (da 10.000 in su, mai usati durante lo sviluppo) con protocollo preregistrato: configurazione,
script e analisi salvati nel repository con un'etichetta di versione prima di ogni esecuzione.

| Risultato | Valore | Registro |
|---|---|---|
| Potenza media, disegno centrale (12 celle con effetto) | 0,753 contro 0,708 della v0.2; differenza appaiata +0,046 (e.s. 0,008) | 15 |
| Errore di I tipo, conferma su 5.000 repliche nulle | 4,58% (IC 95% 4,02–5,20%) | 16 |
| Regret massimo a parità di budget (tutte le strategie al 60%) | 0,18 (IC 0,14–0,25) contro 0,63 di casuale e copertura | 17 |
| Confronto con metodi esistenti (sviluppo) | 0,65 contro 0,20–0,26 di GATES, virtual twins, arricchimento a soglia; 0,54 del due stadi casuale | 13 |
| Gruppo isolato dell'1% nelle covariate (sviluppo) | 98% contro 78% della v0.2 con effetto debole | 12 |

Sull'errore di I tipo vanno riportati entrambi i risultati: 6,2% sulle prime 1.000 repliche (sopra la soglia H1 del piano) e
4,58% sulla conferma preregistrata, cinque volte più grande. Il piano (sez. 7) prevede l'analisi della causa in caso di
fallimento: il test finale da solo è calibrato (4,94% su 240.000 test) e la stessa soglia avrebbe fatto fallire anche l'oracolo
B5 (6,5%). La v0.2 ha superato H1 solo su 1.000 repliche (5,3%) e non è mai stata confermata.

Ipotesi del piano per la 0.3.0: H1 superata (con conferma), H4 superata. H2 e H3 non superate come scritte, per motivi di criterio
(punto 15c): con 300 repliche il criterio 93–97% per ogni cella fallisce in media 1,4 celle anche con intervalli perfetti (la
copertura media è 95–96%); H3 richiede "maggiore stretto" e quattro celle sono al 100% per tutti i metodi.

## 4. Che cosa non è stato dimostrato

- **Che l'adattività aggiunga valore.** A parità di budget ACC e un mix fisso 50/50 hanno lo stesso regret massimo (differenza
  −0,01, IC −0,08–+0,06) e la stessa potenza (differenza +0,006, e.s. 0,008). Nessuna delle regole adattive provate, comprese le
  tre previste dal piano, batte la v0.2 (punti 3, 4, 10, 17).
- **Che la stratificazione funzioni sui semi riservati.** È stata validata solo su semi di sviluppo e su tre forme di
  popolazione (normale, t di Student, gruppo isolato). Nel disegno centrale gaussiano non si attiva mai.
- **Le regioni rare nell'effetto.** Con B = 1.000 nessun metodo, concorrenti compresi, supera circa il 55% di potenza su una
  regione dell'1% (punti 7, 11, 15).
- **Dimensioni maggiori e covariate non continue**: tutto è valutato con d = 5.
- I concorrenti sono implementati in forma semplificata (per esempio GATES con una sola divisione del campione).

## 5. Che cosa si è imparato (anche dai risultati negativi)

- **Il collo di bottiglia è il rilevamento in fase 1.** La potenza è quasi tutto-o-niente: se R̂ contiene almeno il 5% di regione
  vera il test rifiuta in circa il 90% dei casi, altrimenti in circa il 10% (punto 6).
- **La fase 1 non riconosce il proprio successo.** Un test di eterogeneità sulle sue unità diluisce effetti locali; la fase 2,
  che concentra unità nuove nella regione candidata, è l'unico controllo sensibile (punto 10).
- **Con code pesanti la copertura insegue gli outlier** e la stima di τ(x) si rompe; la diagnostica della v0.2 satura a λ0 = 0,8
  (punto 8).
- **Più regioni candidate catturano più spesso la regione rara, ma la diluiscono** (punto 11).
- **Il campionamento casuale resta il migliore nelle regioni dense** (punti 13 e 17).

## 6. Programma per la 0.4

1. **Valutare la stratificazione sui semi riservati** su popolazioni non gaussiane (t di Student, gruppo isolato, covariate
   asimmetriche o discrete), con lo stesso protocollo del regret a parità di budget: è l'unico punto in cui ACC fa qualcosa di
   diverso da un mix fisso.
2. **Decidere sull'adattività**: o una regola che si sposti davvero verso la strategia giusta, con un segnale di fase 1 più forte;
   o sostituire la regola con il mix fisso 50/50, più semplice e circa tre volte più veloce, a parità di risultati.
3. **Randomizzazione vera** dell'assegnazione (`jitter = 0,05`) nella configurazione di default, con verifica dell'errore di I tipo.
4. **Revisione dei criteri H2 e H3** del piano: copertura media o correzione per il numero di celle; "maggiore o uguale", oppure
   esclusione delle celle con potenza al 100%.
5. **Concorrenti nelle versioni originali** (GATES con divisioni ripetute, causal forest onesta, SIDES) e le variazioni del piano
   non ancora eseguite (d = 20 e 100, budget 500 e 2.000).

## 7. Riproducibilità

Tutti gli script sono deterministici dati i semi. Le etichette git `v0.3-dev-frozen`, `v0.3-dev-eval`, `v0.3-dev-null-confirm` e
`v0.3-dev-regret60` indicano lo stato esatto del codice prima di ciascuna esecuzione sui semi riservati. I comandi sono nel README.

Semi usati: pilota 0–99; sviluppo 1.000–9.039 (sempre sotto 10.000); valutazione 10.000–22.999; conferma del nullo 30.000–34.999;
regret a parità di budget 41.000–52.299.
