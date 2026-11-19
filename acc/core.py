"""
acc.core (v0.2) - Prototipo "Anticlustering-Clustering per la focalizzazione".

Componenti:
  anticluster()        Fase 0 e bilanciamento dei bracci: partizione in K gruppi
                       massimamente eterogenei (euristica assignment-based, stile ABA).
  select_hybrid()      Fase 1: selezione di S1 che bilancia rappresentativita'
                       (k-means -> punti piu' vicini ai centroidi) e copertura
                       (k-center greedy), regolata da lambda.
  fit_cate()           Stima di tau(x) con un T-learner (ExtraTrees).
  define_region()      Fase 2: regione candidata R_hat = {x : tau_hat(x) > delta}.
  ols_effect()         Stima dell'effetto con aggiustamento per covariate e SE robusti (HC1).
  coverage_radius(), mmd()  Le due componenti dell'obiettivo J (diagnostica).

Solo numpy, scipy, scikit-learn.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import pairwise_distances_argmin_min


# ---------------------------------------------------------------- utilita'
def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / X.std(0)


# ---------------------------------------------------------------- anticlustering
def anticluster(X: np.ndarray, K: int, rng: np.random.Generator | None = None,
                jitter: float = 0.0) -> np.ndarray:
    """Partiziona le righe di X in K gruppi di uguale dimensione (+-1), simili tra loro.

    Euristica assignment-based (idea di Baumann et al. 2026, ABA):
    gli oggetti, ordinati per distanza decrescente dal centroide globale, sono
    processati a blocchi di K; ogni blocco e' assegnato ai K gruppi (uno per
    gruppo) risolvendo un problema di assegnamento che massimizza la distanza
    di ciascun oggetto dal centroide corrente del gruppo che lo riceve.
    Complessita' ~ O(N log N + N K^2), adatta a N nell'ordine dei milioni con K piccolo.

    jitter: con jitter = 0 (default, comportamento v0.2) la partizione e' una funzione
    deterministica di X: cambia solo quale gruppo riceve quale etichetta. Con jitter > 0
    l'ordine di elaborazione e' perturbato con rumore gaussiano di deviazione standard
    jitter * sd(dist): esecuzioni diverse danno partizioni diverse, tutte bilanciate.
    Serve quando la partizione e' usata come assegnazione trattamento/controllo e si
    vuole una randomizzazione vera (jitter = 0.05 basta, con bilanciamento quasi invariato).
    """
    rng = rng or np.random.default_rng()
    Z = standardize(X)
    n, d = Z.shape
    dist = ((Z - Z.mean(0)) ** 2).sum(1)
    # piccolo rumore per rompere i pareggi
    key = dist + 1e-9 * rng.random(n)
    if jitter > 0:
        key = key + jitter * dist.std() * rng.standard_normal(n)
    order = np.argsort(-key)
    labels = np.empty(n, dtype=int)
    sums = np.zeros((K, d))
    counts = np.zeros(K)
    for start in range(0, n, K):
        idx = order[start:start + K]
        m = len(idx)
        if start == 0:
            g = rng.permutation(K)[:m]
        else:
            cent = sums / np.maximum(counts, 1)[:, None]
            cost = -((Z[idx, None, :] - cent[None, :, :]) ** 2).sum(2)  # m x K
            if m < K:  # ultimo blocco incompleto: solo i gruppi piu' piccoli
                allowed = np.argsort(counts, kind="stable")[:m]
                r, c = linear_sum_assignment(cost[:, allowed])
                g = np.empty(m, dtype=int)
                g[r] = allowed[c]
            else:
                r, c = linear_sum_assignment(cost)
                g = np.empty(m, dtype=int)
                g[r] = c
        labels[idx] = g
        np.add.at(sums, g, Z[idx])
        np.add.at(counts, g, 1)
    return labels


def balance_report(X: np.ndarray, labels: np.ndarray) -> dict:
    """Max |SMD| delle medie e max |log rapporto di varianze| tra i gruppi e la popolazione."""
    Z = standardize(X)
    smd, lvr = 0.0, 0.0
    for g in np.unique(labels):
        Zg = Z[labels == g]
        smd = max(smd, np.abs(Zg.mean(0)).max())
        lvr = max(lvr, np.abs(np.log(Zg.var(0))).max())
    return {"max_abs_smd": smd, "max_abs_log_var_ratio": lvr}


# ---------------------------------------------------------------- fase 1
def select_representative(Z: np.ndarray, n: int, rng: np.random.Generator,
                          exclude: np.ndarray | None = None) -> np.ndarray:
    """n punti che seguono la densita' della popolazione: k-means con n centri,
    poi il punto reale piu' vicino a ciascun centroide (duplicati sostituiti a caso)."""
    if n <= 0:
        return np.array([], dtype=int)
    pool = np.arange(len(Z)) if exclude is None else np.setdiff1d(np.arange(len(Z)), exclude)
    km = MiniBatchKMeans(n_clusters=n, batch_size=4096, n_init=1, max_iter=50,
                         random_state=int(rng.integers(1 << 31))).fit(Z[pool])
    nearest, _ = pairwise_distances_argmin_min(km.cluster_centers_, Z[pool])
    chosen = np.unique(pool[nearest])
    if len(chosen) < n:
        rest = np.setdiff1d(pool, chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, n - len(chosen), replace=False)])
    return chosen


def select_kcenter(Z: np.ndarray, n: int, rng: np.random.Generator,
                   start: np.ndarray | None = None) -> np.ndarray:
    """k-center greedy (farthest-point): aggiunge n punti, ognuno il piu' lontano
    dall'insieme gia' scelto (che parte da `start`, se dato)."""
    if n <= 0:
        return np.array([], dtype=int)
    chosen = [] if start is None else list(start)
    if chosen:
        dmin = np.full(len(Z), np.inf)
        for s in chosen:
            dmin = np.minimum(dmin, ((Z - Z[s]) ** 2).sum(1))
    else:
        first = int(rng.integers(len(Z)))
        chosen.append(first)
        dmin = ((Z - Z[first]) ** 2).sum(1)
        n -= 1
    new = []
    for _ in range(n):
        j = int(np.argmax(dmin))
        new.append(j)
        dmin = np.minimum(dmin, ((Z - Z[j]) ** 2).sum(1))
    out = np.array(chosen + new if start is None else new, dtype=int)
    return np.concatenate([start, out]) if start is not None else out


def select_hybrid(Z: np.ndarray, n: int, lam: float, rng: np.random.Generator) -> np.ndarray:
    """Fase 1: (1-lam)*n punti rappresentativi, poi lam*n punti k-center che
    riempiono i vuoti lasciati dai primi (code e regioni rare)."""
    n_cov = int(round(lam * n))
    rep = select_representative(Z, n - n_cov, rng)
    if n_cov == 0:
        return rep
    return select_kcenter(Z, n_cov, rng, start=rep if len(rep) else None)


def knn_radius(Z: np.ndarray, rng: np.random.Generator, k: int = 20, m: int = 10_000) -> np.ndarray:
    """Distanza dal k-esimo vicino (in un sottocampione di m punti): proxy di p(x)^(-1/d)."""
    from sklearn.neighbors import NearestNeighbors
    probe = rng.choice(len(Z), min(len(Z), m), replace=False)
    return NearestNeighbors(n_neighbors=k).fit(Z[probe]).kneighbors(Z)[0][:, -1]


def select_tempered(Z: np.ndarray, n: int, lam: float, rng: np.random.Generator,
                    rk: np.ndarray | None = None, clip: float = 0.99) -> np.ndarray:
    """Fase 1, variante "densita' temperata": S1 rappresenta la distribuzione
    q(x) proporzionale a p(x)^(1-lam), con p stimata dalla distanza dal k-esimo vicino.
    lam=0 -> copia della popolazione; lam=1 -> copertura ~uniforme del supporto.
    Implementazione: k-means pesato (pesi p^-lam, troncati al quantile `clip`)
    e punto reale piu' vicino a ogni centroide."""
    d = Z.shape[1]
    if rk is None:
        rk = knn_radius(Z, rng)
    logw = lam * d * np.log(rk + 1e-12)          # p^-lam  ~  r_k^(d*lam)
    logw = np.minimum(logw, np.quantile(logw, clip))
    w = np.exp(logw - logw.max())
    km = MiniBatchKMeans(n_clusters=n, batch_size=4096, n_init=1, max_iter=50,
                         random_state=int(rng.integers(1 << 31))).fit(Z, sample_weight=w)
    nearest, _ = pairwise_distances_argmin_min(km.cluster_centers_, Z)
    chosen = np.unique(nearest)
    if len(chosen) < n:
        rest = np.setdiff1d(np.arange(len(Z)), chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, n - len(chosen), replace=False,
                                                    p=w[rest] / w[rest].sum())])
    return chosen


def coverage_radius(Z: np.ndarray, S: np.ndarray, m: int = 5000, rng=None) -> float:
    """max_x min_s d(x, s), stimato su un sottocampione di m punti."""
    rng = rng or np.random.default_rng(0)
    probe = rng.choice(len(Z), min(m, len(Z)), replace=False)
    _, dist = pairwise_distances_argmin_min(Z[probe], Z[S])
    return float(dist.max())


def mmd(Z: np.ndarray, S: np.ndarray, m: int = 2000, rng=None) -> float:
    """MMD^2 con kernel RBF (bandwidth = mediana) tra S e un sottocampione di P."""
    rng = rng or np.random.default_rng(0)
    A = Z[rng.choice(len(Z), min(m, len(Z)), replace=False)]
    B = Z[S] if len(S) <= m else Z[rng.choice(S, m, replace=False)]
    def k(U, V, h):
        d2 = ((U[:, None, :] - V[None, :, :]) ** 2).sum(2)
        return np.exp(-d2 / (2 * h * h))
    h = np.sqrt(np.median(((A[:200, None] - A[None, :200]) ** 2).sum(2)) / 2)
    return float(k(A, A, h).mean() + k(B, B, h).mean() - 2 * k(A, B, h).mean())


# ---------------------------------------------------------------- fase 2
def fit_cate(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int = 0,
             learner: str = "dr", leaf: int = 20):
    """Stima di tau(x). Restituisce una funzione x -> tau_hat(x).

    learner="t":  T-learner (due foreste, una per braccio).
    learner="dr": pseudo-outcome con propensity nota (0.5, da randomizzazione):
                  m(x) = E[Y|x] con cross-fitting a 2 fold, psi = 2(2T-1)(Y - m(x)),
                  poi una foresta di psi su x. E[psi | x] = tau(x).
    """
    kw = dict(n_estimators=150, n_jobs=1, random_state=seed)
    if learner == "t":
        m1 = ExtraTreesRegressor(min_samples_leaf=5, **kw).fit(X[T == 1], Y[T == 1])
        m0 = ExtraTreesRegressor(min_samples_leaf=5, **kw).fit(X[T == 0], Y[T == 0])
        return lambda Xn: m1.predict(Xn) - m0.predict(Xn)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(Y)) % 2
    m = np.empty(len(Y))
    for f in (0, 1):
        mf = ExtraTreesRegressor(min_samples_leaf=10, **kw).fit(X[fold != f], Y[fold != f])
        m[fold == f] = mf.predict(X[fold == f])
    psi = 2 * (2 * T - 1) * (Y - m)
    g = ExtraTreesRegressor(min_samples_leaf=leaf, **kw).fit(X, psi)
    return g.predict


def define_region(tau_hat: np.ndarray, delta: float) -> np.ndarray:
    """Regola (a): R_hat = {x : tau_hat(x) > delta}. Restituisce una maschera."""
    return tau_hat > delta


def ols_effect(X: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha: float = 0.05,
               se_type: str = "hc1") -> dict:
    """Y ~ 1 + T + X, SE robusti. Test unilaterale H0: tau <= 0.

    se_type="hc1" (default, v0.2) oppure "hc3": piu' prudente quando ci sono punti ad alta leva,
    per esempio nello strato esterno di `outer_stratum` (vedi experiments/dev_null_strat.py).
    """
    D = np.column_stack([np.ones(len(Y)), T, X])
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    e = Y - D @ beta
    n, p = D.shape
    bread = np.linalg.pinv(D.T @ D)
    if se_type == "hc3":
        h = np.einsum("ij,jk,ik->i", D, bread, D)
        meat = (D * (e ** 2 / np.clip(1 - h, 1e-6, None) ** 2)[:, None]).T @ D
        V = bread @ meat @ bread
    else:
        meat = (D * e[:, None] ** 2).T @ D
        V = bread @ meat @ bread * n / (n - p)
    est, se = float(beta[1]), float(np.sqrt(V[1, 1]))
    z = est / se
    return {"est": est, "se": se, "lo": est - norm.ppf(1 - alpha / 2) * se,
            "hi": est + norm.ppf(1 - alpha / 2) * se,
            "reject": bool(z > norm.ppf(1 - alpha))}


# ---------------------------------------------------------------- fase 1 adattiva
def population_diagnostic(Z: np.ndarray) -> dict:
    """Diagnosi della popolazione dalle sole covariate (prima di misurare esiti).

    tail_ratio: quota di elementi con almeno una |z| > 2.5, divisa per quella attesa
                sotto una normale con le stesse dimensioni (1 = code "normali").
    excess_kurtosis: curtosi in eccesso media delle covariate.
    lam0: lambda di partenza = 0.5 corretto verso la copertura se le code sono
          pesanti e verso la rappresentativita' se sono leggere, troncato in [0.2, 0.8].
    Non dice dove sta l'effetto: dice quanto e' rischioso affidarsi al solo
    campionamento proporzionale alla densita'.
    """
    from scipy.stats import kurtosis
    d = Z.shape[1]
    expected = 1 - (1 - 2 * norm.sf(2.5)) ** d
    tail_ratio = float((np.abs(Z) > 2.5).any(1).mean() / expected)
    exk = float(np.mean(kurtosis(Z, axis=0)))
    lam0 = float(np.clip(0.5 + 0.15 * np.log(tail_ratio) + 0.05 * exk, 0.2, 0.8))
    return {"tail_ratio": tail_ratio, "excess_kurtosis": exk, "lam0": lam0}


def _oof_parts(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int = 0, leaf: int = 10):
    """Pseudo-outcome DR e tau_hat fuori campione (2 fold) per le sole unita' misurate.

    psi_i = 2(2T_i - 1)(Y_i - m(x_i)), con m stimata sull'altro fold (come in fit_cate);
    tau_oof_i = previsione di una foresta di psi su x addestrata sull'altro fold.
    Serve alle regole di attribuzione che confrontano i bracci sulle unita' misurate:
    con tau_hat in campione ogni unita' "vota" per se stessa.
    """
    kw = dict(n_estimators=150, n_jobs=1, random_state=seed)
    fold = np.random.default_rng(seed).permutation(len(Y)) % 2
    m = np.empty(len(Y))
    for f in (0, 1):
        mf = ExtraTreesRegressor(min_samples_leaf=10, **kw).fit(X[fold != f], Y[fold != f])
        m[fold == f] = mf.predict(X[fold == f])
    psi = 2 * (2 * T - 1) * (Y - m)
    tau_oof = np.empty(len(Y))
    for f in (0, 1):
        gf = ExtraTreesRegressor(min_samples_leaf=leaf, **kw).fit(X[fold != f], psi[fold != f])
        tau_oof[fold == f] = gf.predict(X[fold == f])
    return psi, tau_oof


def _psi_within(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Pseudo-outcome DR con m(x) stimato a 2 fold usando solo queste unita'."""
    kw = dict(n_estimators=150, n_jobs=1, random_state=seed, min_samples_leaf=10)
    fold = np.random.default_rng(seed).permutation(len(Y)) % 2
    m = np.empty(len(Y))
    for f in (0, 1):
        m[fold == f] = ExtraTreesRegressor(**kw).fit(X[fold != f], Y[fold != f]).predict(X[fold == f])
    return 2 * (2 * T - 1) * (Y - m)


def heterogeneity_t(X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int = 0, leaf: int = 10) -> float:
    """Test onesto di eterogeneita' sulle unita' misurate.

    Le unita' sono divise in due meta' A e B (bilanciate per braccio). Su A si stima tau(x)
    (fit_cate); su B si calcola lo pseudo-outcome con un m(x) stimato solo su B e lo si regredisce
    su tau_A(x) (pendenza = quanto la stima fatta su A predice chi risponde di piu' in B).
    Si scambiano i ruoli e si combinano le due statistiche t: (t_AB + t_BA) / sqrt(2).
    Nessun esito di B entra nella stima usata per valutarlo. Sotto assenza di eterogeneita'
    la statistica e' circa N(0, 1).
    """
    g = np.random.default_rng([seed, 11])
    half = np.zeros(len(Y), int)
    for a in (0, 1):
        idx = np.flatnonzero(T == a)
        half[g.permutation(idx)[: len(idx) // 2]] = 1
    ts = []
    for h in (0, 1):
        A, Bm = half == h, half != h
        tau_A = fit_cate(X[A], T[A], Y[A], seed=seed, learner="dr", leaf=leaf)(X[Bm])
        psi_B = _psi_within(X[Bm], T[Bm], Y[Bm], seed=seed + 1)
        tc = tau_A - tau_A.mean()
        den = (tc ** 2).sum()
        if den <= 1e-12:
            ts.append(0.0)
            continue
        slope = (tc * psi_B).sum() / den
        resid = psi_B - psi_B.mean() - slope * tc
        se = np.sqrt((tc ** 2 * resid ** 2).sum()) / den
        ts.append(float(slope / se) if se > 0 else 0.0)
    return float(sum(ts) / np.sqrt(2))


ATTRIBUTION_RULES = ("v02", "density", "yield", "thompson", "none")


def adaptive_phase1(Z: np.ndarray, X: np.ndarray, measure, n1: int, rng: np.random.Generator,
                    lam0: float = 0.5, rounds: int = 4, rho: float = 0.03,
                    rk: np.ndarray | None = None, floor: float = 0.15, seed: int = 0,
                    rule: str = "v02", top_q: float = 0.10, focus: float = 0.0):
    """Fase 1 sequenziale con due "bracci" di selezione:
         rand = campionamento proporzionale alla densita' (rappresentativo)
         cov  = densita' temperata con lambda = 1 (copertura)
    Ciclo r: si estraggono n1/rounds unita', una quota `share` dal braccio cov.
    Dopo ogni ciclo (tranne l'ultimo): si stima tau_hat su tutte le unita' misurate,
    si prende la regione candidata (quota rho con tau_hat piu' alto) e si attribuisce
    ogni suo punto al braccio dell'unita' misurata piu' vicina, pesando per tau_hat.
    La quota di copertura del ciclo dopo = media tra la quota corrente e quella
    attribuita, con un minimo `floor` per braccio (esplorazione).

    measure(idx, T) -> Y  misura gli esiti (in simulazione: genera l'esito).
    Restituisce S1, T1, Y1, arm (0 = rand, 1 = cov, 2 = focus) e la storia delle quote.

    rule: regola di attribuzione (candidati della sez. 6 del piano di simulazione).
      "v02"      regola originale descritta sopra (default, comportamento invariato).
      "density"  come v02, ma l'attribuzione e' normalizzata per densita': si confronta la
                 quota attribuita a cov nella regione candidata (s_top) con la quota che cov
                 riceve sull'intera popolazione (b), cioe' quella attesa se la regione non
                 dicesse nulla sui bracci. Il rapporto di odds aggiorna la quota corrente,
                 che quindi resta ferma in assenza di segnale.
      "yield"    rendimento per estrazione: per ogni braccio, somma del pseudo-outcome DR
                 delle sue unita' che cadono nel `top_q` di tau_hat fuori campione, divisa
                 per il numero di unita' del braccio.
      "thompson" ricompensa = numero di unita' del braccio nel `top_q` di tau_hat fuori
                 campione; a posteriori Beta(1 + successi, 1 + insuccessi) per braccio,
                 quota obiettivo = P(tasso di cov > tasso di rand).
      "none"     nessuna attribuzione: la quota resta lam0 (mix fisso a cicli).

    focus: quota di ogni ciclo dal secondo in poi estratta direttamente dalla regione
      candidata corrente (a caso tra i punti con tau_hat nel top `rho`), invece che dai due
      bracci. Default 0 = comportamento v0.2. Con focus > 0 la fase 1 non si limita a
      scegliere tra due modi "ciechi" di campionare: va a verificare dove la stima vede
      l'effetto. Il resto del ciclo (1 - focus) resta esplorazione rand/cov.
      Le unita' cosi' estratte hanno arm = 2 e non contano nell'attribuzione.
      NOTA: non e' tra i candidati della sez. 6 del piano congelato (vedi CHANGELOG).
    """
    if rule not in ATTRIBUTION_RULES:
        raise ValueError(f"rule deve essere una tra {ATTRIBUTION_RULES}")
    m = n1 // rounds
    lists = {0: rng.permutation(len(Z)),
             1: rng.permutation(select_tempered(Z, n1, 1.0, rng, rk=rk))}
    ptr = {0: 0, 1: 0}
    used = np.zeros(len(Z), bool)
    S, T, Y, A, shares = [], [], [], [], []
    share = lam0
    pending = np.array([], dtype=int)      # unita' "focus" decise alla fine del ciclo precedente
    for r in range(rounds):
        size = m if r < rounds - 1 else n1 - m * (rounds - 1)
        rest = size - len(pending)
        k_cov = int(round(share * rest))
        batch, arm = [int(j) for j in pending], [2] * len(pending)
        used[pending] = True
        for a, k in ((1, k_cov), (0, rest - k_cov)):
            got = 0
            while got < k and ptr[a] < len(lists[a]):
                j = int(lists[a][ptr[a]]); ptr[a] += 1
                if not used[j]:
                    used[j] = True; batch.append(j); arm.append(a); got += 1
            if got < k:  # braccio esaurito: si completa con l'altro
                while got < k:
                    j = int(lists[0][ptr[0]]); ptr[0] += 1
                    if not used[j]:
                        used[j] = True; batch.append(j); arm.append(0); got += 1
        batch = np.array(batch)
        Tb = anticluster(X[batch], 2, rng)
        Yb = measure(batch, Tb)
        S.append(batch); T.append(Tb); Y.append(Yb); A.append(np.array(arm))
        shares.append(share)
        if r == rounds - 1:
            break
        Sx, Tx, Yx, Ax = (np.concatenate(v) for v in (S, T, Y, A))
        tau_hat = fit_cate(X[Sx], Tx, Yx, seed=seed + r, learner="dr", leaf=10)(X)
        tau_hat[Sx] = -np.inf
        top = np.argsort(-tau_hat)[:int(rho * len(Z))]
        w = np.clip(tau_hat[top], 0, None)
        if rule == "none":
            s_cov = share
        elif rule == "v02":
            near = Ax[pairwise_distances_argmin_min(Z[top], Z[Sx])[0]]
            if focus > 0:                  # i punti piu' vicini a un'unita' focus non votano
                w = w * (near < 2)
            s_cov = float((w * (near == 1)).sum() / w.sum()) if w.sum() > 0 else 0.5
        elif rule == "density":
            near_all = Ax[pairwise_distances_argmin_min(Z, Z[Sx])[0]]
            free = ~used
            b = float((near_all[free] == 1).mean())
            s_top = float((w * (near_all[top] == 1)).sum() / w.sum()) if w.sum() > 0 else b
            eps = 1e-3
            b, s_top = np.clip(b, eps, 1 - eps), np.clip(s_top, eps, 1 - eps)
            odds = (s_top / (1 - s_top)) / (b / (1 - b))
            s_cov = float(share * odds / (share * odds + 1 - share))
        else:
            psi, tau_oof = _oof_parts(X[Sx], Tx, Yx, seed=seed + r, leaf=10)
            in_top = tau_oof >= np.quantile(tau_oof, 1 - top_q)
            n_a = np.array([(Ax == 0).sum(), (Ax == 1).sum()])
            if n_a.min() == 0:
                s_cov = share
            elif rule == "yield":
                y = np.array([max(psi[(Ax == a) & in_top].sum() / n_a[a], 0.0) for a in (0, 1)])
                s_cov = float(y[1] / y.sum()) if y.sum() > 0 else share
            else:  # thompson
                h = np.array([((Ax == a) & in_top).sum() for a in (0, 1)])
                g = np.random.default_rng([seed, r, 7])
                th = [g.beta(1 + h[a], 1 + n_a[a] - h[a], 4000) for a in (0, 1)]
                s_cov = float((th[1] > th[0]).mean())
        share = float(np.clip(0.5 * share + 0.5 * s_cov, floor, 1 - floor))
        if focus > 0:
            nxt = m if r + 1 < rounds - 1 else n1 - m * (rounds - 1)
            pending = rng.choice(top, min(int(round(focus * nxt)), len(top)), replace=False)
    S1, T1, Y1, A1 = (np.concatenate(v) for v in (S, T, Y, A))
    return S1, T1, Y1, A1, shares


# ---------------------------------------------------------------- vie esplorative (fuori piano)
def select_projected(Z: np.ndarray, n: int, rng: np.random.Generator, max_dim: int = 2) -> np.ndarray:
    """Copertura per proiezioni: invece di pesare la rarita' sulla densita' congiunta
    (che porta alle code estreme), si copre in modo uniforme ogni covariata e ogni coppia
    di covariate, lasciando le altre libere di seguire la popolazione.
    Per ogni proiezione si usa `select_tempered` (lambda = 1) sulle sole colonne scelte;
    il budget n e' diviso in parti uguali tra le proiezioni. Favorisce le zone "insolite
    per una combinazione di poche covariate" ma ordinarie nelle altre."""
    from itertools import combinations
    d = Z.shape[1]
    projs = [c for k in range(1, max_dim + 1) for c in combinations(range(d), k)]
    order = rng.permutation(len(projs))
    quota = np.full(len(projs), n // len(projs))
    quota[order[: n - quota.sum()]] += 1
    used = np.zeros(len(Z), bool)
    for p, q in zip(projs, quota):
        if q == 0:
            continue
        free = np.flatnonzero(~used)
        sub = Z[free][:, list(p)]
        used[free[select_tempered(sub, int(q), 1.0, rng)]] = True
    return np.flatnonzero(used)


def zoom_phase1(Z: np.ndarray, X: np.ndarray, measure, n1: int, rng: np.random.Generator,
                K: int = 16, rounds: int = 4, keep: float = 0.5, seed: int = 0):
    """Fase 1 "a zoom": ricerca per gruppi invece che per singole unita'.
    Ciclo 0: la popolazione e' divisa in K celle (k-means); il lotto e' ripartito in parti
    uguali tra le celle. Dopo ogni ciclo si stima l'effetto medio di ogni cella attiva
    (differenza trattati - controlli dei residui di Y su X) con il suo errore standard;
    si tiene la quota `keep` di celle con stima + errore standard piu' alta, ciascuna e'
    divisa in due (k-means) e il ciclo dopo campiona solo li'. Le unita' gia' misurate
    restano nelle sottocelle a cui appartengono, quindi le stime si accumulano.
    Restituisce S1, T1, Y1 e, per ogni ciclo, il numero di celle attive."""
    cell = MiniBatchKMeans(n_clusters=K, n_init=1, batch_size=4096,
                           random_state=int(rng.integers(1 << 31))).fit_predict(Z)
    active = list(range(K))
    nxt = K
    used = np.zeros(len(Z), bool)
    m = n1 // rounds
    S, T, Y, hist = [], [], [], []
    for r in range(rounds):
        size = m if r < rounds - 1 else n1 - m * (rounds - 1)
        quota = np.full(len(active), size // len(active))
        quota[rng.permutation(len(active))[: size - quota.sum()]] += 1
        batch = []
        for c, q in zip(active, quota):
            free = np.flatnonzero((cell == c) & ~used)
            batch.extend(rng.choice(free, min(q, len(free)), replace=False))
        if len(batch) < size:                       # celle esaurite: si completa a caso
            free = np.setdiff1d(np.flatnonzero(~used), batch)
            batch.extend(rng.choice(free, size - len(batch), replace=False))
        batch = np.array(batch, dtype=int)
        used[batch] = True
        Tb = anticluster(X[batch], 2, rng)
        Yb = measure(batch, Tb)
        S.append(batch); T.append(Tb); Y.append(Yb); hist.append(len(active))
        if r == rounds - 1:
            break
        Sx, Tx, Yx = (np.concatenate(v) for v in (S, T, Y))
        D = np.column_stack([np.ones(len(Yx)), X[Sx]])
        e = Yx - D @ np.linalg.lstsq(D, Yx, rcond=None)[0]
        s2 = e.var()
        score = []
        for c in active:
            inc = cell[Sx] == c
            n1c, n0c = (inc & (Tx == 1)).sum(), (inc & (Tx == 0)).sum()
            if min(n1c, n0c) == 0:
                score.append(np.inf)                # cella non ancora valutabile: resta attiva
                continue
            est = e[inc & (Tx == 1)].mean() - e[inc & (Tx == 0)].mean()
            score.append(est + np.sqrt(s2 * (1 / n1c + 1 / n0c)))
        kept = [active[i] for i in np.argsort(-np.array(score))[: max(1, int(round(keep * len(active))))]]
        active = []
        for c in kept:
            idx = np.flatnonzero(cell == c)
            if len(idx) < 40:
                active.append(c)
                continue
            half = MiniBatchKMeans(n_clusters=2, n_init=1, batch_size=4096,
                                   random_state=seed + r).fit_predict(Z[idx])
            cell[idx[half == 1]] = nxt
            active += [c, nxt]
            nxt += 1
    S1, T1, Y1 = (np.concatenate(v) for v in (S, T, Y))
    return S1, T1, Y1, hist


def sequential_phase1(Z: np.ndarray, X: np.ndarray, measure, rng: np.random.Generator,
                      round_size: int = 100, max_n: int = 800, min_rounds: int = 3,
                      threshold: float = 0.5, rho: float = 0.03, cov_share: float = 0.5,
                      rk: np.ndarray | None = None, seed: int = 0, leaf: int = 10,
                      criterion: str = "signal"):
    """Fase 1 a lunghezza variabile (esplorativa, fuori piano).

    Cicli da `round_size` unita' non ancora usate: `cov_share` dal braccio di copertura
    (densita' temperata, lambda = 1) e il resto a caso; trattamento/controllo bilanciati con
    anticlustering dentro ogni ciclo. Dopo ogni ciclo si stima tau(x) su tutte le unita'
    misurate e si calcola R_hat (top `rho` del learning set, unita' misurate escluse).
    Criteri di arresto, valutati dal ciclo `min_rounds` in poi (oppure si arriva a `max_n`):
      "stability": Jaccard(R_hat_r, R_hat_{r-1}) >= threshold. Sconsigliato: la sovrapposizione
                   cresce anche senza effetto, perche' i dati di cicli consecutivi coincidono quasi.
      "signal":    `heterogeneity_t` >= threshold: la stima di tau(x) fatta su meta' delle unita'
                   predice chi risponde di piu' nell'altra meta' (simile al test "best linear
                   predictor"). Senza eterogeneita' la statistica e' circa N(0, 1).
    Il budget restante va alla fase 2.

    Il campionamento non dipende dalla regola di arresto: con threshold = inf si ottiene la
    sequenza completa, e qualsiasi regola corrisponde a un suo prefisso.
    L'arresto usa solo dati di fase 1, quindi l'inferenza di fase 2 su unita' nuove resta valida.

    Restituisce S1, T1, Y1, history: per ogni ciclo n misurate e Jaccard con il ciclo prima.
    """
    if rk is None:
        rk = knn_radius(Z, rng)
    n_keep = int(rho * len(Z))
    used = np.zeros(len(Z), bool)
    S, T, Y, hist = [], [], [], []
    prev = None
    while used.sum() < max_n:
        size = min(round_size, max_n - int(used.sum()))
        free = np.flatnonzero(~used)
        k = int(round(cov_share * size))
        cov = free[select_tempered(Z[free], k, 1.0, rng, rk=rk[free])] if k > 0 else np.array([], int)
        rest = np.setdiff1d(free, cov)
        batch = np.concatenate([cov, rng.choice(rest, size - len(cov), replace=False)])
        used[batch] = True
        Tb = anticluster(X[batch], 2, rng)
        S.append(batch); T.append(Tb); Y.append(measure(batch, Tb))
        Sx, Tx, Yx = (np.concatenate(v) for v in (S, T, Y))
        th = fit_cate(X[Sx], Tx, Yx, seed=seed, learner="dr", leaf=leaf)(X)
        th[used] = -np.inf
        cur = np.argsort(-th)[:n_keep]
        jac = np.nan
        if prev is not None:
            a, b = set(cur.tolist()), set(prev.tolist())
            jac = len(a & b) / len(a | b)
        tstat = heterogeneity_t(X[Sx], Tx, Yx, seed=seed, leaf=leaf)
        hist.append(dict(n=int(used.sum()), jaccard=jac, tstat=tstat))
        prev = cur
        stat = jac if criterion == "stability" else tstat
        if len(hist) >= min_rounds and stat >= threshold:
            break
    S1, T1, Y1 = (np.concatenate(v) for v in (S, T, Y))
    return S1, T1, Y1, hist


def outer_stratum(X: np.ndarray, q: float = 0.02) -> np.ndarray:
    """Strato esterno, dalle sole covariate (esplorativo, fuori piano).

    Standardizzazione robusta (mediana e scarto interquartile / 1,349), poi distanza euclidea
    dal centro. Restituisce la maschera delle unita' nel quantile superiore `q` della distanza:
    valori estremi e gruppi lontani dal corpo principale. Non usa esiti, quindi puo' essere
    fissata prima dell'esperimento come strato del disegno.
    """
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0)) / 1.349
    Zr = (X - med) / np.where(iqr > 0, iqr, 1.0)
    dist = np.sqrt((Zr ** 2).sum(1))
    return dist > np.quantile(dist, 1 - q)


def outer_ratio(X: np.ndarray) -> dict:
    """Diagnostica per decidere se isolare lo strato esterno (esplorativa, fuori piano).

    Rapporto tra il quantile 99,5% e il 90% della distanza robusta dal centro (come in
    `outer_stratum`), confrontato con il valore atteso per una normale nella stessa dimensione.
    Vicino a 1 (rapporto relativo): nessun gruppo isolato ne' code pesanti, conviene non stratificare.
    Molto sopra 1: valori estremi o gruppi lontani dal corpo principale.
    """
    from scipy.stats import chi2
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0)) / 1.349
    d = np.sqrt((((X - med) / np.where(iqr > 0, iqr, 1.0)) ** 2).sum(1))
    obs = float(np.quantile(d, 0.995) / np.quantile(d, 0.90))
    exp = float(np.sqrt(chi2.ppf(0.995, X.shape[1]) / chi2.ppf(0.90, X.shape[1])))
    return {"ratio": obs, "expected_normal": exp, "relative": obs / exp}


# ---------------------------------------------------------------- configurazione congelata v0.3-dev
V03_CONFIG = dict(n1_share=0.60, rho=0.03, q_outer=0.02, ratio_threshold=1.10,
                  outer_share=0.16, strat_n1_share=0.52, alpha=0.05, rounds=4)


def pipeline_v03(X: np.ndarray, measure, B: int, rng: np.random.Generator, seed: int = 0,
                 cfg: dict | None = None) -> dict:
    """Versione congelata il 19 novembre 2026 per la valutazione (tag v0.3-dev-frozen).

    Combina le modifiche validate sui semi di sviluppo (registro, punti 9, 12, 14):
      - fase 1 adattiva v0.2 (`adaptive_phase1`, regola invariata) con il 60% del budget;
      - stratificazione condizionata: se `outer_ratio(X)["relative"] > 1,10`, lo strato esterno
        (`outer_stratum`, 2%) riceve il 16% del budget e un test proprio (HC3) a alpha/2; il resto
        segue la pipeline con fase 1 al 52% del budget e fase 2 sul resto, test a alpha/2;
      - fase 2: R_hat = top 3% di tau_hat (unita' di fase 1 escluse), test unilaterale HC1.
    measure(idx, T) -> Y sugli indici di X. Restituisce rifiuto globale, i test eseguiti
    (gruppo, risultato, soglia) e se la stratificazione si e' attivata.
    """
    c = dict(V03_CONFIG, **(cfg or {}))
    out = dict(stratified=False, tests=[])

    def focused(idx_pool, n1, n2, alpha):
        Xs = X[idx_pool]
        Zs = standardize(Xs)
        rk = knn_radius(Zs, rng)
        lam0 = population_diagnostic(Zs)["lam0"]
        meas = lambda i, T: measure(idx_pool[i], T)
        S1, T1, Y1, _, _ = adaptive_phase1(Zs, Xs, meas, n1, rng, lam0=lam0, rounds=c["rounds"],
                                           rk=rk, seed=seed)
        th = fit_cate(Xs[S1], T1, Y1, seed=seed, learner="dr", leaf=10)(Xs)
        th[S1] = -np.inf
        pool = np.argsort(-th)[:int(c["rho"] * len(Xs))]
        S2 = rng.choice(pool, n2, replace=False)
        T2 = anticluster(Xs[S2], 2, rng)
        Y2 = meas(S2, T2)
        res = ols_effect(Xs[S2], T2, Y2, alpha=alpha)
        return dict(group=idx_pool[pool], S1=idx_pool[S1], S2=idx_pool[S2], res=res, alpha=alpha)

    all_idx = np.arange(len(X))
    if outer_ratio(X)["relative"] > c["ratio_threshold"]:
        out["stratified"] = True
        O = outer_stratum(X, c["q_outer"])
        nO = int(round(c["outer_share"] * B))
        n1 = int(round(c["strat_n1_share"] * B))
        out["tests"].append(focused(all_idx[~O], n1, B - nO - n1, c["alpha"] / 2))
        idxO = all_idx[O]
        S = rng.choice(idxO, min(nO, len(idxO)), replace=False)
        T = anticluster(X[S], 2, rng)
        Y = measure(S, T)
        out["tests"].append(dict(group=idxO, S1=np.array([], int), S2=S, alpha=c["alpha"] / 2,
                                 res=ols_effect(X[S], T, Y, alpha=c["alpha"] / 2, se_type="hc3")))
    else:
        n1 = int(round(c["n1_share"] * B))
        out["tests"].append(focused(all_idx, n1, B - n1, c["alpha"]))
    out["reject"] = any(t["res"]["reject"] for t in out["tests"])
    return out
