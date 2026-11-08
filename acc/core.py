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
def anticluster(X: np.ndarray, K: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Partiziona le righe di X in K gruppi di uguale dimensione (+-1), simili tra loro.

    Euristica assignment-based (idea di Baumann et al. 2026, ABA):
    gli oggetti, ordinati per distanza decrescente dal centroide globale, sono
    processati a blocchi di K; ogni blocco e' assegnato ai K gruppi (uno per
    gruppo) risolvendo un problema di assegnamento che massimizza la distanza
    di ciascun oggetto dal centroide corrente del gruppo che lo riceve.
    Complessita' ~ O(N log N + N K^2), adatta a N nell'ordine dei milioni con K piccolo.
    """
    rng = rng or np.random.default_rng()
    Z = standardize(X)
    n, d = Z.shape
    dist = ((Z - Z.mean(0)) ** 2).sum(1)
    # piccolo rumore per rompere i pareggi
    order = np.argsort(-(dist + 1e-9 * rng.random(n)))
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


def ols_effect(X: np.ndarray, T: np.ndarray, Y: np.ndarray, alpha: float = 0.05) -> dict:
    """Y ~ 1 + T + X, SE robusti HC1. Test unilaterale H0: tau <= 0."""
    D = np.column_stack([np.ones(len(Y)), T, X])
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    e = Y - D @ beta
    n, p = D.shape
    bread = np.linalg.pinv(D.T @ D)
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


def adaptive_phase1(Z: np.ndarray, X: np.ndarray, measure, n1: int, rng: np.random.Generator,
                    lam0: float = 0.5, rounds: int = 4, rho: float = 0.03,
                    rk: np.ndarray | None = None, floor: float = 0.15, seed: int = 0):
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
    Restituisce S1, T1, Y1, arm (0 = rand, 1 = cov) e la storia delle quote.
    """
    m = n1 // rounds
    lists = {0: rng.permutation(len(Z)),
             1: rng.permutation(select_tempered(Z, n1, 1.0, rng, rk=rk))}
    ptr = {0: 0, 1: 0}
    used = np.zeros(len(Z), bool)
    S, T, Y, A, shares = [], [], [], [], []
    share = lam0
    for r in range(rounds):
        size = m if r < rounds - 1 else n1 - m * (rounds - 1)
        k_cov = int(round(share * size))
        batch, arm = [], []
        for a, k in ((1, k_cov), (0, size - k_cov)):
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
        near = Ax[pairwise_distances_argmin_min(Z[top], Z[Sx])[0]]
        s_cov = float((w * (near == 1)).sum() / w.sum()) if w.sum() > 0 else 0.5
        share = float(np.clip(0.5 * share + 0.5 * s_cov, floor, 1 - floor))
    S1, T1, Y1, A1 = (np.concatenate(v) for v in (S, T, Y, A))
    return S1, T1, Y1, A1, shares
