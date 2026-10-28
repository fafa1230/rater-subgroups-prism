"""Statistical helpers: permutation p-values, multiplicity corrections, effect sizes, association."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

import config
from src.algorithm1 import sqdist


def p_upper(real, null):
    """One-sided permutation p-value that the real value is larger than the null values."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= real)) / (1 + len(null)))


def p_holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


def p_bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[o] * len(p) / (len(p) - np.arange(len(p)))).clip(max=1)
    out = np.empty_like(adj)
    out[o] = adj
    return out


def validity_indices(inst, U, labels, kappa=config.KAPPA):
    """Cluster-validity indices of one fuzzy partition."""
    X, S = inst["F"], inst["S_curr"]
    c = U.shape[1]
    sizes = np.bincount(labels, minlength=c)
    out = {"c": int(c), "sizes": sorted(sizes.tolist(), reverse=True)}
    if sizes.min() < 2:
        return out | {k: np.nan for k in ("mod_sil", "sil_euc", "sil_reach", "fpc", "fpc_norm", "xb", "db", "ch")}
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Um = U ** 2
    V = (Um.T @ X) / Um.sum(0)[:, None]
    dV = sqdist(V, V)
    np.fill_diagonal(dV, np.inf)
    out["sil_euc"] = float(silhouette_score(X, labels))
    out["sil_reach"] = float(silhouette_score(D, labels, metric="precomputed"))
    out["mod_sil"] = kappa * out["sil_euc"] + (1 - kappa) * out["sil_reach"]
    out["fpc"] = float(Um.sum() / len(X))
    out["fpc_norm"] = float((out["fpc"] - 1 / c) / (1 - 1 / c))
    out["xb"] = float((Um * sqdist(X, V)).sum() / (len(X) * dV.min()))
    out["db"] = float(davies_bouldin_score(X, labels))
    out["ch"] = float(calinski_harabasz_score(X, labels))
    return out


def cramers_v(a, b):
    """Cramer's V between two categorical arrays (bias-uncorrected)."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    T = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(T, (ai, bi), 1)
    if min(T.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(T, correction=False)[0]
    return float(np.sqrt(chi2 / (T.sum() * (min(T.shape) - 1))))


def permanova_r2(X, groups):
    """Share of the total Euclidean sum of squares of the rows of X explained by the groups."""
    Xc = X - X.mean(0)
    _, gi = np.unique(groups, return_inverse=True)
    H = np.eye(gi.max() + 1)[gi]
    n_g = H.sum(0)
    means = (H.T @ Xc) / n_g[:, None]
    return float((n_g[:, None] * means ** 2).sum() / (Xc ** 2).sum())
