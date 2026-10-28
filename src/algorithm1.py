"""Algorithm 1 of the manuscript: adapted SCO-FCM for participant clustering and consensus reaching.

  1  priority <- zeta * PageRank(S_hist) + (1 - zeta) * PageRank(S_curr)
  3  initial centres: highest priority first, then the participant least reachable from those chosen
  4  Fuzzy C-Means -> U_e;  U <- kappa * U_e + (1 - kappa) * U_r  (U_r from S_curr);  labels = argmax U
  5  S(c) <- kappa * silhouette(X, Euclidean) + (1 - kappa) * silhouette(1 - S_curr)
  7  c* <- argmax_c S(c)
  9  w_k <- mu * cohesion_k(S_curr) + (1 - mu) * size_k
 11  consensus score of a model <- sum_k w_k * mean score of subgroup k
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score

import config


def sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def priority(inst, zeta=config.ZETA):
    return zeta * inst["pr_h"] + (1 - zeta) * inst["pr_c"]


def centre_order(prio, S, c_max):
    order = [int(np.argmax(prio))]
    while len(order) < c_max:
        reach = S[:, order].max(axis=1)
        reach[order] = np.inf
        cand = np.flatnonzero(reach == reach.min())
        order.append(int(cand[np.argmax(prio[cand])]))
    return order


def fcm(X, V0, m=config.FUZZIFIER, tol=config.FCM_TOL, t_max=config.T_MAX):
    V = V0.copy()
    U_prev = None
    for _ in range(t_max):
        inv = (sqdist(X, V) + 1e-12) ** (-1.0 / (m - 1.0))
        U = inv / inv.sum(1, keepdims=True)
        Um = U ** m
        V = (Um.T @ X) / Um.sum(0)[:, None]
        if U_prev is not None and np.abs(U - U_prev).max() < tol:
            break
        U_prev = U
    return U, V


def reach_membership(S, labels, c):
    """U_r(i, k): mean similarity of participant i to the members of subgroup k, row-normalized."""
    onehot = np.eye(c)[labels]
    M = (S @ onehot) / np.maximum(onehot.sum(0), 1)
    return M / np.maximum(M.sum(1, keepdims=True), 1e-12)


def cluster(X, S, prio, c, kappa=config.KAPPA, order=None):
    order = order if order is not None else centre_order(prio, S, c)
    U_e, _ = fcm(X, X[order[:c]])
    U = kappa * U_e + (1 - kappa) * reach_membership(S, U_e.argmax(1), c)
    return U, U.argmax(1)


def modified_silhouette(X, S, labels, kappa=config.KAPPA):
    if len(np.unique(labels)) < 2:
        return np.nan
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return kappa * silhouette_score(X, labels) + (1 - kappa) * silhouette_score(D, labels, metric="precomputed")


def all_c(inst, zeta=config.ZETA, kappa=config.KAPPA, cands=config.C_CANDIDATES):
    """Partitions for every candidate number of subgroups (Algorithm 1, lines 2-6)."""
    prio = priority(inst, zeta)
    order = centre_order(prio, inst["S_curr"], max(cands))
    return {c: cluster(inst["F"], inst["S_curr"], prio, c, kappa, order) for c in cands}


def select_c(inst, parts, kappa=config.KAPPA):
    """Line 7: c* maximizes the modified silhouette over partitions without tiny (< 2) subgroups."""
    sil, best = {}, None
    for c, (U, lab) in parts.items():
        s = modified_silhouette(inst["F"], inst["S_curr"], lab, kappa) if np.bincount(lab, minlength=c).min() >= 2 else np.nan
        sil[c] = s
        if np.isfinite(s) and (best is None or s > best[1]):
            best = (c, s)
    return best[0], best[1], sil


def run(inst, zeta=config.ZETA, kappa=config.KAPPA, mu=config.MU):
    parts = all_c(inst, zeta, kappa)
    c, f1, sil = select_c(inst, parts, kappa)
    U, lab = parts[c]
    w, coh = subgroup_weights(inst["S_curr"], lab, c, mu)
    return {"c": c, "f1": f1, "sil_by_c": sil, "U": U, "labels": lab, "w": w, "cohesion": coh, "parts": parts}


def subgroup_weights(S, labels, c, mu=config.MU):
    coh = np.array([S[np.ix_(labels == k, labels == k)].mean() for k in range(c)])
    size = np.bincount(labels, minlength=c) / len(labels)
    w = mu * coh / coh.sum() + (1 - mu) * size
    return w / w.sum(), coh


def group_scores(F_raw, labels, c):
    """Mean observed score of every subgroup on every model (NaN where no member rated it)."""
    obs = ~np.isnan(F_raw)
    onehot = np.eye(c)[labels]
    num = onehot.T @ np.where(obs, F_raw, 0.0)
    cnt = onehot.T @ obs.astype(float)
    return np.divide(num, cnt, out=np.full_like(num, np.nan), where=cnt > 0)


def consensus(F_raw, labels, w):
    """Weighted consensus (line 11) and the flat ranking over all participants."""
    c = len(w)
    G = group_scores(F_raw, labels, c)
    has = ~np.isnan(G)
    W = w[:, None] * has
    cons = np.nansum(W * np.nan_to_num(G), axis=0) / W.sum(axis=0)
    flat = np.nanmean(F_raw, axis=0)
    return cons, flat, G
