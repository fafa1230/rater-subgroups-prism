"""SCO-FCM as published by Yang et al. (2026): the initial centres of Algorithm 2 (Eqs. 5-10), the reachability
matrix of Eq. (11), and fuzzy C-means whose squared distances carry the consensus cost of joining a cluster
(Eqs. 12-19).

The consensus cost of participant y in cluster g_x is cc = prod_{s in g_x} (1 - R_ys). The printed objective (Eq. 12)
divides each squared distance by cc, so the memberships of Eq. (19) rise as reachability to the cluster falls; the
text of the paper describes the opposite (high reachability lowers the objective), which multiplies by cc. Both
readings are implemented (`printed`). The product runs over the other members of the cluster and is kept in log
space, because here it spans hundreds of participants.
"""
from __future__ import annotations

import numpy as np

import config
from src.algorithm1 import sqdist


def reachability(S_hist, S_curr, zeta=config.ZETA):
    """Eq. (11): R = zeta * HBC + (1 - zeta) * CRC, with the background and current relations in their place."""
    return zeta * S_hist + (1 - zeta) * S_curr


def initial_centres(F, R, centrality, c):
    """Algorithm 2: the most central participant first (Eq. 5), then argmin_k q_k / p_k (Eqs. 6-10), where p_k adds the
    normalized centrality to the normalized distance from the chosen centres and q_k is the normalized reachability
    to them."""
    D = np.sqrt(sqdist(F, F))
    chosen = [int(np.argmax(centrality))]
    while len(chosen) < c:
        rest = np.ones(len(F), bool)
        rest[chosen] = False
        p = centrality / centrality[rest].sum() + D[:, chosen].sum(1) / D[np.ix_(rest, chosen)].sum()
        q = R[:, chosen].sum(1) / R[np.ix_(rest, chosen)].sum()
        chosen.append(int(np.argmin(np.where(rest, q / p, np.inf))))
    return chosen


def log_consensus_cost(R, labels, c):
    """log cc(y, g_x) = sum over the other members s of g_x of log(1 - R_ys); an empty cluster costs nothing."""
    L = np.log1p(-np.clip(R, 0.0, 1.0 - 1e-9))
    np.fill_diagonal(L, 0.0)
    return L @ np.eye(c)[labels]


def centre_cost(R, centres):
    """log cc before the first assignment, when every cluster holds only its initial centre (Step 3-1)."""
    L = np.log1p(-np.clip(R[:, centres], 0.0, 1.0 - 1e-9))
    L[centres, np.arange(len(centres))] = 0.0
    return L


def cluster(F, S_hist, S_curr, pr_h, pr_c, c, zeta=config.ZETA, a=config.FUZZIFIER, printed=True, rho=1e-4,
            t_max=config.T_MAX):
    """Memberships, labels and iterations of SCO-FCM (Eqs. 12-19). The clusters g_x that enter the consensus cost hold
    only their centres at first and then the current labels, updated after every iteration as in Algorithm 3, which
    stops when no centre moves by more than rho."""
    R = reachability(S_hist, S_curr, zeta)
    centres = initial_centres(F, R, zeta * pr_h + (1 - zeta) * pr_c, c)
    V = F[centres].copy()
    labels = None
    sign = -1.0 if printed else 1.0                     # printed: d^2 / cc; described: d^2 * cc
    for t in range(1, t_max + 1):
        log_cc = centre_cost(R, centres) if labels is None else log_consensus_cost(R, labels, c)
        z = -(np.log(sqdist(F, V) + 1e-12) + sign * log_cc) / (a - 1.0)   # Eq. (19) in log space
        z -= z.max(1, keepdims=True)
        U = np.exp(z)
        U /= U.sum(1, keepdims=True)
        logw = a * np.log(U + 1e-300) + sign * log_cc     # Eq. (15): centres weighted by u^a / cc (or u^a * cc)
        logw -= logw.max(0, keepdims=True)
        W = np.exp(logw)
        V_new = (W.T @ F) / W.sum(0)[:, None]
        shift = float(np.abs(V_new - V).max())
        V, labels = V_new, U.argmax(1)
        if shift <= rho:
            break
    return U, labels, t


def cluster_algorithm3(F, S_hist, S_curr, pr_h, pr_c, c, zeta=config.ZETA, inverse=False, rho=1e-4,
                       t_max=config.T_MAX):
    """Algorithm 3 as printed: every participant joins the cluster of least total cost TC = co + dis (Steps 3-2 and
    3-3), with dis the Euclidean distance to the centre and co the consensus cost cc, or 1/cc (`inverse`, the
    definition given before Eq. 12); each centre then moves to the mean of its members (Step 3-4) until no centre
    moves by more than rho (Step 3-5)."""
    R = reachability(S_hist, S_curr, zeta)
    centres = initial_centres(F, R, zeta * pr_h + (1 - zeta) * pr_c, c)
    V = F[centres].copy()
    labels = None
    for t in range(1, t_max + 1):
        log_cc = centre_cost(R, centres) if labels is None else log_consensus_cost(R, labels, c)
        co = np.exp(-log_cc) if inverse else np.exp(log_cc)
        labels = (co + np.sqrt(sqdist(F, V))).argmin(1)
        V_new = np.array([F[labels == x].mean(0) if (labels == x).any() else V[x] for x in range(c)])
        shift = float(np.abs(V_new - V).max())
        V = V_new
        if shift <= rho:
            break
    return labels, t
