"""Analyses of the revised manuscript.

    python -I 03_analyze.py main        # Algorithm 1 at the manuscript's parameters, consensus vs flat ranking
    python -I 03_analyze.py validity    # validity indices for c = 2..8 against 1,000 permutation nulls
    python -I 03_analyze.py stability   # 80% participant subsamples
    python -I 03_analyze.py split       # split-half replication and held-out tests (real and null)
    python -I 03_analyze.py reliability # split-half consistency of individual preferences (no clustering)
    python -I 03_analyze.py baselines   # other clustering methods and the published SCO-FCM, each against its own null
    python -I 03_analyze.py exposure    # whether the models a participant met depend on the participant
    python -I 03_analyze.py external    # sociodemographic validation and representation of groups
    python -I 03_analyze.py relative    # robustness: graded instead of pairwise P-HFE encoding
    python -I 03_analyze.py membership  # largest fuzzy membership, real and null
    python -I 03_analyze.py sensitivity # zeta x kappa grid (11 x 11) against 1,000 permutation nulls

Results go to results/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, kruskal  # noqa: E402
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans, SpectralClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

import config  # noqa: E402
from src import algorithm1 as A1  # noqa: E402
from src import sco_fcm  # noqa: E402
from src.data import build_instance, pagerank, reassign_conversations, split_conversations  # noqa: E402
from src.stats import (cramers_v, p_bh, p_holm, p_upper, permanova_r2,  # noqa: E402
                       validity_indices)

OUT = config.RESULTS
OUT.mkdir(exist_ok=True)
_z = np.load(config.PROCESSED / "instance.npz", allow_pickle=False)
USERS, MODELS = list(_z["users"]), list(_z["models"])
REAL = {k: _z[k] for k in ("F", "F_raw", "observed", "S_hist", "S_curr", "pr_h", "pr_c")}
BACKGROUND = (REAL["S_hist"], REAL["pr_h"])
OP = pd.read_csv(config.PROCESSED / "opening_turns.csv")
EXT = pd.read_csv(config.PROCESSED / "external.csv")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build(op, encoding=config.PHFE_ENCODING):
    return build_instance(op, USERS, MODELS, BACKGROUND, encoding)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))


def main_result():
    if (OUT / "main_labels.npy").exists():
        return int(json.loads((OUT / "main.json").read_text())["c_star"]), np.load(OUT / "main_labels.npy")
    raise RuntimeError("run the 'main' stage first")


# ---------------------------------------------------------------- main
def stage_main():
    r = A1.run(REAL)
    lab, c = r["labels"], r["c"]
    np.save(OUT / "main_labels.npy", lab)
    np.save(OUT / "main_U.npy", r["U"])
    cons, flat, G = A1.consensus(REAL["F_raw"], lab, r["w"])
    rank = lambda v: pd.Series(v).rank(ascending=False, method="min").to_numpy()
    by_mu = {}
    for mu in (0.0, 0.3, 0.5, 0.7, 1.0):
        w, _ = A1.subgroup_weights(REAL["S_curr"], lab, c, mu)
        cm, _, _ = A1.consensus(REAL["F_raw"], lab, w)
        by_mu[mu] = {"weights": w.round(4).tolist(), "kendall_tau_vs_flat": float(kendalltau(cm, flat).statistic),
                     "mean_abs_rank_shift": float(np.abs(rank(cm) - rank(flat)).mean())}
    top = pd.DataFrame({"model": MODELS, "flat": flat, "consensus": cons,
                        **{f"subgroup_{k + 1}": G[k] for k in range(c)}}).sort_values("flat", ascending=False)
    top.to_csv(OUT / "main_rankings.csv", index=False)
    res = {"c_star": c, "f1_modified_silhouette": r["f1"], "silhouette_by_c": r["sil_by_c"],
           "sizes": np.bincount(lab).tolist(), "weights": r["w"].tolist(), "cohesion": r["cohesion"].tolist(),
           "kendall_tau_consensus_vs_flat": float(kendalltau(cons, flat).statistic),
           "mean_abs_rank_shift": float(np.abs(rank(cons) - rank(flat)).mean()),
           "kendall_tau_subgroup_vs_flat": [float(kendalltau(G[k][~np.isnan(G[k])], flat[~np.isnan(G[k])]).statistic) for k in range(c)],
           "effect_of_mu": by_mu}
    dump("main.json", res)
    log(json.dumps({k: v for k, v in res.items() if k != "effect_of_mu"}, indent=1, default=str))
    log("\n" + top.head(10).round(3).to_string(index=False))


# ---------------------------------------------------------------- held-out style tests
def subgroup_tests(F_raw, lab, c):
    """Per-model Kruskal-Wallis test of the subgroup difference on observed scores."""
    rows = []
    for j, m in enumerate(MODELS):
        obs = ~np.isnan(F_raw[:, j])
        groups = [F_raw[obs & (lab == k), j] for k in range(c)]
        if min(len(g) for g in groups) < 5:
            continue
        H, p = kruskal(*groups)
        means = [g.mean() for g in groups]
        rows.append({"model": m, "H": H, "p": p, "eps2": H / (obs.sum() - 1), "gap": max(means) - min(means)})
    t = pd.DataFrame(rows)
    t["p_holm"], t["p_bh"] = p_holm(t["p"]), p_bh(t["p"])
    return t


def summary_tests(t):
    return {"sig_bh": int((t.p_bh < config.ALPHA).sum()), "sig_holm": int((t.p_holm < config.ALPHA).sum()),
            "max_gap": float(t.gap.max()), "max_eps2": float(t.eps2.max())}


def deviation_profile(F_raw, lab, c):
    G = A1.group_scores(F_raw, lab, c)
    return G - np.nanmean(F_raw, axis=0)[None, :]


def profile_agreement(devA, devB, sizes):
    rs = []
    for k in range(len(devA)):
        ok = np.isfinite(devA[k]) & np.isfinite(devB[k])
        rs.append(np.corrcoef(devA[k][ok], devB[k][ok])[0, 1] if ok.sum() > 3 else np.nan)
    rs = np.array(rs)
    return float(np.nansum(rs * sizes) / sizes[np.isfinite(rs)].sum())


# ---------------------------------------------------------------- validity vs null
def stage_validity():
    c_star, lab = main_result()
    parts = A1.all_c(REAL)
    real = {c: validity_indices(REAL, *parts[c]) for c in parts}
    naive = summary_tests(subgroup_tests(REAL["F_raw"], lab, c_star))
    rng = np.random.default_rng(config.SEED)
    rows = []
    for b in range(config.N_NULL):
        d = build(reassign_conversations(OP, rng))
        p = A1.all_c(d)
        cs, s_max, _ = A1.select_c(d, p)
        ix = validity_indices(d, *p[c_star])
        nv = summary_tests(subgroup_tests(d["F_raw"], p[c_star][1], c_star))
        rows.append({"b": b, "cstar": cs, "mod_sil_max": s_max, **{f"{k}": ix[k] for k in ix if k not in ("c", "sizes")},
                     **{f"naive_{k}": v for k, v in nv.items()}})
        if b % 100 == 0:
            log("null", b)
    null = pd.DataFrame(rows)
    null.to_csv(OUT / "validity_null.csv", index=False)
    r = real[c_star]
    tests = {}
    for k, better in [("mod_sil", 1), ("sil_euc", 1), ("sil_reach", 1), ("fpc_norm", 1), ("ch", 1), ("xb", -1), ("db", -1)]:
        nv = null[k].to_numpy()
        tests[k] = {"real": r[k], "null_mean": float(np.nanmean(nv)), "null_q025": float(np.nanquantile(nv, .025)),
                    "null_q975": float(np.nanquantile(nv, .975)), "p": p_upper(better * r[k], better * nv)}
    best = max(v["mod_sil"] for v in real.values() if np.isfinite(v["mod_sil"]))
    tests["mod_sil_max_over_c"] = {"real": best, "null_mean": float(null.mod_sil_max.mean()),
                                   "null_q975": float(null.mod_sil_max.quantile(.975)), "p": p_upper(best, null.mod_sil_max)}
    naive_tests = {k: {"real": naive[k], "null_mean": float(null[f"naive_{k}"].mean()),
                       "null_q975": float(null[f"naive_{k}"].quantile(.975)), "p": p_upper(naive[k], null[f"naive_{k}"])}
                   for k in naive}
    res = {"c_star": c_star, "real_by_c": real, "tests_at_cstar": tests, "null_cstar_counts": null.cstar.value_counts().to_dict(),
           "naive_in_sample_tests": naive_tests, "n_null": config.N_NULL}
    dump("validity.json", res)
    log(json.dumps({"tests": tests, "naive": naive_tests, "null_cstar": res["null_cstar_counts"]}, indent=1, default=float))


# ---------------------------------------------------------------- stability
def subset(d, idx):
    return {"F": d["F"][idx], "F_raw": d["F_raw"][idx], "S_curr": d["S_curr"][np.ix_(idx, idx)],
            "S_hist": d["S_hist"][np.ix_(idx, idx)], "pr_h": pagerank(d["S_hist"][np.ix_(idx, idx)]),
            "pr_c": pagerank(d["S_curr"][np.ix_(idx, idx)])}


def cluster_jaccard(a, b):
    return [max(np.sum((a == k) & (b == j)) / np.sum((a == k) | (b == j)) for j in np.unique(b)) for k in np.unique(a)]


def stage_stability():
    c_star, lab0 = main_result()
    km0 = KMeans(c_star, n_init=10, random_state=0).fit_predict(REAL["F"])
    rng = np.random.default_rng(config.SEED + 1)
    n = len(USERS)
    rows = []
    for b in range(config.N_STABILITY):
        idx = np.sort(rng.choice(n, int(0.8 * n), replace=False))
        d = subset(REAL, idx)
        p = A1.all_c(d)
        lab = p[c_star][1]
        jac = cluster_jaccard(lab0[idx], lab)
        km = KMeans(c_star, n_init=10, random_state=b).fit_predict(d["F"])
        rows.append({"b": b, "cstar": A1.select_c(d, p)[0], "ari": adjusted_rand_score(lab0[idx], lab),
                     "jac_min": min(jac), "jac_mean": float(np.mean(jac)), "ari_kmeans": adjusted_rand_score(km0[idx], km)})
    st = pd.DataFrame(rows)
    st.to_csv(OUT / "stability.csv", index=False)
    log("\n" + st.drop(columns="b").describe().T[["mean", "50%", "min", "max"]].round(3).to_string()
        + "\nc* in subsamples: " + str(st.cstar.value_counts().to_dict()))


# ---------------------------------------------------------------- split-half
def stage_split():
    c_star, _ = main_result()
    rows = []
    for kind in ("real", "permuted"):
        rng = np.random.default_rng(config.SEED + (2 if kind == "real" else 3))
        for s in range(config.N_SPLIT):
            op = OP if kind == "real" else reassign_conversations(OP, rng)
            hA, hB = split_conversations(op, rng)
            dA, dB = build(hA), build(hB)
            pA, pB = A1.all_c(dA), A1.all_c(dB)
            labA, labB = pA[c_star][1], pB[c_star][1]
            sizes = np.bincount(labA, minlength=c_star)
            held, ins = subgroup_tests(dB["F_raw"], labA, c_star), subgroup_tests(dA["F_raw"], labA, c_star)
            rows.append({"kind": kind, "split": s, "ari_AB": adjusted_rand_score(labA, labB),
                         "cstar_A": A1.select_c(dA, pA)[0],
                         "profile_r": profile_agreement(deviation_profile(dA["F_raw"], labA, c_star),
                                                        deviation_profile(dB["F_raw"], labA, c_star), sizes),
                         **{f"held_{k}": v for k, v in summary_tests(held).items()},
                         **{f"in_{k}": v for k, v in summary_tests(ins).items()}})
        log("split", kind)
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT / "split.csv", index=False)
    summ = sp.drop(columns="split").groupby("kind").agg(["mean", "median", "min", "max"]).round(4)
    summ.to_csv(OUT / "split_summary.csv")
    log("\n" + summ.T.to_string())


# ---------------------------------------------------------------- reliability of individual preferences
def reliability(dA, dB):
    """Split-half consistency of individual preferences, independent of any clustering:
    r_model  - per model, correlation across participants of their half-A and half-B scores
               (Fisher-averaged over models): stable individual differences in liking a model;
    r_person - per participant, correlation across the models rated in both halves (median):
               stable ordering of models, which also contains the shared consensus."""
    a, b = dA["F_raw"], dB["F_raw"]
    rm = []
    for j in range(a.shape[1]):
        ok = ~np.isnan(a[:, j]) & ~np.isnan(b[:, j])
        if ok.sum() > 30 and a[ok, j].std() > 0 and b[ok, j].std() > 0:
            rm.append(np.corrcoef(a[ok, j], b[ok, j])[0, 1])
    rp = []
    for i in range(a.shape[0]):
        ok = ~np.isnan(a[i]) & ~np.isnan(b[i])
        if ok.sum() >= 4 and a[i, ok].std() > 0 and b[i, ok].std() > 0:
            rp.append(np.corrcoef(a[i, ok], b[i, ok])[0, 1])
    return float(np.tanh(np.mean(np.arctanh(np.clip(rm, -.999, .999))))), float(np.median(rp))


def stage_reliability():
    rows = []
    for kind in ("real", "permuted"):
        rng = np.random.default_rng(config.SEED + (7 if kind == "real" else 8))
        for s in range(config.N_SPLIT):
            op = OP if kind == "real" else reassign_conversations(OP, rng)
            hA, hB = split_conversations(op, rng)
            r_model, r_person = reliability(build(hA), build(hB))
            rows.append({"kind": kind, "split": s, "r_model": r_model, "r_person": r_person})
    rel = pd.DataFrame(rows)
    rel.to_csv(OUT / "reliability.csv", index=False)
    summ = pd.DataFrame({f"{v}_{q}": rel.groupby("kind")[v].quantile(p) if p is not None else rel.groupby("kind")[v].mean()
                         for v in ("r_model", "r_person") for q, p in (("mean", None), ("q025", .025), ("q975", .975))})
    summ.to_csv(OUT / "reliability_summary.csv")
    log("\n" + summ.round(4).to_string())


# ---------------------------------------------------------------- baselines
def behavioural_graph(d):
    W = d["S_curr"].copy()
    np.fill_diagonal(W, 0)
    return W


def louvain_labels(d):
    """Louvain community detection on the weighted behavioural graph; the number of communities is data-driven."""
    lab = np.empty(len(d["S_curr"]), int)
    for c, nodes in enumerate(nx.community.louvain_communities(nx.from_numpy_array(behavioural_graph(d)), seed=0)):
        lab[list(nodes)] = c
    return lab


def modularity(W, lab):
    k, m2 = W.sum(1), W.sum()
    return float(sum(W[np.ix_(lab == c, lab == c)].sum() / m2 - (k[lab == c].sum() / m2) ** 2 for c in np.unique(lab)))


def baseline_labels(d, k, fast=False):
    X = d["F"]
    prio = A1.priority(d)
    order = A1.centre_order(prio, d["S_curr"], k)
    out = {"Algorithm 1 (SCO-FCM)": A1.cluster(X, d["S_curr"], prio, k, config.KAPPA, order)[1],
           "Ablation: no reachability (kappa = 1)": A1.cluster(X, d["S_curr"], prio, k, 1.0, order)[1],
           "Standard FCM (random init)": A1.fcm(X, X[np.random.default_rng(0).choice(len(X), k, replace=False)])[0].argmax(1),
           "K-Means": KMeans(k, n_init=10, random_state=0).fit_predict(X),
           "GMM (diagonal)": GaussianMixture(k, covariance_type="diag", random_state=0, n_init=3).fit_predict(X),
           "Agglomerative (Ward)": AgglomerativeClustering(k, linkage="ward").fit_predict(X),
           "Spectral (behavioural graph)": SpectralClustering(k, affinity="precomputed", random_state=0).fit_predict(d["S_curr"]),
           "Spectral (kNN on profiles)": SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, random_state=0).fit_predict(X)}
    if not fast:
        out["HDBSCAN (data-driven c)"] = HDBSCAN(min_cluster_size=20).fit_predict(X)
        out["Louvain (behavioural graph, data-driven c)"] = louvain_labels(d)
        rel = (X, d["S_hist"], d["S_curr"], d["pr_h"], d["pr_c"], k)
        out["Published SCO-FCM, objective as printed"] = sco_fcm.cluster(*rel, printed=True)[1]
        out["Published SCO-FCM, objective as described"] = sco_fcm.cluster(*rel, printed=False)[1]
        out["Published SCO-FCM, Algorithm 3 (cost cc)"] = sco_fcm.cluster_algorithm3(*rel)[0]
        out["Published SCO-FCM, Algorithm 3 (cost 1/cc)"] = sco_fcm.cluster_algorithm3(*rel, inverse=True)[0]
    return out


def partition_indices(d, lab):
    core = lab >= 0
    labs = lab[core]
    k = len(np.unique(labs))
    r = {"n_clusters": int(k), "noise": int((~core).sum()),
         "sizes": sorted(np.bincount(labs - labs.min()).tolist(), reverse=True) if k else []}
    if k < 2:
        return r | {"sil_euc": np.nan, "sil_reach": np.nan, "ch": np.nan, "modularity": np.nan}
    D = 1.0 - d["S_curr"]
    np.fill_diagonal(D, 0)
    r["sil_euc"] = float(silhouette_score(d["F"][core], labs))
    r["sil_reach"] = float(silhouette_score(D[np.ix_(core, core)], labs, metric="precomputed"))
    r["ch"] = float(calinski_harabasz_score(d["F"][core], labs))
    r["modularity"] = modularity(behavioural_graph(d)[np.ix_(core, core)], labs)
    return r


def stage_baselines():
    c_star, _ = main_result()
    real = baseline_labels(REAL, c_star)
    ref = real["Algorithm 1 (SCO-FCM)"]
    rows = {name: partition_indices(REAL, lab) | {"ari_vs_algorithm1": adjusted_rand_score(ref, lab)} for name, lab in real.items()}
    rng = np.random.default_rng(config.SEED + 4)
    null = {name: [] for name in real}
    for b in range(config.N_NULL_BASELINES):
        d = build(reassign_conversations(OP, rng))
        for name, lab in baseline_labels(d, c_star).items():
            ix = partition_indices(d, lab)
            null[name].append((ix["sil_euc"], ix["ch"], ix["modularity"]))
        if b % 25 == 0:
            log("baseline null", b)
    for name in rows:
        a = np.array(null[name], float)
        for j, key in enumerate(("sil_euc", "ch", "modularity")):
            v = a[:, j][np.isfinite(a[:, j])]
            rows[name][f"{key}_null_mean"] = float(v.mean()) if len(v) else np.nan
            rows[name][f"{key}_p"] = p_upper(rows[name][key], v) if len(v) and np.isfinite(rows[name][key]) else np.nan
    tab = pd.DataFrame(rows).T
    tab.to_csv(OUT / "baselines.csv")
    log("\n" + tab[["n_clusters", "noise", "sizes", "sil_euc", "sil_euc_null_mean", "sil_euc_p", "sil_reach", "ch",
                    "ch_null_mean", "ch_p", "modularity", "modularity_null_mean", "modularity_p",
                    "ari_vs_algorithm1"]].to_string())


# ---------------------------------------------------------------- exposure to models
def exposure_chi2(op):
    """Dispersion of the participant x model exposure counts (opening turns in which a model appeared)."""
    E = pd.crosstab(op["user_id"], op["model_name"]).to_numpy(float)
    expected = E.sum(1, keepdims=True) * E.sum(0, keepdims=True) / E.sum()
    return float(((E - expected) ** 2 / expected).sum())


def stage_exposure():
    """Whether the models a participant met depend on the participant: observed dispersion against
    conversations dealt out at random, as in the permutation null."""
    rng = np.random.default_rng(config.SEED + 10)
    real = exposure_chi2(OP)
    null = np.array([exposure_chi2(reassign_conversations(OP, rng)) for _ in range(config.N_NULL)])
    res = {"chi2": real, "null_mean": float(null.mean()), "null_q025": float(np.quantile(null, .025)),
           "null_q975": float(np.quantile(null, .975)), "p_upper": p_upper(real, null), "n_null": config.N_NULL}
    dump("exposure.json", res)
    log(res)


# ---------------------------------------------------------------- external validation
def stage_external():
    c_star, lab = main_result()
    variables = [c for c in EXT.columns if c not in ("user_id", "us_rep", "uk_rep")]
    rng = np.random.default_rng(config.SEED + 5)
    rows = []
    for v in variables:
        ok = EXT[v].notna().to_numpy()
        x = EXT.loc[ok, v].to_numpy()
        cv = cramers_v(lab[ok], x)
        cv_null = [cramers_v(lab[ok], rng.permutation(x)) for _ in range(config.N_PERM_EXTERNAL)]
        r2 = permanova_r2(REAL["F"][ok], x)
        r2_null = [permanova_r2(REAL["F"][ok], rng.permutation(x)) for _ in range(config.N_PERM_EXTERNAL)]
        rows.append({"variable": v, "n": int(ok.sum()), "groups": int(len(np.unique(x))),
                     "cramers_v_cluster": cv, "cramers_v_null_mean": float(np.mean(cv_null)), "p_cluster": p_upper(cv, cv_null),
                     "R2_profiles": r2, "R2_null_mean": float(np.mean(r2_null)), "p_profiles": p_upper(r2, r2_null)})
    tab = pd.DataFrame(rows)
    tab["p_cluster_holm"], tab["p_profiles_holm"] = p_holm(tab.p_cluster), p_holm(tab.p_profiles)
    tab["R2_excess"] = tab.R2_profiles - tab.R2_null_mean
    tab.to_csv(OUT / "external_association.csv", index=False)
    r2_cluster = permanova_r2(REAL["F"], lab)

    # representation of real-world groups by the flat and the SCO-FCM consensus rankings
    w, _ = A1.subgroup_weights(REAL["S_curr"], lab, c_star)
    cons, flat, _ = A1.consensus(REAL["F_raw"], lab, w)
    rep = []
    for v in ("region", "age", "gender", "religion", "ethnicity"):
        ok = EXT[v].notna().to_numpy()
        x = EXT[v].to_numpy()
        for g in sorted(pd.unique(EXT.loc[ok, v])):
            members = x == g
            sg = np.nanmean(np.where(members[:, None], REAL["F_raw"], np.nan), axis=0)
            t_flat = kendalltau(sg, flat, nan_policy="omit").statistic
            t_cons = kendalltau(sg, cons, nan_policy="omit").statistic
            null_t = []
            for _ in range(config.N_PERM_EXTERNAL):
                pm = rng.permutation(members[ok])
                full = np.zeros(len(x), bool)
                full[np.flatnonzero(ok)[pm]] = True
                null_t.append(kendalltau(np.nanmean(np.where(full[:, None], REAL["F_raw"], np.nan), axis=0), flat,
                                         nan_policy="omit").statistic)
            null_t = np.array(null_t)
            rep.append({"variable": v, "group": g, "n": int(members.sum()), "tau_flat": t_flat, "tau_consensus": t_cons,
                        "tau_null_mean": float(null_t.mean()), "tau_null_q025": float(np.quantile(null_t, .025)),
                        "tau_null_q975": float(np.quantile(null_t, .975)),
                        "p_underrepresented": float((1 + np.sum(null_t <= t_flat)) / (1 + len(null_t)))})
    rep = pd.DataFrame(rep)
    rep["p_holm"] = p_holm(rep.p_underrepresented)
    rep.to_csv(OUT / "external_representation.csv", index=False)
    dump("external.json", {"R2_by_algorithm1_labels": r2_cluster})
    log("\n" + tab.round(4).to_string(index=False) + f"\nR2 of profiles by Algorithm-1 labels: {r2_cluster:.4f}\n\n"
        + rep.round(3).to_string(index=False))


# ---------------------------------------------------------------- fuzzy memberships
def stage_membership():
    """Largest membership of every participant at c*, real and under 50 permutation nulls."""
    c_star, _ = main_result()
    U = np.load(OUT / "main_U.npy")
    rng = np.random.default_rng(config.SEED + 9)
    null = np.array([A1.all_c(build(reassign_conversations(OP, rng)))[c_star][0].max(1) for _ in range(50)])
    np.savez_compressed(OUT / "membership.npz", real=U.max(1), null=null)
    log(f"real max membership: median {np.median(U.max(1)):.4f}, 99th pct {np.quantile(U.max(1), .99):.4f}; "
        f"null median {np.median(null):.4f}, 99th pct {np.quantile(null, .99):.4f}")


# ---------------------------------------------------------------- sensitivity to zeta and kappa
def silhouette_fast(D, labels, c):
    """Mean silhouette from a precomputed distance matrix (same definition as scikit-learn)."""
    n = len(labels)
    onehot = np.eye(c)[labels]
    counts = onehot.sum(0)
    sums = D @ onehot
    n_own = counts[labels] - 1
    a = sums[np.arange(n), labels] / np.maximum(n_own, 1)
    other = sums / counts[None, :]
    other[np.arange(n), labels] = np.inf
    b = other.min(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.nan_to_num((b - a) / np.maximum(a, b))
    s[n_own == 0] = 0.0
    return float(s.mean())


def grid(d, zetas, kappas, keep_labels=False):
    """Algorithm 1 over a zeta x kappa grid. U_e depends on zeta (through the centres) but not on
    kappa, so FCM runs once per (zeta, c); silhouettes are cached per distinct partition."""
    X, S = d["F"], d["S_curr"]
    D_euc = np.sqrt(A1.sqdist(X, X))
    np.fill_diagonal(D_euc, 0.0)
    D_reach = 1.0 - S
    np.fill_diagonal(D_reach, 0.0)
    nz, nk = len(zetas), len(kappas)
    out = {"cstar": np.zeros((nz, nk), int), "f1": np.full((nz, nk), np.nan), "sil_euc": np.full((nz, nk), np.nan)}
    labels_at = {}
    for i, z in enumerate(zetas):
        order = A1.centre_order(A1.priority(d, z), S, max(config.C_CANDIDATES))
        per_c = {}
        for c in config.C_CANDIDATES:
            U_e, _ = A1.fcm(X, X[order[:c]])
            per_c[c] = (U_e, A1.reach_membership(S, U_e.argmax(1), c))
        cache = {}
        for j, k in enumerate(kappas):
            best = None
            for c, (U_e, U_r) in per_c.items():
                lab = (k * U_e + (1 - k) * U_r).argmax(1)
                if np.bincount(lab, minlength=c).min() < 2:
                    continue
                key = (c, lab.tobytes())
                if key not in cache:
                    cache[key] = (silhouette_fast(D_euc, lab, c), silhouette_fast(D_reach, lab, c))
                se, sr = cache[key]
                s = k * se + (1 - k) * sr
                if best is None or s > best[0]:
                    best = (s, c, se, lab)
            out["f1"][i, j], out["cstar"][i, j], out["sil_euc"][i, j] = best[0], best[1], best[2]
            if keep_labels:
                labels_at[(i, j)] = best[3]
    return out, labels_at


def stage_sensitivity():
    c_star, lab_main = main_result()
    zetas = kappas = np.round(np.linspace(0, 1, 11), 1)
    real, labels_at = grid(REAL, zetas, kappas, keep_labels=True)
    i0, j0 = list(zetas).index(config.ZETA), list(kappas).index(config.KAPPA)
    assert abs(real["f1"][i0, j0] - json.loads((OUT / "main.json").read_text())["f1_modified_silhouette"]) < 1e-9
    rng = np.random.default_rng(config.SEED)        # same null draws as the validity stage
    null_f1 = np.full((config.N_NULL_SENSITIVITY, len(zetas), len(kappas)), np.nan)
    null_c = np.zeros_like(null_f1, dtype=int)
    for b in range(config.N_NULL_SENSITIVITY):
        g, _ = grid(build(reassign_conversations(OP, rng)), zetas, kappas)
        null_f1[b], null_c[b] = g["f1"], g["cstar"]
        if b % 200 == 0:
            log("sensitivity null", b)
    rows = []
    for i, z in enumerate(zetas):
        for j, k in enumerate(kappas):
            lab = labels_at[(i, j)]
            w, _ = A1.subgroup_weights(REAL["S_curr"], lab, real["cstar"][i, j])
            cons, flat, _ = A1.consensus(REAL["F_raw"], lab, w)
            rows.append({"zeta": z, "kappa": k, "cstar": int(real["cstar"][i, j]),
                         "sizes": "/".join(map(str, sorted(np.bincount(lab).tolist(), reverse=True))),
                         "f1": real["f1"][i, j], "f1_null_mean": float(np.nanmean(null_f1[:, i, j])),
                         "p": p_upper(real["f1"][i, j], null_f1[:, i, j]),
                         "null_share_same_cstar": float(np.mean(null_c[:, i, j] == real["cstar"][i, j])),
                         "sil_euc": real["sil_euc"][i, j], "ari_vs_main": adjusted_rand_score(lab_main, lab),
                         "tau_consensus_vs_flat": float(kendalltau(cons, flat).statistic)})
    sens = pd.DataFrame(rows)
    sens["p_bh"] = p_bh(sens.p)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    np.savez_compressed(OUT / "sensitivity_null.npz", f1=null_f1, cstar=null_c, zetas=zetas, kappas=kappas)
    summ = {"configs": len(sens), "cstar_counts": sens.cstar.value_counts().sort_index().to_dict(),
            "share_p_lt_05": float((sens.p < config.ALPHA).mean()), "share_p_bh_lt_05": float((sens.p_bh < config.ALPHA).mean()),
            "f1_range": [float(sens.f1.min()), float(sens.f1.max())],
            "ari_vs_main_median": float(sens.ari_vs_main.median()), "share_ari_gt_08": float((sens.ari_vs_main > 0.8).mean()),
            "tau_consensus_vs_flat_min": float(sens.tau_consensus_vs_flat.min())}
    dump("sensitivity.json", summ)
    log(json.dumps(summ, indent=1, default=float))


# ---------------------------------------------------------------- robustness: graded encoding
def stage_relative():
    d = build(OP, "relative")
    r = A1.run(d)
    c = r["c"]
    ix = validity_indices(d, *r["parts"][c])
    c_main, lab_main = main_result()
    rng = np.random.default_rng(config.SEED + 6)
    rows = []
    for b in range(200):
        dn = build(reassign_conversations(OP, rng), "relative")
        p = A1.all_c(dn)
        cs, s_max, _ = A1.select_c(dn, p)
        nix = validity_indices(dn, *p[c])
        rows.append({"cstar": cs, "mod_sil_max": s_max, "mod_sil": nix["mod_sil"], "sil_euc": nix["sil_euc"], "fpc_norm": nix["fpc_norm"]})
    null = pd.DataFrame(rows)
    res = {"c_star": c, "sizes": np.bincount(r["labels"]).tolist(), "indices": ix,
           "ari_vs_pairwise_partition": adjusted_rand_score(lab_main, r["labels"]) if c == c_main else None,
           "p": {k: p_upper(ix[k], null[k]) for k in ("mod_sil", "sil_euc", "fpc_norm")},
           "null_mean": null[["mod_sil", "sil_euc", "fpc_norm"]].mean().to_dict(),
           "null_cstar_counts": null.cstar.value_counts().to_dict()}
    dump("robust_relative.json", res)
    log(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"main": stage_main, "validity": stage_validity, "stability": stage_stability, "split": stage_split,
     "reliability": stage_reliability, "baselines": stage_baselines, "exposure": stage_exposure,
     "external": stage_external,
     "relative": stage_relative, "membership": stage_membership, "sensitivity": stage_sensitivity}[sys.argv[1]]()
