"""PRISM data: loading, participant selection, P-HFE profiles, and participant relations."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw(raw: Path = config.RAW) -> None:
    for name, digest in config.FILES.items():
        if sha256(raw / name) != digest:
            raise RuntimeError(f"{name} does not match the pinned PRISM revision; run 01_download.py")


# ---------------------------------------------------------------------------
# Loading and selection
# ---------------------------------------------------------------------------
def load_survey(raw: Path = config.RAW) -> pd.DataFrame:
    return pd.read_json(raw / "survey.jsonl", lines=True)


def load_opening_turns(raw: Path = config.RAW) -> pd.DataFrame:
    """Ratings of the opening turn, where up to four models answer the same prompt."""
    u = pd.read_json(raw / "utterances.jsonl", lines=True)
    cols = ["user_id", "conversation_id", "conversation_type", "model_name", "score", "if_chosen"]
    return u.loc[u["turn"] == config.OPENING_TURN, cols].reset_index(drop=True)


def select(op: pd.DataFrame):
    """Keep opening turns with at least two models and participants with the full set of conversations."""
    k = op.groupby("conversation_id")["model_name"].transform("nunique")
    op = op[k >= 2]
    n_conv = op.groupby("user_id")["conversation_id"].nunique()
    users = sorted(n_conv[n_conv >= config.MIN_CONVERSATIONS].index)
    op = op[op["user_id"].isin(users)].reset_index(drop=True)
    models = sorted(op["model_name"].unique())
    return op, users, models


# ---------------------------------------------------------------------------
# P-HFE profiles
# ---------------------------------------------------------------------------
def pairwise_outcomes(op: pd.DataFrame) -> pd.DataFrame:
    """Each opening turn implies a pairwise comparison between every two of its models:
    gamma = 1 for the model with the higher score, 0.5 for equal scores, 0 for the lower one."""
    a = op[["conversation_id", "user_id", "model_name", "score"]]
    m = a.merge(a, on=["conversation_id", "user_id"], suffixes=("", "_o"))
    m = m[m["model_name"] != m["model_name_o"]]
    gamma = np.where(m["score"] > m["score_o"], 1.0,
                     np.where(m["score"] == m["score_o"], config.TIE_VALUE, 0.0))
    return pd.DataFrame({"user_id": m["user_id"].to_numpy(), "model": m["model_name"].to_numpy(), "gamma": gamma})


def relative_outcomes(op: pd.DataFrame) -> pd.DataFrame:
    """Graded alternative: (s - min) / (max - min) within the opening turn, 0.5 if all scores are equal."""
    g = op.groupby("conversation_id")["score"]
    lo, hi = g.transform("min"), g.transform("max")
    rel = np.where(hi > lo, (op["score"] - lo) / (hi - lo).where(hi > lo, 1), config.TIE_VALUE)
    return pd.DataFrame({"user_id": op["user_id"].to_numpy(), "model": op["model_name"].to_numpy(), "gamma": rel})


def phfe_scores(out: pd.DataFrame, users, models) -> np.ndarray:
    """Score function of the baseline (Yang et al., 2026) applied to the P-HFE of every
    participant-model pair: f(h) = sum_l gamma_l p_l - (max_l gamma_l p_l - min_l gamma_l p_l) / |h|,
    where p_l is the relative frequency of outcome gamma_l. NaN where a model was never rated."""
    g = out.groupby(["user_id", "model", "gamma"]).size().rename("cnt").reset_index()
    tot = g.groupby(["user_id", "model"])["cnt"].transform("sum")
    g["gp"] = g["gamma"] * g["cnt"] / tot
    agg = g.groupby(["user_id", "model"])["gp"].agg(["sum", "max", "min", "size"])
    val = agg["sum"] - (agg["max"] - agg["min"]) / agg["size"]
    F = np.full((len(users), len(models)), np.nan)
    ui = agg.index.get_level_values(0).map({u: i for i, u in enumerate(users)}).to_numpy()
    mi = agg.index.get_level_values(1).map({m: j for j, m in enumerate(models)}).to_numpy()
    F[ui, mi] = val.to_numpy()
    return F


# ---------------------------------------------------------------------------
# Participant relations
# ---------------------------------------------------------------------------
def jaccard(B: np.ndarray) -> np.ndarray:
    B = B.astype(np.float64)
    inter = B @ B.T
    size = B.sum(axis=1)
    union = size[:, None] + size[None, :] - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(J, 1.0)
    return J


def chosen_similarity(op: pd.DataFrame, users, models) -> np.ndarray:
    """Current (behavioural) relation: Jaccard similarity of the sets of models a participant
    chose as best in their opening turns, the analogue of the winner sets of the original study."""
    c = op[op["if_chosen"]]
    B = np.zeros((len(users), len(models)), dtype=bool)
    B[c["user_id"].map({u: i for i, u in enumerate(users)}).to_numpy(),
      c["model_name"].map({m: j for j, m in enumerate(models)}).to_numpy()] = True
    return jaccard(B)


def background_similarity(survey: pd.DataFrame, users) -> np.ndarray:
    """Background relation: similarity of the stated importance of eight response attributes
    (survey), with a Gaussian kernel on standardized ratings and the median heuristic."""
    prefs = survey.set_index("user_id").loc[users, "stated_prefs"]
    X = np.array([[d.get(k, np.nan) if isinstance(d, dict) else np.nan for k in config.STATED_PREFS]
                  for d in prefs], dtype=float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    D2 = np.maximum((Z * Z).sum(1)[:, None] + (Z * Z).sum(1)[None, :] - 2 * Z @ Z.T, 0.0)
    S = np.exp(-D2 / np.median(D2[np.triu_indices(len(Z), 1)]))
    np.fill_diagonal(S, 1.0)
    return S


def pagerank(W: np.ndarray, d: float = config.PAGERANK_DAMPING, tol: float = 1e-10, it: int = 1000) -> np.ndarray:
    """PageRank on a weighted undirected network without self-loops."""
    A = W.copy()
    np.fill_diagonal(A, 0.0)
    out = A.sum(1)
    n = len(A)
    P = np.divide(A, out[:, None], out=np.zeros_like(A), where=out[:, None] > 0)
    x = np.full(n, 1.0 / n)
    for _ in range(it):
        x_new = d * (x @ P + x[out == 0].sum() / n) + (1 - d) / n
        if np.abs(x_new - x).sum() < n * tol:
            return x_new
        x = x_new
    return x


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------
def build_instance(op, users, models, background, encoding=config.PHFE_ENCODING) -> dict:
    """All inputs of Algorithm 1. `background` = (S_hist, pr_h), which depends on the survey only."""
    out = pairwise_outcomes(op) if encoding == "pairwise" else relative_outcomes(op)
    F_raw = phfe_scores(out, users, models)
    observed = ~np.isnan(F_raw)
    F = np.where(observed, F_raw, np.nanmean(F_raw, axis=0)[None, :])   # model-mean imputation
    S_curr = chosen_similarity(op, users, models)
    S_hist, pr_h = background
    return {"F": F, "F_raw": F_raw, "observed": observed, "S_hist": S_hist, "S_curr": S_curr,
            "pr_h": pr_h, "pr_c": pagerank(S_curr)}


def background_of(survey, users):
    S = background_similarity(survey, users)
    return S, pagerank(S)


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------
def reassign_conversations(op: pd.DataFrame, rng) -> pd.DataFrame:
    """Null model: conversations are dealt out again at random among participants, within each
    conversation type, so that everyone keeps the same number of conversations of each type.
    Model quality, which models met, and every judgement inside a conversation are kept; only the
    link between a participant and their own judgements is broken."""
    conv = op.drop_duplicates("conversation_id")[["conversation_id", "user_id", "conversation_type"]]
    new_user = conv["user_id"].to_numpy().copy()
    for idx in conv.groupby("conversation_type").indices.values():
        new_user[idx] = new_user[rng.permutation(idx)]
    out = op.copy()
    out["user_id"] = out["conversation_id"].map(dict(zip(conv["conversation_id"], new_user)))
    return out


def split_conversations(op: pd.DataFrame, rng):
    """Split every participant's conversations into two halves, alternating within conversation type."""
    conv = op.drop_duplicates("conversation_id")[["conversation_id", "user_id", "conversation_type"]].copy()
    conv["r"] = rng.random(len(conv))
    conv = conv.sort_values(["user_id", "conversation_type", "r"])
    start = conv["user_id"].map(dict(zip(conv["user_id"].unique(), rng.integers(0, 2, conv["user_id"].nunique()))))
    in_a = set(conv.loc[((conv.groupby("user_id").cumcount() + start) % 2) == 0, "conversation_id"])
    mask = op["conversation_id"].isin(in_a)
    return op[mask].reset_index(drop=True), op[~mask].reset_index(drop=True)
