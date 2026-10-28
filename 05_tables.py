"""Tables of the revised manuscript, numbered as in the manuscript, from the files in results/.

    python -I 05_tables.py

Writes results/tables/tables.md, tables.html (copy into Word), tables.json (titles and notes), and one CSV per table.
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

RES = config.RESULTS
OUT = RES / "tables"


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def f3(x, d=3):
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = s[1:] if float(s) == 0 else "−" + s[1:]       # typographic minus
    return s


def fsmall(x):
    """Three decimals, or four for values below 0.01 in magnitude."""
    return f3(x, 4 if abs(x) < 0.01 else 3)


def load(name):
    return json.loads((RES / name).read_text())


TABLES = []


def table(title, df, note=None, slug=None):
    TABLES.append((title, df, note, slug))
    df.to_csv(OUT / f"{slug}.csv", index=False)


# ---------------------------------------------------------------- Table 2.1 data
def t_data():
    d = load("data_summary.json")
    rows = [("Participants (six or more conversations)", f"{d['participants']:,}"),
            ("Language models", d["models"]),
            ("Conversations", f"{d['conversations']:,}"),
            ("Opening-turn ratings (1–100)", f"{d['opening_ratings']:,}"),
            ("Pairwise outcomes derived from opening turns", f"{d['pairwise_outcomes']:,}"),
            ("Opening turns with 4 / 3 / 2 models", " / ".join(f"{d['models_per_opening_turn'][k]:,}" for k in ("4", "3", "2"))),
            ("Share of tied outcomes", f"{100 * d['tie_share_of_outcomes']:.1f}%"),
            ("Density of the participant × model profile", f"{100 * d['density']:.1f}%"),
            ("Models rated per participant (mean)", f"{d['models_rated_per_participant_mean']:.1f}"),
            ("Outcomes per participant–model cell (median)", f"{d['outcomes_per_cell_median']:.0f}"),
            ("Participants per model (min–max)", "–".join(map(str, d["participants_per_model_min_max"]))),
            ("Conversation types (unguided / values / controversy)",
             " / ".join(f"{d['conversation_types'][k]:,}" for k in ("unguided", "values guided", "controversy guided")))]
    table("Table 2.1 Data used in this study (PRISM opening turns)", pd.DataFrame(rows, columns=["Item", "Value"]),
          slug="table2_1_data")


# ---------------------------------------------------------------- Table 2.2 mapping of the method to the data
def t_mapping():
    rows = [("Decision maker (DM)", "Expert or stakeholder", "PRISM participant who rated model responses"),
            ("Assessment", "Probabilistic hesitant fuzzy element (P-HFE)",
             "Win (1), tie (0.5), and loss (0) outcomes of a model in the participant's opening turns, with their relative frequencies"),
            ("Score of a P-HFE", "Score function with a fluctuation penalty", "Unchanged"),
            ("Background relation, weighted by ζ", "Historical background correlation",
             "Similarity of the stated importance of eight response attributes (survey)"),
            ("Current relation", "Current research correlation",
             "Jaccard similarity of the sets of models a participant chose as best"),
            ("κ", "Weight of distance vs reachability in clustering and in the modified silhouette", "Unchanged (0.6)"),
            ("μ", "Weight of cohesion vs size in subgroup weights", "Unchanged (0.5; 0–1 examined)"),
            ("Number of subgroups", "c ∈ {2, …, 8}, maximum modified silhouette", "Unchanged")]
    table("Table 2.2 SCO-FCM components and their counterparts in this study",
          pd.DataFrame(rows, columns=["Component", "Baseline [5]", "This study"]), slug="table2_2_mapping")


# ---------------------------------------------------------------- Table 3.1 validity against the null
def t_validity():
    v = load("validity.json")
    names = {"mod_sil": "Modified silhouette at c* (selection criterion)", "sil_euc": "Silhouette, Euclidean",
             "sil_reach": "Silhouette, current relation", "fpc_norm": "Normalized fuzzy partition coefficient",
             "xb": "Xie–Beni index (lower is better)", "db": "Davies–Bouldin index (lower is better)",
             "ch": "Calinski–Harabasz index"}
    rows = []
    for k, label in names.items():
        t = v["tests_at_cstar"][k]
        if abs(t["real"]) < 10:
            obs, nul = fsmall(t["real"]), f"{fsmall(t['null_mean'])} [{fsmall(t['null_q025'])}, {fsmall(t['null_q975'])}]"
        else:
            obs, nul = f"{t['real']:.1f}", f"{t['null_mean']:.1f} [{t['null_q025']:.1f}, {t['null_q975']:.1f}]"
        rows.append((label, obs, nul, fp(t["p"])))
    sel = pd.read_csv(RES / "validity_null.csv")["mod_sil_max"].dropna()
    t = v["tests_at_cstar"]["mod_sil_max_over_c"]
    rows.insert(1, ("Modified silhouette, best over c = 2–8 (selection-aware)", f3(t["real"]),
                    f"{sel.mean():.3f} [{sel.quantile(.025):.3f}, {sel.quantile(.975):.3f}]", fp(t["p"])))
    share = v["null_cstar_counts"].get("2", 0) / v["n_null"]
    table(f"Table 3.1 Cluster validity of the selected partition (c* = {v['c_star']}) against {v['n_null']:,} permutation nulls",
          pd.DataFrame(rows, columns=["Index", "Observed", "Null mean [95% range]", "p"]),
          note=(f"One-sided permutation p-values. The selection-aware row compares the best value over c = 2–8 with the best "
                f"value attained on each null dataset, as in Table 3.5. The null selected c* = 2 in {100 * share:.0f}% of replicates."),
          slug="table3_1_validity")


# ---------------------------------------------------------------- Table 3.2 stability and replication
def t_replication():
    st = pd.read_csv(RES / "stability.csv")
    s = pd.read_csv(RES / "split.csv")
    rel = pd.read_csv(RES / "reliability_summary.csv").set_index("kind")
    obs, nul = s[s.kind == "real"], s[s.kind == "permuted"]
    rows = [("Stability: ARI with the full-data partition, 80% subsamples (median)", f3(st.ari.median()), "–"),
            ("Stability: Jaccard similarity of the smaller subgroup (median)", f3(st.jac_min.median()), "–"),
            ("Replication: ARI between the partitions of halves A and B", f3(obs.ari_AB.mean(), 4), f3(nul.ari_AB.mean(), 4)),
            ("Largest subgroup difference, in-sample (half A)", f3(obs.in_max_gap.mean()), f3(nul.in_max_gap.mean())),
            ("Largest subgroup difference, held-out (half B)", f3(obs.held_max_gap.mean()), f3(nul.held_max_gap.mean())),
            ("Models (of 21) with a significant difference, in-sample", f3(obs.in_sig_bh.mean(), 2), f3(nul.in_sig_bh.mean(), 2)),
            ("Models (of 21) with a significant difference, held-out", f3(obs.held_sig_bh.mean(), 2), f3(nul.held_sig_bh.mean(), 2)),
            ("Correlation of subgroup preference profiles, A vs B", f3(obs.profile_r.mean()), f3(nul.profile_r.mean())),
            ("Split-half reliability of individual preferences (r per model)", f3(rel.loc["real", "r_model_mean"]),
             f"{f3(rel.loc['permuted', 'r_model_mean'])} [{f3(rel.loc['permuted', 'r_model_q025'])}, "
             f"{f3(rel.loc['permuted', 'r_model_q975'])}]")]
    table("Table 3.2 Stability and replication of the subgroups",
          pd.DataFrame(rows, columns=["Measure", "Observed", "Permutation null"]),
          note=("Stability: 200 subsamples of 80% of the participants. Replication: means over 50 random splits of each "
                "participant's conversations into halves A and B; subgroups formed on half A; significance by Kruskal–Wallis "
                "tests with Benjamini–Hochberg adjustment over 21 models. Reliability: null mean [95% range]."),
          slug="table3_2_replication")


# ---------------------------------------------------------------- Table 3.3 baselines
def t_baselines():
    b = pd.read_csv(RES / "baselines.csv", index_col=0)
    rows = []
    for name, r in b.iterrows():
        if r.n_clusters < 2:
            rows.append((name, "no clusters (all noise)", "–", "–", "–", f3(r.ari_vs_algorithm1)))
            continue
        sizes = "/".join(str(int(x)) for x in json.loads(r.sizes.replace("'", '"')))
        rows.append((name.replace("kappa", "κ"), sizes, f3(r.sil_euc), f3(r.sil_euc_null_mean), fp(r.sil_euc_p),
                     "–" if name.startswith("Algorithm 1") else f3(r.ari_vs_algorithm1)))
    table("Table 3.3 Algorithm 1 and other clustering methods at c = 2, each against its own permutation null",
          pd.DataFrame(rows, columns=["Method", "Sizes", "Silhouette (Euclidean)", "Null mean", "p", "ARI with Algorithm 1"]),
          note="Silhouettes are computed on the same P-HFE profiles; 100 null replicates per method.", slug="table3_3_baselines")


# ---------------------------------------------------------------- Table 3.4 external validation
def t_external():
    a = pd.read_csv(RES / "external_association.csv")
    names = {"region": "Region", "age": "Age", "gender": "Gender", "education": "Education", "employment": "Employment",
             "marital": "Marital status", "english": "English proficiency", "religion": "Religion", "ethnicity": "Ethnicity",
             "lm_familiarity": "Familiarity with LLMs", "lm_use": "Frequency of LLM use"}
    rows = [(names[r.variable], f"{r.n:,}", r.groups, f3(r.cramers_v_cluster), fp(r.p_cluster), fp(r.p_cluster_holm),
             f3(r.R2_profiles), fp(r.p_profiles), fp(r.p_profiles_holm)) for r in a.itertuples()]
    table("Table 3.4 Sociodemographic validation of the subgroups",
          pd.DataFrame(rows, columns=["Attribute", "n", "Groups", "Cramér's V", "p", "p (Holm)",
                                      "R² of profiles", "p", "p (Holm)"]),
          note=("Attributes were not used for clustering. Cramér's V: association with subgroup membership; R²: share of the "
                "variation in preference profiles explained by the attribute (PERMANOVA). 999 permutations each."),
          slug="table3_4_external")


# ---------------------------------------------------------------- Table 3.5 sensitivity and robustness
def t_sensitivity():
    s = pd.read_csv(RES / "sensitivity.csv")
    j = load("sensitivity.json")
    rob = load("robust_relative.json")
    main = load("main.json")
    mu = main["effect_of_mu"]
    n_null = np.load(RES / "sensitivity_null.npz")["f1"].shape[0]
    sig = s[s.p < config.ALPHA]
    rows = [("Configurations (ζ, κ ∈ {0, 0.1, …, 1}; μ = 0.5)", len(s)),
            ("Selected number of subgroups", ", ".join(f"c* = {k} in {v}" for k, v in j["cstar_counts"].items())),
            ("Modified silhouette at c* (min–max)", f"{j['f1_range'][0]:.3f}–{j['f1_range'][1]:.3f}"),
            ("Configurations with p < 0.05, unadjusted", f"{len(sig)} ({100 * j['share_p_lt_05']:.0f}%), all with κ ≥ {sig.kappa.min():.1f}"),
            ("Configurations with p < 0.05 after Benjamini–Hochberg (121 tests)", f"{int((s.p_bh < config.ALPHA).sum())}"),
            ("Smallest p, unadjusted / adjusted", f"{fp(s.p.min())} / {fp(s.p_bh.min())}"),
            ("ARI with the partition at ζ = 0.4, κ = 0.6 (median; min–max)",
             f"{s.ari_vs_main.median():.2f}; {s.ari_vs_main.min():.2f}–{s.ari_vs_main.max():.2f}"),
            ("Kendall's τ, consensus vs pooled ranking (min over configurations)", f3(j["tau_consensus_vs_flat_min"])),
            ("Subgroup weights for μ = 0 to 1 (larger subgroup)", f"{mu['0.0']['weights'][0]:.3f} to {mu['1.0']['weights'][0]:.3f}"),
            ("Graded instead of pairwise encoding: modified silhouette (p)",
             f"{rob['indices']['mod_sil']:.3f} ({fp(rob['p']['mod_sil'])}); ARI with the main partition {rob['ari_vs_pairwise_partition']:.2f}")]
    table("Table 3.5 Sensitivity to the balancing parameters and robustness checks", pd.DataFrame(rows, columns=["Measure", "Value"]),
          note=f"Each configuration was compared with the same {n_null:,} permutation nulls as Table 3.1; the graded encoding with 200.",
          slug="table3_5_sensitivity")


# ---------------------------------------------------------------- Table A1 representation
def t_representation():
    r = pd.read_csv(RES / "external_representation.csv")
    names = {"region": "Region", "age": "Age", "gender": "Gender", "religion": "Religion", "ethnicity": "Ethnicity"}
    rows = [(names[x.variable], x.group, x.n, f3(x.tau_flat), f3(x.tau_consensus),
             f"{x.tau_null_mean:.3f} [{x.tau_null_q025:.3f}, {x.tau_null_q975:.3f}]", fp(x.p_underrepresented), fp(x.p_holm))
            for x in r.itertuples()]
    table("Table A1 Representation of sociodemographic groups by the pooled and the SCO-FCM consensus rankings",
          pd.DataFrame(rows, columns=["Attribute", "Group", "n", "τ, pooled", "τ, consensus",
                                      "τ, random group [95% range]", "p", "p (Holm)"]),
          note="Kendall's τ between a group's model ranking and the ranking named in the column; p tests whether the group is "
               "represented worse than a random group of the same size (999 permutations).", slug="tableA1_representation")


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def to_html(df):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    for f in (t_data, t_mapping, t_validity, t_replication, t_baselines, t_external, t_sensitivity, t_representation):
        f()
    md, ht = ["# Tables of the revised manuscript\n"], []
    for title, df, note, _ in TABLES:
        md += [f"**{title}**\n", to_markdown(df), "" if note is None else f"\n*Note.* {note}", ""]
        ht.append(f"<p class='cap'>{html.escape(title)}</p>{to_html(df)}"
                  + ("" if note is None else f"<p class='note'><i>Note.</i> {html.escape(note)}</p>"))
    (OUT / "tables.md").write_text("\n".join(md))
    (OUT / "tables.json").write_text(json.dumps({slug: {"title": title, "note": note} for title, _, note, slug in TABLES},
                                                  indent=2, ensure_ascii=False))
    style = ("body{font-family:Cambria,Georgia,serif;font-size:10pt;max-width:900px;margin:24px auto;padding:0 16px;"
             "color:#0b0b0b;background:#ffffff}table{border-collapse:collapse;margin:6px 0 4px;width:100%}"
             "th,td{padding:3px 8px;text-align:left;vertical-align:top}thead th{border-top:1.2px solid #0b0b0b;"
             "border-bottom:0.8px solid #0b0b0b}tbody tr:last-child td{border-bottom:1.2px solid #0b0b0b}"
             ".cap{font-weight:bold;margin:26px 0 4px}.note{font-size:9pt;color:#52514e;margin:2px 0 0}")
    (OUT / "tables.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Manuscript tables</title>"
                                     f"<style>{style}</style></head><body>{''.join(ht)}</body></html>")
    print("wrote", OUT / "tables.md", "and tables.html;", len(TABLES), "tables")


if __name__ == "__main__":
    main()
