"""Figures of the revised manuscript, drawn at their printed width (16 cm text width, A4).

    python -I 04_figures.py

Writes PNG (600 dpi) and PDF files to figures/. Every figure has a table twin (05_tables.py).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

import config  # noqa: E402

RES = config.RESULTS
FIG = config.ROOT / "figures"
FIG.mkdir(exist_ok=True)
CM = 1 / 2.54
WIDTH = 16 * CM

# roles: observed data = categorical slot 1, SCO-FCM consensus = slot 2, permutation null = de-emphasis gray
OBS, CONS = "#2a78d6", "#eb6834"
NULL_FILL, NULL_MARK = "#d3d2cb", "#898781"
INK, INK2, AXIS, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3,
    "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "axes.titlelocation": "left",
    "axes.titlepad": 6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------------------------------------------------------------- 0. research framework (Figure 2.1)
def fig_framework():
    """Three stages: data representation, Algorithm 1, and validation. Coordinates in cm."""
    LIGHT, DARK, STAGE = "#dbe9fb", "#1c5cab", "#f6f6f3"
    H = 12.8                       # layout height in cm; drawn at 12.0 cm, so boxes compress slightly but text does not
    with plt.rc_context({"mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "mathtext.cal": "Arial"}):
        fig = plt.figure(figsize=(WIDTH, 12.0 * CM))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 16)
        ax.set_ylim(0, H)
        ax.axis("off")

        def box(x0, y0, w, h, text, fc="white", ec=NULL_MARK, tc=INK, lw=0.6, fs=7):
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=ec, lw=lw))
            ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.2)

        def arrow(x0, y0, x1, y1, lw=0.8, scale=7):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=scale, lw=lw,
                                         color=INK2, shrinkA=0, shrinkB=0))

        def stage(y0, y1, title):
            ax.add_patch(FancyBboxPatch((0.2, y0), 15.6, y1 - y0, boxstyle="round,pad=0,rounding_size=0.2", fc=STAGE, ec="none"))
            ax.text(0.55, y1 - 0.3, title, ha="left", va="center", fontsize=8, fontweight="bold", color=INK)

        # stage 1: data representation
        stage(8.6, 12.6, "1   Data representation (Sections 2.2–2.4)")
        xs, w = (0.6, 5.65, 10.7), 4.7
        inputs = ("PRISM opening turns:\n1,267 participants, 21 LLMs,\n28,602 ratings (1–100)",
                  "Survey: stated importance\nof eight response attributes",
                  "Model chosen as best\nin each opening turn")
        derived = ("Pairwise outcomes (win 1, tie 0.5,\nloss 0) → P-HFE scores, Eq. (1):\nprofile matrix $F$",
                   "Background relation $S_\\mathrm{hist}$:\nGaussian kernel, Eq. (2)",
                   "Current relation $S_\\mathrm{curr}$:\nJaccard similarity, Eq. (3)")
        for x, t_in, t_out in zip(xs, inputs, derived):
            box(x, 11.0, w, 1.0, t_in)
            box(x, 9.75, w, 1.0, t_out, fc=LIGHT, ec="#2a78d6")
            arrow(x + w / 2, 11.0, x + w / 2, 10.75)
        box(5.65, 8.85, 9.75, 0.55,
            "PageRank centralities → priority $\\pi = \\zeta\\, PR_\\mathrm{hist} + (1 - \\zeta)\\, PR_\\mathrm{curr}$, Eq. (4)",
            fc=LIGHT, ec="#2a78d6")
        for x in (8.0, 13.05):
            arrow(x, 9.75, x, 9.4)

        arrow(8.0, 8.6, 8.0, 8.1, lw=1.2, scale=9)
        ax.text(8.3, 8.35, "$F$, $S_\\mathrm{curr}$, $\\pi$", ha="left", va="center", fontsize=7, color=INK2)

        # stage 2: Algorithm 1
        stage(4.7, 8.1, "2   Algorithm 1: adapted SCO-FCM (Section 2.5)")
        xs2, w2 = (0.6, 4.375, 8.15, 11.925), 3.475
        steps = ("Initial centers: highest $\\pi$\nfirst, then the least\nreachable participant",
                 "FCM smoothed by the\ncurrent relation ($\\kappa$),\nEq. (5), $c$ = 2, …, 8",
                 "Modified silhouette ($\\kappa$),\nEq. (6); $c$* = best $c$",
                 "Subgroup weights from\ncohesion and size ($\\mu$),\nEq. (7)")
        for k, (x, t) in enumerate(zip(xs2, steps)):
            box(x, 6.3, w2, 1.2, t, fc=LIGHT, ec="#2a78d6")
            if k:
                arrow(xs2[k - 1] + w2, 6.9, x, 6.9)
        box(8.15, 4.95, 7.25, 0.8, "Consensus ranking, Eq. (8),\ncompared with the flat ranking", fc=LIGHT, ec="#2a78d6")
        arrow(xs2[3] + w2 / 2, 6.3, xs2[3] + w2 / 2, 5.75)

        arrow(8.0, 4.7, 8.0, 4.2, lw=1.2, scale=9)
        ax.text(8.3, 4.45, "subgroups ($c$*, memberships) and consensus ranking", ha="left", va="center", fontsize=7, color=INK2)

        # stage 3: validation
        stage(0.2, 4.2, "3   Validation: discovered or constructed? (Section 2.6)")
        xs3, w3 = (0.6, 3.62, 6.64, 9.66, 12.68), 2.72
        tests = ("Permutation null:\n1,000 null datasets\n(Section 3.3)",
                 "Stability and\nsplit-half replication\n(Section 3.4)",
                 "Nine other\nclustering methods\n(Section 3.5)",
                 "11 sociodemographic\nattributes and\nrepresentation\n(Sections 3.6–3.7)",
                 "Sensitivity to\n$\\zeta$ and $\\kappa$:\n121 settings\n(Section 3.8)")
        for x, t in zip(xs3, tests):
            box(x, 2.1, w3, 1.45, t, fc=DARK, ec=DARK, tc="white")
            arrow(x + w3 / 2, 2.1, x + w3 / 2, 1.45)
        box(0.6, 0.45, 14.8, 1.0, "Ontological, epistemological, and axiological discussion (Sections 3.9–3.11):\n"
            "are the subgroups discovered in the data or constructed by the method?", ec=DARK, lw=1.0, fs=7.5)
        save(fig, "fig_framework")


# ---------------------------------------------------------------- 1. validity against the null
def fig_null_validity():
    v = json.loads((RES / "validity.json").read_text())["tests_at_cstar"]
    null = pd.read_csv(RES / "validity_null.csv")
    panels = [("mod_sil", "(a) Modified silhouette\n(selection criterion of Algorithm 1)", "Modified silhouette"),
              ("sil_euc", "(b) Euclidean silhouette", "Silhouette (Euclidean)"),
              ("ch", "(c) Calinski–Harabasz index", "Calinski–Harabasz index")]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.8 * CM), layout="constrained")
    for ax, (key, title, xlabel) in zip(axes, panels):
        x = null[key].dropna().to_numpy()
        real, p = v[key]["real"], v[key]["p"]
        counts, _, _ = ax.hist(x, bins=28, color=NULL_FILL, edgecolor="white", linewidth=0.4)
        ax.axvline(real, color=OBS, linewidth=1.4)
        lo, hi = min(x.min(), real), max(x.max(), real)
        ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_ylim(0, counts.max() * 1.42)          # headroom so the label sits above the bars
        ax.text(real - 0.02 * (hi - lo), counts.max() * 1.38, f"Observed {real:.3g}\n{fmt_p(p)}",
                ha="right", va="top", fontsize=7.5, color=INK)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.yaxis.grid(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Permutation replicates")
    save(fig, "fig_null_validity")


# ---------------------------------------------------------------- 2. fuzzy memberships
def fig_membership():
    m = np.load(RES / "membership.npz")
    real, null = m["real"], m["null"].ravel()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 4.8 * CM), layout="constrained",
                               gridspec_kw={"width_ratios": [1.1, 1]})
    counts, _, _ = a.hist(real, bins=np.linspace(0.5, 1.0, 51), color=OBS, edgecolor="white", linewidth=0.4)
    a.set_xlim(0.5, 1.0)
    a.set_ylim(0, counts.max() * 1.15)
    a.set_title("(a) Largest membership of each participant (c = 2)")
    a.set_xlabel("Largest membership degree")
    a.set_ylabel("Participants")
    a.annotate(f"all {len(real):,} participants lie\nbetween {real.min():.2f} and {real.max():.2f}:\nmembership is split almost evenly",
               xy=(0.512, counts.max() * 0.97), xytext=(0.6, counts.max() * 0.95), va="top", fontsize=7.5, color=INK2,
               arrowprops={"arrowstyle": "-", "color": AXIS, "linewidth": 0.6, "shrinkA": 2, "shrinkB": 2})
    a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5\nfully fuzzy", "0.6", "0.7", "0.8", "0.9", "1.0\ncrisp"])
    a.yaxis.grid(True)
    bins = np.linspace(0.5, max(real.max(), np.quantile(null, 0.999)) + 0.002, 40)
    b.hist(null, bins=bins, density=True, color=NULL_FILL, edgecolor="white", linewidth=0.4, label="Permutation null")
    b.hist(real, bins=bins, density=True, histtype="step", color=OBS, linewidth=1.4, label="Observed")
    b.set_title("(b) Observed vs permutation null (zoomed)")
    b.set_xlabel("Largest membership degree")
    b.set_ylabel("Density")
    b.yaxis.grid(True)
    b.legend(loc="upper right", handlelength=1.4, borderaxespad=0.2)
    for ax in (a, b):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    save(fig, "fig_membership")


# ---------------------------------------------------------------- 3. in-sample vs held-out
def fig_insample_heldout():
    s = pd.read_csv(RES / "split.csv")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.0 * CM), layout="constrained")
    specs = [(axes[0], ("in_max_gap", "held_max_gap"), "(a) Largest difference between subgroup means",
              "Difference in P-HFE score"),
             (axes[1], ("in_sig_bh", "held_sig_bh"), "(b) Models with a significant subgroup difference",
              "Significant models (of 21)")]
    for ax, (k_in, k_out), title, ylabel in specs:
        for xi, key in enumerate((k_in, k_out)):
            for dx, kind, color in ((-0.17, "real", OBS), (0.17, "permuted", NULL_MARK)):
                y = s.loc[s.kind == kind, key].to_numpy(float)
                jitter = rng.uniform(-0.07, 0.07, len(y))
                ax.scatter(xi + dx + jitter, y, s=9, color=color, edgecolors="white", linewidths=0.4, zorder=3)
                med = np.median(y)
                ax.plot([xi + dx - 0.11, xi + dx + 0.11], [med, med], color=INK, linewidth=1.2, zorder=4)
        ax.set_xticks([0, 1], ["In-sample\n(subgroups formed\nand tested on half A)", "Held-out\n(subgroups from A,\ntested on half B)"])
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=-0.6)
    axes[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handles = [Line2D([], [], marker="o", linestyle="", markersize=4.5, color=OBS, markeredgecolor="white", label="Observed data"),
               Line2D([], [], marker="o", linestyle="", markersize=4.5, color=NULL_MARK, markeredgecolor="white", label="Permutation null"),
               Line2D([], [], color=INK, linewidth=1.2, label="Median of 50 splits")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, handletextpad=0.4, columnspacing=1.6)
    save(fig, "fig_insample_heldout")


# ---------------------------------------------------------------- 4. representation of real-world groups
def fig_representation():
    r = pd.read_csv(RES / "external_representation.csv")
    names = {"region": "Region", "age": "Age", "gender": "Gender", "religion": "Religion", "ethnicity": "Ethnicity"}
    panels = [("region", "age"), ("gender", "religion", "ethnicity")]       # two columns keep the figure short

    def layout(variables):
        rows, labels, headers, y = [], [], [], 0
        for var in variables:
            g = r[r.variable == var]
            g = g.sort_values("group") if var == "age" else g.sort_values("n", ascending=False)
            headers.append((y, names[var]))
            y += 1
            for _, row in g.iterrows():
                short = (row.group.replace(" years old", "").replace("Latin America and the Caribbean", "Latin America & Caribbean")
                         .replace("Australia and New Zealand", "Australia & New Zealand").replace("Non-binary / third gender", "Non-binary"))
                rows.append((y, row))
                labels.append((y, f"{short} (n = {row.n})"))
                y += 1
            y += 0.4
        return rows, labels, headers, y

    layouts = [layout(p) for p in panels]
    height = max(lay[3] for lay in layouts)
    xmin = np.floor((r.tau_null_q025.min() - 0.02) * 20) / 20
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, (0.34 * height + 2.4) * CM), layout="constrained")
    for ax, (rows, labels, headers, _) in zip(axes, layouts):
        for yy, row in rows:
            ax.plot([row.tau_null_q025, row.tau_null_q975], [yy, yy], color=NULL_FILL, linewidth=3.8, solid_capstyle="round", zorder=1)
            ax.plot(row.tau_flat, yy, "o", markersize=4.5, color=OBS, markeredgecolor="white", markeredgewidth=0.7, zorder=3)
            ax.plot(row.tau_consensus, yy, "|", markersize=7, color=CONS, markeredgewidth=1.3, zorder=4)
            if row.p_underrepresented < 0.10:
                ax.text(xmin + 0.01, yy, fmt_p(row.p_underrepresented), va="center", ha="left", fontsize=6.5, color=INK2)
        ticks = sorted(labels + headers)
        ax.set_yticks([yy for yy, _ in ticks], [t for _, t in ticks])
        header_rows = {yy for yy, _ in headers}
        for tick, (yy, _) in zip(ax.get_yticklabels(), ticks):
            if yy in header_rows:
                tick.set_fontweight("bold")
                tick.set_color(INK)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_ylim(height - 0.6, -0.8)
        ax.set_xlim(xmin, 1.0)
        ax.xaxis.grid(True)
        ax.spines["left"].set_visible(False)
    fig.supxlabel("Kendall's τ between the group's model ranking and the pooled ranking", fontsize=7.5)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5, color=OBS, markeredgecolor="white", label="Pooled (flat) ranking"),
               Line2D([], [], marker="|", linestyle="", markersize=8, markeredgewidth=1.4, color=CONS, label="SCO-FCM weighted consensus"),
               Line2D([], [], color=NULL_FILL, linewidth=4.2, solid_capstyle="round", label="95% range for a random group of equal size")]
    fig.legend(handles=handles, loc="outside upper center", ncols=3, handlelength=2.2, handletextpad=0.7, columnspacing=1.6)
    save(fig, "fig_representation")


# ---------------------------------------------------------------- 5. sensitivity to zeta and kappa
def fig_sensitivity():
    s = pd.read_csv(RES / "sensitivity.csv")
    zetas, kappas = np.sort(s.zeta.unique()), np.sort(s.kappa.unique())
    grid = lambda col: s.pivot(index="zeta", columns="kappa", values=col).loc[zetas, kappas].to_numpy()
    seq = LinearSegmentedColormap.from_list("seq_blue", BLUES)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 6.0 * CM), layout="constrained")
    edges_k = np.append(kappas - 0.05, kappas[-1] + 0.05)
    edges_z = np.append(zetas - 0.05, zetas[-1] + 0.05)

    def colorbar(mappable, ax, label, ticks=None):
        cb = fig.colorbar(mappable, ax=ax, location="bottom", shrink=0.9, aspect=28, pad=0.03, ticks=ticks)
        cb.set_label(label)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
        return cb

    c = grid("cstar")
    levels = sorted(np.unique(c))
    cmap_c = ListedColormap([BLUES[[2, 7, 11][i]] for i in range(len(levels))])
    norm_c = BoundaryNorm(np.arange(levels[0] - 0.5, levels[-1] + 1.5), cmap_c.N)
    m0 = axes[0].pcolormesh(edges_k, edges_z, c, cmap=cmap_c, norm=norm_c, edgecolors="white", linewidth=0.5)
    colorbar(m0, axes[0], "Number of subgroups c*", ticks=levels)
    axes[0].set_title("(a) Selected number of subgroups")

    f1 = grid("f1")
    m1 = axes[1].pcolormesh(edges_k, edges_z, f1, cmap=seq, edgecolors="white", linewidth=0.5)
    sig_raw, sig_bh = grid("p") < config.ALPHA, grid("p_bh") < config.ALPHA
    kk, zz = np.meshgrid(kappas, zetas)
    axes[1].scatter(kk[sig_raw & ~sig_bh], zz[sig_raw & ~sig_bh], s=9, facecolors="white", edgecolors=INK, linewidths=0.6)
    axes[1].scatter(kk[sig_bh], zz[sig_bh], s=9, color=INK, edgecolors="white", linewidths=0.6)
    colorbar(m1, axes[1], "Modified silhouette at c*")
    axes[1].set_title("(b) Validity vs permutation null")

    a = grid("ari_vs_main")
    m2 = axes[2].pcolormesh(edges_k, edges_z, a, cmap=seq, vmin=0, vmax=1, edgecolors="white", linewidth=0.5)
    axes[2].add_patch(plt.Rectangle((config.KAPPA - 0.05, config.ZETA - 0.05), 0.1, 0.1, fill=False, edgecolor=INK, linewidth=1.0))
    colorbar(m2, axes[2], "ARI with the main partition (outlined)")
    axes[2].set_title("(c) Agreement with main partition")

    for ax in axes:
        ax.set_xlabel("κ")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("ζ", rotation=0, labelpad=8)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=3.6, markerfacecolor="white", markeredgecolor=INK,
                      markeredgewidth=0.6, label="p < 0.05 (unadjusted)"),
               Line2D([], [], marker="o", linestyle="", markersize=3.6, color=INK, label="p < 0.05 after BH correction (none)")]
    fig.legend(handles=handles, loc="outside upper center", ncols=2, handletextpad=0.3, columnspacing=1.6)
    save(fig, "fig_sensitivity")


if __name__ == "__main__":
    which = sys.argv[1:] or ["framework", "null_validity", "membership", "insample_heldout", "representation", "sensitivity"]
    for name in which:
        globals()[f"fig_{name}"]()
