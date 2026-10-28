# Figure captions (draft)

All figures are drawn at the printed text width (16 cm) by `04_figures.py`; PNG at 600 dpi and PDF.
Blue marks the observed data, gray the permutation null, and orange the SCO-FCM weighted consensus.

**fig_framework** (Figure 2.1) — Research process: data representation, Algorithm 1, and validation.

**fig_null_validity** — Permutation null distributions of three cluster-validity indices for the
partition selected by Algorithm 1 (c* = 2, ζ = 0.4, κ = 0.6). Gray bars: 1,000 replicates in which
conversations were reassigned at random among participants within each conversation type; blue line:
observed value. One-sided permutation p-values.

**fig_membership** — (a) Largest fuzzy membership degree of each of the 1,267 participants at c* = 2,
shown on its full range: 0.5 means equal membership in both subgroups and 1 means crisp assignment.
(b) The same distribution (blue outline) against 50 permutation-null replicates (gray), zoomed.

**fig_insample_heldout** — Subgroup differences in-sample and held-out. In each of 50 splits, every
participant's conversations were divided at random into halves A and B; subgroups were formed on half A
and their differences in model scores were tested on half A (in-sample) and on half B (held-out).
(a) Largest difference between subgroup means over the 21 models. (b) Number of models with a
significant subgroup difference (Kruskal–Wallis, Benjamini–Hochberg-adjusted p < 0.05). Black line: median.

**fig_representation** — Representation of sociodemographic groups. Kendall's τ between each group's model
ranking and the pooled (flat) ranking (blue dot) or the SCO-FCM weighted consensus (orange tick). Gray bar:
95% range of τ for a random group of the same size (999 permutations). One-sided p-values are shown where
p < 0.10; none is significant after Holm correction.

**fig_sensitivity** — Sensitivity of Algorithm 1 to ζ and κ (μ = 0.5). (a) Selected number of subgroups.
(b) Modified silhouette at c*; open circles mark configurations where it exceeds the same 1,000 permutation
nulls at unadjusted p < 0.05, none of which survives Benjamini–Hochberg correction over the 121
configurations. (c) Adjusted Rand index between the partition of each configuration and the partition at
ζ = 0.4, κ = 0.6 (outlined).
