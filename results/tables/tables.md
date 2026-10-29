# Tables of the revised manuscript

**Table 2.1 Data used in this study (PRISM opening turns)**

| Item | Value |
|---|---|
| Participants (six or more conversations) | 1,267 |
| Language models | 21 |
| Conversations | 7,607 |
| Opening-turn ratings (1–100) | 28,602 |
| Pairwise outcomes | 80,618 |
| Outcomes per observed cell (median) | 3 |


**Table 2.2 SCO-FCM components and their counterparts in this study**

| Component | Baseline [6] | This study |
|---|---|---|
| Decision maker (DM) | Expert or stakeholder | PRISM participant |
| Assessment | Probabilistic hesitant fuzzy element (P-HFE) | Win (1), tie (0.5), and loss (0) outcomes in opening turns, with relative frequencies |
| Background relation, weighted by ζ | Historical background correlation | Similarity of the stated importance of eight response attributes |
| Current relation | Current research correlation | Jaccard similarity of the models chosen as best |
| Score function, κ, μ, and c | Fluctuation penalty; κ: distance vs reachability; μ: membership density vs size; c from 2 to k/2 for k DMs | Same function and values (κ = 0.6, μ = 0.5); κ also smooths memberships (Eq. 5); μ: cohesion vs size; c ∈ {2, …, 8} |


**Table 3.1 Cluster validity of the selected partition (c* = 2) against 1,000 permutation nulls**

| Index | Observed | Null mean [95% range] | p |
|---|---|---|---|
| Modified silhouette at c* | 0.042 | 0.037 [0.031, 0.042] | 0.042 |
| Modified silhouette, best c (selection-aware) | 0.042 | 0.037 [0.031, 0.043] | 0.045 |
| Silhouette, Euclidean | 0.027 | 0.021 [0.017, 0.025] | 0.003 |
| Silhouette, current relation | 0.064 | 0.060 [0.050, 0.070] | 0.254 |
| Normalized FPC | 0.0022 | 0.0015 [0.0011, 0.0019] | < 0.001 |
| Xie–Beni (lower is better) | 405.6 | 743.7 [527.9, 1053.7] | < 0.001 |
| Davies–Bouldin (lower is better) | 6.126 | 6.693 [6.118, 7.274] | 0.031 |
| Calinski–Harabasz | 32.1 | 27.1 [22.7, 32.6] | 0.035 |

*Note.* One-sided permutation p-values. The selection-aware row compares the best value over c = 2–8 with the best value attained on each null dataset. The null selected c* = 2 in 86% of replicates.

**Table 3.2 Stability and replication of the subgroups**

| Measure | Observed | Permutation null |
|---|---|---|
| Stability: ARI with the full-data partition (median) | 0.777 | – |
| Stability: Jaccard, smaller subgroup (median) | 0.873 | – |
| Replication: ARI between halves A and B | −0.0001 | 0.0003 |
| Largest difference, in-sample (A) | 0.506 [0.371, 0.563] | 0.464 [0.342, 0.553] |
| Largest difference, held-out (B) | 0.066 [0.042, 0.094] | 0.065 [0.039, 0.089] |
| Largest ε², in-sample (A) | 0.375 [0.196, 0.491] | 0.323 [0.187, 0.450] |
| Largest ε², held-out (B) | 0.010 [0.005, 0.018] | 0.009 [0.004, 0.018] |
| Profile correlation, A vs B | 0.129 [−0.218, 0.671] | 0.021 [−0.342, 0.476] |
| Reliability of individual preferences (r) | 0.037 | 0.000 [−0.022, 0.027] |

*Note.* Stability: 200 subsamples of 80% of the participants. Replication: means [2.5th, 97.5th percentiles] over 50 splits into halves A and B, subgroups formed on A; ε² = H/(n − 1) of Kruskal–Wallis tests; profile correlation: correlation over the models between each subgroup's deviations from the mean scores in A and in B, weighted by subgroup size. Reliability: null mean [95% range].

**Table 3.3 Algorithm 1 and other clustering methods, each against its own permutation null**

| Method | Sizes | Silhouette | Null mean | p | ARI with Alg. 1 |
|---|---|---|---|---|---|
| Algorithm 1 (SCO-FCM) | 723/544 | 0.027 | 0.021 | 0.020 | – |
| No reachability (κ = 1) | 683/584 | 0.051 | 0.046 | 0.020 | 0.227 |
| FCM, random init | 676/591 | 0.051 | 0.045 | 0.020 | 0.217 |
| K-Means | 954/313 | 0.058 | 0.060 | 0.584 | 0.113 |
| GMM (diagonal) | 958/309 | 0.057 | 0.057 | 0.535 | 0.111 |
| Ward | 1152/115 | 0.032 | 0.041 | 0.832 | −0.001 |
| Spectral, current relation | 686/581 | 0.022 | 0.020 | 0.099 | 0.499 |
| Spectral, kNN profiles | 935/332 | 0.055 | 0.055 | 0.574 | −0.001 |
| HDBSCAN | no clusters (all noise) | – | – | – | 0.000 |
| Louvain, current relation | 448/370/227/222 | 0.014 | 0.014 | 0.594 | 0.006 |
| Published SCO-FCM, objective | one subgroup (all participants) | – | – | – | 0.000 |
| Published SCO-FCM, Algorithm 3 | 824/443 | 0.054 | 0.057 | 0.624 | 0.176 |

*Note.* Euclidean silhouettes on the same P-HFE profiles; 100 null replicates per method. All methods except HDBSCAN and Louvain use c = 2. Modularity of the Louvain communities on the current relation: 0.096 (null mean 0.097, p = 0.733). Published SCO-FCM: the baseline's initial centers and reachability matrix (the background and current relations combined with ζ) with its soft-constrained objective, under either direction of the consensus cost, or with its Algorithm 3, which adds the cost to the distance; with the inverse cost, Algorithm 3 also placed all participants in one subgroup.

**Table 3.4 Sociodemographic validation of the subgroups**

| Attribute | n | Groups | Cramér's V | p (Holm) | R² of profiles | p (Holm) |
|---|---|---|---|---|---|---|
| Region | 1,264 | 9 | 0.063 | 1.000 | 0.008 | 0.649 |
| Age | 1,267 | 6 | 0.087 | 0.750 | 0.005 | 1.000 |
| Gender | 1,267 | 3 | 0.033 | 1.000 | 0.002 | 0.810 |
| Education | 1,225 | 5 | 0.050 | 1.000 | 0.003 | 1.000 |
| Employment | 1,251 | 7 | 0.070 | 1.000 | 0.005 | 1.000 |
| Marital status | 1,233 | 3 | 0.096 | 0.055 | 0.002 | 1.000 |
| English proficiency | 1,262 | 4 | 0.027 | 1.000 | 0.002 | 1.000 |
| Religion | 1,216 | 5 | 0.031 | 1.000 | 0.003 | 1.000 |
| Ethnicity | 1,204 | 6 | 0.072 | 1.000 | 0.004 | 1.000 |
| Familiarity with LLMs | 1,267 | 3 | 0.063 | 0.756 | 0.002 | 1.000 |
| Frequency of LLM use | 1,063 | 5 | 0.086 | 0.756 | 0.004 | 1.000 |

*Note.* Attributes were not used for clustering. Cramér's V: association with subgroup membership; R²: share of the variation in preference profiles explained by the attribute (PERMANOVA). 999 permutations each.

**Table S1 Sensitivity to the balancing parameters and robustness checks (supplementary)**

| Measure | Value |
|---|---|
| Configurations (ζ, κ ∈ {0, 0.1, …, 1}; μ = 0.5) | 121 |
| Selected number of subgroups | c* = 2 in 104, c* = 3 in 17 |
| Modified silhouette at c* (min–max) | 0.024–0.067 |
| Configurations with p < 0.05, unadjusted | 37 (31%), all with κ ≥ 0.6 |
| Configurations with p < 0.05 after Benjamini–Hochberg (121 tests) | 0 |
| Smallest p, unadjusted / adjusted | 0.003 / 0.116 |
| ARI with the partition at ζ = 0.4, κ = 0.6 (median; min–max) | 0.77; 0.02–1.00 |
| Kendall's τ, consensus vs pooled ranking (min over configurations) | 0.962 |
| Subgroup weights for μ = 0 to 1 (larger subgroup) | 0.571 to 0.535 |
| Graded instead of pairwise encoding: modified silhouette (p) | 0.033 (0.010); ARI with the main partition 0.35 |

*Note.* Each configuration was compared with the same 1,000 permutation nulls as Table 3.1; the graded encoding with 200.

**Table S2 Representation of sociodemographic groups by the pooled and the SCO-FCM consensus rankings (supplementary)**

| Attribute | Group | n | τ, pooled | τ, consensus | τ, random group [95% range] | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| Region | Africa | 103 | 0.657 | 0.667 | 0.758 [0.638, 0.857] | 0.052 | 1.000 |
| Region | Asia | 49 | 0.667 | 0.676 | 0.667 [0.514, 0.800] | 0.508 | 1.000 |
| Region | Australia and New Zealand | 113 | 0.800 | 0.790 | 0.773 [0.648, 0.876] | 0.715 | 1.000 |
| Region | Europe | 267 | 0.867 | 0.857 | 0.861 [0.790, 0.924] | 0.605 | 1.000 |
| Region | Latin America and the Caribbean | 128 | 0.762 | 0.771 | 0.783 [0.676, 0.877] | 0.334 | 1.000 |
| Region | Middle East | 40 | 0.562 | 0.571 | 0.639 [0.467, 0.790] | 0.180 | 1.000 |
| Region | Northern America | 43 | 0.600 | 0.610 | 0.649 [0.476, 0.790] | 0.266 | 1.000 |
| Region | UK | 249 | 0.867 | 0.876 | 0.855 [0.771, 0.924] | 0.645 | 1.000 |
| Region | US | 272 | 0.857 | 0.848 | 0.863 [0.781, 0.933] | 0.471 | 1.000 |
| Age | 18-24 years old | 254 | 0.848 | 0.838 | 0.858 [0.780, 0.924] | 0.425 | 1.000 |
| Age | 25-34 years old | 387 | 0.886 | 0.895 | 0.893 [0.829, 0.952] | 0.470 | 1.000 |
| Age | 35-44 years old | 207 | 0.790 | 0.800 | 0.836 [0.743, 0.914] | 0.171 | 1.000 |
| Age | 45-54 years old | 172 | 0.829 | 0.819 | 0.818 [0.714, 0.905] | 0.609 | 1.000 |
| Age | 55-64 years old | 164 | 0.762 | 0.752 | 0.812 [0.723, 0.895] | 0.164 | 1.000 |
| Age | 65+ years old | 83 | 0.629 | 0.638 | 0.733 [0.590, 0.848] | 0.073 | 1.000 |
| Gender | Female | 613 | 0.914 | 0.905 | 0.929 [0.876, 0.971] | 0.318 | 1.000 |
| Gender | Male | 634 | 0.886 | 0.895 | 0.933 [0.876, 0.971] | 0.055 | 1.000 |
| Gender | Non-binary / third gender | 20 | 0.648 | 0.638 | 0.542 [0.342, 0.724] | 0.873 | 1.000 |
| Religion | Christian | 414 | 0.886 | 0.876 | 0.901 [0.838, 0.952] | 0.358 | 1.000 |
| Religion | Jewish | 33 | 0.638 | 0.648 | 0.618 [0.438, 0.771] | 0.612 | 1.000 |
| Religion | Muslim | 29 | 0.724 | 0.733 | 0.595 [0.419, 0.762] | 0.936 | 1.000 |
| Religion | No Affiliation | 717 | 0.943 | 0.933 | 0.944 [0.895, 0.981] | 0.552 | 1.000 |
| Religion | Other | 23 | 0.581 | 0.590 | 0.568 [0.362, 0.734] | 0.556 | 1.000 |
| Ethnicity | Asian | 81 | 0.657 | 0.667 | 0.725 [0.600, 0.848] | 0.138 | 1.000 |
| Ethnicity | Black | 108 | 0.781 | 0.790 | 0.763 [0.648, 0.876] | 0.624 | 1.000 |
| Ethnicity | Hispanic | 110 | 0.743 | 0.752 | 0.765 [0.648, 0.867] | 0.358 | 1.000 |
| Ethnicity | Mixed | 59 | 0.733 | 0.743 | 0.687 [0.543, 0.810] | 0.762 | 1.000 |
| Ethnicity | Other | 30 | 0.514 | 0.524 | 0.596 [0.410, 0.752] | 0.199 | 1.000 |
| Ethnicity | White | 816 | 0.943 | 0.933 | 0.947 [0.905, 0.981] | 0.498 | 1.000 |

*Note.* Kendall's τ between a group's model ranking and the ranking named in the column; p tests whether the group is represented worse than a random group of the same size (999 permutations).
