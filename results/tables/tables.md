# Tables of the revised manuscript

**Table 2.1 Data used in this study (PRISM opening turns)**

| Item | Value |
|---|---|
| Participants (six or more conversations) | 1,267 |
| Language models | 21 |
| Conversations | 7,607 |
| Opening-turn ratings (1–100) | 28,602 |
| Pairwise outcomes derived from opening turns | 80,618 |
| Opening turns with 4 / 3 / 2 models | 5,926 / 1,536 / 145 |
| Share of tied outcomes | 5.7% |
| Density of the participant × model profile | 68.7% |
| Models rated per participant (mean) | 14.4 |
| Outcomes per participant–model cell (median) | 3 |
| Participants per model (min–max) | 521–936 |
| Conversation types (unguided / values / controversy) | 2,925 / 2,350 / 2,332 |


**Table 2.2 SCO-FCM components and their counterparts in this study**

| Component | Baseline [5] | This study |
|---|---|---|
| Decision maker (DM) | Expert or stakeholder | PRISM participant who rated model responses |
| Assessment | Probabilistic hesitant fuzzy element (P-HFE) | Win (1), tie (0.5), and loss (0) outcomes of a model in the participant's opening turns, with their relative frequencies |
| Score of a P-HFE | Score function with a fluctuation penalty | Unchanged |
| Background relation, weighted by ζ | Historical background correlation | Similarity of the stated importance of eight response attributes (survey) |
| Current relation | Current research correlation | Jaccard similarity of the sets of models a participant chose as best |
| κ | Weight of distance vs reachability in clustering and in the modified silhouette | Unchanged (0.6) |
| μ | Weight of cohesion vs size in subgroup weights | Unchanged (0.5; 0–1 examined) |
| Number of subgroups | c ∈ {2, …, 8}, maximum modified silhouette | Unchanged |


**Table 3.1 Cluster validity of the selected partition (c* = 2) against 1,000 permutation nulls**

| Index | Observed | Null mean [95% range] | p |
|---|---|---|---|
| Modified silhouette at c* (selection criterion) | 0.042 | 0.037 [0.031, 0.042] | 0.042 |
| Modified silhouette, best over c = 2–8 (selection-aware) | 0.042 | 0.037 [0.031, 0.043] | 0.045 |
| Silhouette, Euclidean | 0.027 | 0.021 [0.017, 0.025] | 0.003 |
| Silhouette, current relation | 0.064 | 0.060 [0.050, 0.070] | 0.254 |
| Normalized fuzzy partition coefficient | 0.0022 | 0.0015 [0.0011, 0.0019] | < 0.001 |
| Xie–Beni index (lower is better) | 405.6 | 743.7 [527.9, 1053.7] | < 0.001 |
| Davies–Bouldin index (lower is better) | 6.126 | 6.693 [6.118, 7.274] | 0.031 |
| Calinski–Harabasz index | 32.1 | 27.1 [22.7, 32.6] | 0.035 |

*Note.* One-sided permutation p-values. The selection-aware row compares the best value over c = 2–8 with the best value attained on each null dataset, as in Table 3.5. The null selected c* = 2 in 86% of replicates.

**Table 3.2 Stability and replication of the subgroups**

| Measure | Observed | Permutation null |
|---|---|---|
| Stability: ARI with the full-data partition, 80% subsamples (median) | 0.777 | – |
| Stability: Jaccard similarity of the smaller subgroup (median) | 0.873 | – |
| Replication: ARI between the partitions of halves A and B | −0.0001 | 0.0003 |
| Largest subgroup difference, in-sample (half A) | 0.506 [0.371, 0.563] | 0.464 [0.342, 0.553] |
| Largest subgroup difference, held-out (half B) | 0.066 [0.042, 0.094] | 0.065 [0.039, 0.089] |
| Largest effect size ε², in-sample (half A) | 0.375 [0.196, 0.491] | 0.323 [0.187, 0.450] |
| Largest effect size ε², held-out (half B) | 0.010 [0.005, 0.018] | 0.009 [0.004, 0.018] |
| Models (of 21) with a significant difference, in-sample | 12.16 | 11.32 |
| Models (of 21) with a significant difference, held-out | 0.06 | 0.06 |
| Correlation of subgroup preference profiles, A vs B | 0.129 | 0.021 |
| Split-half reliability of individual preferences (r per model) | 0.037 | 0.000 [−0.022, 0.027] |

*Note.* Stability: 200 subsamples of 80% of the participants. Replication: means over 50 random splits of each participant's conversations into halves A and B, with the 2.5th and 97.5th percentiles in brackets; subgroups formed on half A; significance by Kruskal–Wallis tests (equivalent to Mann–Whitney tests for two subgroups) with Benjamini–Hochberg adjustment over 21 models; effect size ε² = H/(n − 1). Reliability: null mean [95% range].

**Table 3.3 Algorithm 1 and other clustering methods, each against its own permutation null**

| Method | Sizes | Silhouette (Euclidean) | Null mean | p | ARI with Algorithm 1 |
|---|---|---|---|---|---|
| Algorithm 1 (SCO-FCM) | 723/544 | 0.027 | 0.021 | 0.020 | – |
| Ablation: no reachability (κ = 1) | 683/584 | 0.051 | 0.046 | 0.020 | 0.227 |
| Standard FCM (random init) | 676/591 | 0.051 | 0.045 | 0.020 | 0.217 |
| K-Means | 954/313 | 0.058 | 0.060 | 0.584 | 0.113 |
| GMM (diagonal) | 958/309 | 0.057 | 0.057 | 0.535 | 0.111 |
| Agglomerative (Ward) | 1152/115 | 0.032 | 0.041 | 0.832 | −0.001 |
| Spectral (behavioural graph) | 686/581 | 0.022 | 0.020 | 0.099 | 0.499 |
| Spectral (kNN on profiles) | 935/332 | 0.055 | 0.055 | 0.574 | −0.001 |
| HDBSCAN (data-driven c) | no clusters (all noise) | – | – | – | 0.000 |
| Louvain (behavioural graph, data-driven c) | 448/370/227/222 | 0.014 | 0.014 | 0.594 | 0.006 |

*Note.* Silhouettes are computed on the same P-HFE profiles; 100 null replicates per method. All methods except HDBSCAN and Louvain use c = 2. Modularity of the Louvain communities on the current relation: 0.096 (null mean 0.097, p = 0.733).

**Table 3.4 Sociodemographic validation of the subgroups**

| Attribute | n | Groups | Cramér's V | p | p (Holm) | R² of profiles | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Region | 1,264 | 9 | 0.063 | 0.753 | 1.000 | 0.008 | 0.059 | 0.649 |
| Age | 1,267 | 6 | 0.087 | 0.075 | 0.750 | 0.005 | 0.143 | 1.000 |
| Gender | 1,267 | 3 | 0.033 | 0.505 | 1.000 | 0.002 | 0.081 | 0.810 |
| Education | 1,225 | 5 | 0.050 | 0.524 | 1.000 | 0.003 | 0.436 | 1.000 |
| Employment | 1,251 | 7 | 0.070 | 0.407 | 1.000 | 0.005 | 0.639 | 1.000 |
| Marital status | 1,233 | 3 | 0.096 | 0.005 | 0.055 | 0.002 | 0.292 | 1.000 |
| English proficiency | 1,262 | 4 | 0.027 | 0.812 | 1.000 | 0.002 | 0.763 | 1.000 |
| Religion | 1,216 | 5 | 0.031 | 0.894 | 1.000 | 0.003 | 0.696 | 1.000 |
| Ethnicity | 1,204 | 6 | 0.072 | 0.278 | 1.000 | 0.004 | 0.501 | 1.000 |
| Familiarity with LLMs | 1,267 | 3 | 0.063 | 0.094 | 0.756 | 0.002 | 0.322 | 1.000 |
| Frequency of LLM use | 1,063 | 5 | 0.086 | 0.084 | 0.756 | 0.004 | 0.446 | 1.000 |

*Note.* Attributes were not used for clustering. Cramér's V: association with subgroup membership; R²: share of the variation in preference profiles explained by the attribute (PERMANOVA). 999 permutations each.

**Table 3.5 Sensitivity to the balancing parameters and robustness checks**

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

**Table A1 Representation of sociodemographic groups by the pooled and the SCO-FCM consensus rankings**

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
