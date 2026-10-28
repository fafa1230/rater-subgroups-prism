# Are rater subgroups discovered or constructed?

Code for the revised manuscript

> I. Lazulfa and S. Patmanthara, "SCO-FCM adaptation for collective LLM evaluation:
> An ontological, epistemological, and axiological study of consensus reality," revised manuscript
> (working title), 2026.

The study applies the adapted SCO-FCM clustering of the manuscript (Algorithm 1) to the opening-turn
ratings of the PRISM dataset. It then asks whether the participant subgroups it finds exist beyond a
permutation null, replicate across halves of the data, agree with other clustering methods, and
correspond to participants' sociodemographic attributes, which are never used for clustering.

## Data

PRISM (Kirk et al., 2024): `HannahRoseKirk/prism-alignment` on Hugging Face, pinned to revision
`18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6`, DOI [10.57967/hf/2113](https://doi.org/10.57967/hf/2113).
Only `survey.jsonl` and `utterances.jsonl` are used. Human-written texts are licensed CC-BY-4.0 and
model responses CC-BY-NC-4.0; the analysis uses ratings, model names, participant IDs, and survey
answers only. The raw files are not redistributed here; `01_download.py` fetches and verifies them.

> H. R. Kirk et al., "The PRISM alignment dataset: What participatory, representative and
> individualised human feedback reveals about the subjective and multicultural alignment of large
> language models," arXiv:2404.16019, 2024.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

| Step | Command | What it does |
|---|---|---|
| 1 | `python -I 01_download.py` | Downloads the two PRISM files at the pinned revision and checks their SHA-256 |
| 2 | `python -I 02_build.py` | Participant x model P-HFE profiles, background and behavioural relations, external variables; `results/data_summary.json` |
| 3 | `python -I 03_analyze.py main` | Algorithm 1 at zeta = 0.4, kappa = 0.6, mu = 0.5; consensus vs flat ranking |
| 4 | `python -I 03_analyze.py validity` | Validity indices for c = 2..8 against 1,000 permutation nulls |
| 5 | `python -I 03_analyze.py stability` | 200 subsamples of 80% of the participants |
| 6 | `python -I 03_analyze.py split` | Split-half replication and held-out tests, real and null |
| 6b | `python -I 03_analyze.py reliability` | Split-half consistency of individual preferences, without clustering |
| 7 | `python -I 03_analyze.py baselines` | Nine other clustering methods, including Louvain community detection, each against its own null |
| 7b | `python -I 03_analyze.py exposure` | Whether the models a participant met depend on the participant (χ² against 1,000 reassignments) |
| 8 | `python -I 03_analyze.py external` | Association with sociodemographics; representation of groups by the rankings |
| 9 | `python -I 03_analyze.py relative` | Robustness: graded instead of pairwise P-HFE encoding |
| 10 | `python -I 03_analyze.py membership` | Largest fuzzy membership of every participant, real and null |
| 11 | `python -I 03_analyze.py sensitivity` | ζ × κ grid (11 × 11), each cell against the same 1,000 nulls as step 4 |
| 12 | `python -I 04_figures.py` | Figures 2.1 and 3.1–3.5 at printed width (16 cm), PNG 600 dpi and PDF, in `figures/` |
| 13 | `python -I 05_tables.py` | Tables 2.1–2.2 and 3.1–3.4 of the article and supplementary Tables S1–S2 as Markdown, HTML (for Word), and CSV in `results/tables/` |

All settings are in `config.py`; random seeds are fixed.

## Method in brief

- **Preferences.** In the opening turn of a PRISM conversation, up to four models answer the same
  prompt and the participant scores each from 1 to 100. Every opening turn is split into the pairwise
  comparisons it implies (win 1, tie 0.5, loss 0). The outcomes of a participant on a model form a
  probabilistic hesitant fuzzy element, scored with the function of Yang et al. (2026).
- **Relations.** The background relation is the similarity of the stated importance of eight response
  attributes (survey). The behavioural relation is the Jaccard similarity of the sets of models a
  participant chose as best. Both enter PageRank-based priorities, as in Algorithm 1.
- **Null model.** Conversations are dealt out again at random among participants within each
  conversation type. This keeps model quality, which models met, and every within-conversation
  judgement, and it removes only the link between a participant and their own judgements.

## License

MIT (code). See the PRISM terms above for the data.
