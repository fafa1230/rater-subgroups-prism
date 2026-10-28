"""All settings of the study in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# --- Data: PRISM (Kirk et al., 2024), pinned revision ---------------------------
HF_DATASET = "HannahRoseKirk/prism-alignment"
HF_REVISION = "18ab5cfb37456f4ec8cbc00212ce54cf7b1239f6"
FILES = {  # name -> sha256
    "survey.jsonl": "076f2c9492ead60c1268a30c00f049f20a798528d62ca877eee600b12245f513",
    "utterances.jsonl": "7829679350031d6678fa42b8106b2d05051bf9feea348a294400347af4a6deb4",
}
MIN_CONVERSATIONS = 6        # participants who completed the full set of conversations
OPENING_TURN = 0             # the only turn in which several models answer the same prompt
TIE_VALUE = 0.5              # equal scores within an opening turn count as a tie
# P-HFE encoding of the opening-turn ratings:
#   "pairwise": every opening turn is split into the pairwise comparisons it implies and coded
#               win = 1, tie = 0.5, loss = 0, as in the original manuscript   <- main analysis
#   "relative": graded score (s - min) / (max - min) within the opening turn  <- robustness
PHFE_ENCODING = "pairwise"
STATED_PREFS = ("values", "creativity", "fluency", "factuality", "diversity", "safety",
                "personalisation", "helpfulness")

# --- Algorithm 1 (adapted SCO-FCM), parameters as in the original manuscript ----
ZETA, KAPPA, MU = 0.4, 0.6, 0.5
C_CANDIDATES = tuple(range(2, 9))
FUZZIFIER = 2.0
FCM_TOL = 1e-5
T_MAX = 300
PAGERANK_DAMPING = 0.85

# --- Validation ------------------------------------------------------------------
SEED = 2026
N_NULL = 1000                # permutation null of the main validity indices (same draws as the sensitivity grid)
N_NULL_BASELINES = 100       # null draws per baseline method
N_STABILITY = 200            # 80% participant subsamples
N_SPLIT = 50                 # split-half replications (real and null)
N_NULL_SENSITIVITY = 1000    # null instances evaluated on the whole zeta x kappa grid
N_PERM_EXTERNAL = 999        # permutations for the external-validation tests
MIN_GROUP = 20               # categories smaller than this are left out of external tests
ALPHA = 0.05
