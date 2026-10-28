"""Build the participant x model P-HFE profiles, the two participant relations, and the
external (sociodemographic) variables, and print the data summary.

    python -I 02_build.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.data import (background_of, build_instance, load_opening_turns, load_survey,  # noqa: E402
                      pairwise_outcomes, select, verify_raw)

EXTERNAL = {  # variable -> how to read it from the survey
    "region": lambda s: pd.json_normalize(s["location"].tolist())["special_region"].to_numpy(),
    "age": lambda s: s["age"].to_numpy(),
    "gender": lambda s: s["gender"].to_numpy(),
    "education": lambda s: s["education"].to_numpy(),
    "employment": lambda s: s["employment_status"].to_numpy(),
    "marital": lambda s: s["marital_status"].to_numpy(),
    "english": lambda s: s["english_proficiency"].to_numpy(),
    "religion": lambda s: pd.json_normalize(s["religion"].tolist())["simplified"].to_numpy(),
    "ethnicity": lambda s: pd.json_normalize(s["ethnicity"].tolist())["simplified"].to_numpy(),
    "lm_familiarity": lambda s: s["lm_familiarity"].to_numpy(),
    "lm_use": lambda s: s["lm_frequency_use"].to_numpy(),
}


def external_table(survey, users):
    s = survey.set_index("user_id").loc[users].reset_index()
    tab = pd.DataFrame({"user_id": users})
    for name, get in EXTERNAL.items():
        v = pd.Series(get(s), dtype=object)
        v = v.where(~v.isin(["Prefer not to say"]) & v.notna())
        small = v.value_counts()
        v = v.where(~v.isin(small[small < config.MIN_GROUP].index))
        tab[name] = v.to_numpy()
    tab["us_rep"] = s["included_in_US_REP"].to_numpy()
    tab["uk_rep"] = s["included_in_UK_REP"].to_numpy()
    return tab


def main():
    verify_raw()
    survey = load_survey()
    op, users, models = select(load_opening_turns())
    inst = build_instance(op, users, models, background_of(survey, users))
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.PROCESSED / "instance.npz", users=np.array(users), models=np.array(models),
                        **{k: v for k, v in inst.items()})
    op.to_csv(config.PROCESSED / "opening_turns.csv", index=False)
    ext = external_table(survey, users)
    ext.to_csv(config.PROCESSED / "external.csv", index=False)

    out = pairwise_outcomes(op)
    cells = out.groupby(["user_id", "model"]).size()
    per_conv = op.groupby("conversation_id")
    stats = {
        "participants": len(users), "models": len(models), "conversations": int(op["conversation_id"].nunique()),
        "opening_ratings": int(len(op)), "pairwise_outcomes": int(len(out)),
        "models_per_opening_turn": per_conv["model_name"].nunique().value_counts().sort_index().to_dict(),
        "opening_turns_with_equal_scores": float(per_conv["score"].apply(lambda x: x.duplicated().any()).mean()),
        "tie_share_of_outcomes": float((out["gamma"] == config.TIE_VALUE).mean()),
        "density": float(inst["observed"].mean()),
        "models_rated_per_participant_mean": float(inst["observed"].sum(1).mean()),
        "outcomes_per_cell_median": float(cells.median()), "share_cells_single_outcome": float((cells == 1).mean()),
        "participants_per_model_min_max": [int(inst["observed"].sum(0).min()), int(inst["observed"].sum(0).max())],
        "conversation_types": op.drop_duplicates("conversation_id")["conversation_type"].value_counts().to_dict(),
        "external_n": {c: int(ext[c].notna().sum()) for c in EXTERNAL},
        "us_rep": int(ext["us_rep"].sum()), "uk_rep": int(ext["uk_rep"].sum()),
    }
    (config.RESULTS).mkdir(exist_ok=True)
    (config.RESULTS / "data_summary.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
