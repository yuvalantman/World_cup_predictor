"""Live tournament page for WC 2026 — real results, standings, simulate forward."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.state.live_state import (
    apply_results_from_csv,
    initialize_live_state,
    record_match_result,
    simulate_forward,
)
from src.tournament.group_standings import build_group_standings, get_group_position_map
UPDATES_CSV            = Path("data/raw/world_cup_updates/all_world_cup_2026_updates.csv")
CALIBRATION_CSV        = Path("data/raw/world_cup_updates/calibration_predictions.csv")
MODEL_ACCURACY_CSV_V4  = Path("data/raw/world_cup_updates/model_accuracy_v4.csv")
MODEL_ACCURACY_CSV_V5  = Path("data/raw/world_cup_updates/model_accuracy_v5.csv")
MODEL_ACCURACY_CSV_V6  = Path("data/raw/world_cup_updates/model_accuracy_v6.csv")
KO_RESULTS_CSV         = Path("data/raw/world_cup_updates/knockout_results.csv")
_MODEL_ACCURACY_COLS   = ["match_id", "team_a", "team_b", "pred_goals_a", "pred_goals_b", "actual_a", "actual_b", "pred_la", "pred_lb"]
_MODELS_DIR            = Path(__file__).resolve().parents[2] / "models"


def _load_ko_results() -> dict:
    """Load knockout match results from CSV → {slot: {...}}."""
    if not KO_RESULTS_CSV.exists():
        return {}
    try:
        df = pd.read_csv(KO_RESULTS_CSV)
        out: dict = {}
        for _, row in df.iterrows():
            slot = str(row["slot"])

            def _f(col):
                v = row.get(col)
                return None if v is None or (isinstance(v, float) and pd.isna(v)) else v

            out[slot] = {
                "team_a":  str(_f("team_a") or ""),
                "team_b":  str(_f("team_b") or ""),
                "goals_a": int(row["goals_a"]),
                "goals_b": int(row["goals_b"]),
                "winner":  None if _f("winner") is None else str(_f("winner")),
                "pred_a":  None if _f("pred_a") is None else int(_f("pred_a")),
                "pred_b":  None if _f("pred_b") is None else int(_f("pred_b")),
                "pred_la": None if _f("pred_la") is None else float(_f("pred_la")),
                "pred_lb": None if _f("pred_lb") is None else float(_f("pred_lb")),
                "date":    str(_f("date") or ""),
            }
        return out
    except Exception:
        return {}


def _update_ko_winner(slot: str, winner: str) -> None:
    """Patch only the winner field for an existing slot in CSV + session state."""
    if KO_RESULTS_CSV.exists():
        df = pd.read_csv(KO_RESULTS_CSV)
        if "winner" not in df.columns:
            df["winner"] = None
        df.loc[df["slot"] == slot, "winner"] = winner
        df.to_csv(KO_RESULTS_CSV, index=False)

    ko = dict(st.session_state.get("ko_results", {}))
    if slot in ko:
        ko[slot] = {**ko[slot], "winner": winner}
    st.session_state.ko_results = ko


def _append_ko_to_accuracy_csvs(
    slot: str, team_a: str, team_b: str,
    actual_a: int, actual_b: int,
    pred_la: float | None, pred_lb: float | None,
) -> None:
    """Append one KO row to each model accuracy CSV using stored lambdas. No model loading."""
    if pred_la is None or pred_lb is None:
        return
    mid = _KO_SLOT_MATCH_ID.get(slot, 19999)
    for name, csv, score_fn in _ko_score_fns():
        if not csv.exists():
            continue
        try:
            # Skip if this slot already recorded
            existing = pd.read_csv(csv, usecols=["match_id"])
            if mid in existing["match_id"].astype(int).values:
                continue
            pa, pb = score_fn(float(pred_la), float(pred_lb))
            new_row = pd.DataFrame([{
                "match_id": mid, "team_a": team_a, "team_b": team_b,
                "pred_goals_a": int(pa), "pred_goals_b": int(pb),
                "actual_a": actual_a, "actual_b": actual_b,
                "pred_la": float(pred_la), "pred_lb": float(pred_lb),
            }])
            full = pd.read_csv(csv)
            pd.concat([full, new_row], ignore_index=True).to_csv(csv, index=False)
        except Exception:
            pass


def _save_ko_result(
    slot: str,
    team_a: str,
    team_b: str,
    date,
    goals_a: int,
    goals_b: int,
    winner: str | None = None,
    pred_a=None,
    pred_b=None,
    pred_la=None,
    pred_lb=None,
) -> None:
    """Persist a knockout match result (with optional penalty winner) to CSV and session state."""
    KO_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "slot":    slot,
        "team_a":  team_a,
        "team_b":  team_b,
        "date":    str(date)[:10] if date is not None else "",
        "goals_a": int(goals_a),
        "goals_b": int(goals_b),
        "winner":  winner,
        "pred_a":  pred_a,
        "pred_b":  pred_b,
        "pred_la": pred_la,
        "pred_lb": pred_lb,
    }

    if KO_RESULTS_CSV.exists():
        df = pd.read_csv(KO_RESULTS_CSV)
        if "winner" not in df.columns:
            df["winner"] = None
    else:
        df = pd.DataFrame(columns=list(row.keys()))

    df = df[df["slot"] != slot].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(KO_RESULTS_CSV, index=False)

    # Update accuracy CSVs immediately — no model loading needed
    _append_ko_to_accuracy_csvs(slot, team_a, team_b, int(goals_a), int(goals_b), pred_la, pred_lb)

    ko = dict(st.session_state.get("ko_results", {}))
    ko[slot] = {
        "team_a":  team_a,
        "team_b":  team_b,
        "goals_a": int(goals_a),
        "goals_b": int(goals_b),
        "winner":  winner,
        "pred_a":  pred_a,
        "pred_b":  pred_b,
        "pred_la": pred_la,
        "pred_lb": pred_lb,
        "date":    str(date)[:10] if date is not None else "",
    }
    st.session_state.ko_results = ko

    # Apply to live state so ELO / form / Team Inspector update immediately
    if "true_state" in st.session_state:
        st.session_state.true_state = _apply_ko_results_to_state(
            st.session_state.true_state, {slot: ko[slot]}
        )


def _remove_ko_result(slot: str) -> None:
    """Remove a KO result from CSV, session state, accuracy CSVs, and rebuild live state."""
    # 1. Remove from knockout_results.csv
    if KO_RESULTS_CSV.exists():
        df = pd.read_csv(KO_RESULTS_CSV)
        df = df[df["slot"] != slot].copy()
        df.to_csv(KO_RESULTS_CSV, index=False)

    # 2. Remove from session_state.ko_results
    ko = dict(st.session_state.get("ko_results", {}))
    ko.pop(slot, None)
    st.session_state.ko_results = ko

    # 3. Remove from accuracy CSVs by synthetic match_id
    mid = _KO_SLOT_MATCH_ID.get(slot)
    if mid is not None:
        for _, csv, _ in _ko_score_fns():
            if csv.exists():
                try:
                    acc = pd.read_csv(csv)
                    acc = acc[acc["match_id"].astype(int) != mid]
                    acc.to_csv(csv, index=False)
                except Exception:
                    pass

    # 4. Rebuild true_state from scratch so ELO is correct without this game
    base_hist = st.session_state.get("_base_historical")
    base_fix  = st.session_state.get("_base_fixtures")
    if base_hist is not None and base_fix is not None:
        from src.state.tournament_calibration import initialize_calibration, load_calibration_from_csv, load_prior
        state = initialize_live_state(base_hist, base_fix)
        state = apply_results_from_csv(state, UPDATES_CSV)
        if ko:
            state = _apply_ko_results_to_state(state, ko)
        # Preserve calibration from existing state
        old_state = st.session_state.get("true_state", {})
        state["calibration"]       = old_state.get("calibration", {})
        state["match_predictions"] = old_state.get("match_predictions", {})
        st.session_state.true_state = state
    else:
        # Fallback: force full reinit on next run
        st.session_state.pop("true_state", None)


def persist_real_result_to_csv(fixture, goals_a: int, goals_b: int) -> None:
    """Append/update a real World Cup match result into the updates CSV."""
    UPDATES_CSV.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "date": fixture["date"],
        "team_a": fixture["team_a"],
        "team_b": fixture["team_b"],
        "goals_a": int(goals_a),
        "goals_b": int(goals_b),
        "competition": "FIFA World Cup",
        "location": fixture.get("location", "neutral"),
    }

    if UPDATES_CSV.exists():
        df = pd.read_csv(UPDATES_CSV)
    else:
        df = pd.DataFrame(columns=row.keys())

    # remove previous result for the same fixture if exists
    same_match = (
        (df["team_a"] == row["team_a"])
        & (df["team_b"] == row["team_b"])
        & (df["date"].astype(str) == str(row["date"]))
    )

    df = df[~same_match].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(UPDATES_CSV, index=False)

def clear_saved_results_csv() -> None:
    """Clear all saved real World Cup results, calibration data, and model accuracy data."""
    from src.state.tournament_calibration import clear_calibration_csv

    UPDATES_CSV.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        columns=["date", "team_a", "team_b", "goals_a", "goals_b", "competition", "location"]
    ).to_csv(UPDATES_CSV, index=False)

    clear_calibration_csv(CALIBRATION_CSV)

    for acc_csv in (MODEL_ACCURACY_CSV_V4, MODEL_ACCURACY_CSV_V5, MODEL_ACCURACY_CSV_V6):
        pd.DataFrame(columns=_MODEL_ACCURACY_COLS).to_csv(acc_csv, index=False)

    # Clear knockout results
    pd.DataFrame(
        columns=["slot","team_a","team_b","date","goals_a","goals_b","pred_a","pred_b","pred_la","pred_lb"]
    ).to_csv(KO_RESULTS_CSV, index=False)
    import streamlit as _st
    _st.session_state.pop("ko_results", None)

ROUND_LABELS = {
    "GROUPS": "Group Stage",
    "R32": "Round of 32",
    "R16": "Round of 16",
    "QF": "Quarter Finals",
    "SF": "Semi Finals",
    "FINAL_STAGE": "Final Stage",
    "DONE": "Tournament Finished",
}

_FLAGS: dict[str, str] = {
    "Argentina": "🇦🇷", "Brazil": "🇧🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Spain": "🇪🇸", "Germany": "🇩🇪", "Portugal": "🇵🇹", "Netherlands": "🇳🇱",
    "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾", "Mexico": "🇲🇽",
    "United States": "🇺🇸", "USA": "🇺🇸", "Canada": "🇨🇦", "Japan": "🇯🇵",
    "South Korea": "🇰🇷", "Korea Republic": "🇰🇷", "Morocco": "🇲🇦",
    "Senegal": "🇸🇳", "Switzerland": "🇨🇭", "Colombia": "🇨🇴",
    "Ecuador": "🇪🇨", "Australia": "🇦🇺", "Iran": "🇮🇷", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Ghana": "🇬🇭", "Tunisia": "🇹🇳", "Egypt": "🇪🇬",
    "Turkey": "🇹🇷", "Norway": "🇳🇴", "Sweden": "🇸🇪",
    "Czechia": "🇨🇿", "Austria": "🇦🇹", "Algeria": "🇩🇿",
    "Ivory Coast": "🇨🇮", "New Zealand": "🇳🇿", "Panama": "🇵🇦",
    "Paraguay": "🇵🇾", "South Africa": "🇿🇦", "Cape Verde": "🇨🇻",
    "Haiti": "🇭🇹", "Jordan": "🇯🇴", "Iraq": "🇮🇶", "Uzbekistan": "🇺🇿",
    "DR Congo": "🇨🇩", "Bosnia and Herzegovina": "🇧🇦", "Curaçao": "🇨🇼",
    "Venezuela": "🇻🇪", "Chile": "🇨🇱", "Peru": "🇵🇪", "Bolivia": "🇧🇴",
    "Costa Rica": "🇨🇷", "Honduras": "🇭🇳", "Jamaica": "🇯🇲",
    "Nigeria": "🇳🇬", "Cameroon": "🇨🇲", "Ghana": "🇬🇭", "Mali": "🇲🇱",
    "Serbia": "🇷🇸", "Poland": "🇵🇱", "Ukraine": "🇺🇦", "Romania": "🇷🇴",
    "Hungary": "🇭🇺", "Slovakia": "🇸🇰", "Greece": "🇬🇷", "Denmark": "🇩🇰",
    "Finland": "🇫🇮", "Iceland": "🇮🇸", "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "New Zealand": "🇳🇿", "Indonesia": "🇮🇩", "Thailand": "🇹🇭",
}


def _flag(team: str) -> str:
    return _FLAGS.get(team, "⚽")


def _team_label(team: str) -> str:
    return f"{_flag(team)} {team}"

def _format_rank_change(value) -> str:
    if pd.isna(value):
        return "–"
    value = int(value)
    if value > 0:
        return f"▲ {value}"
    if value < 0:
        return f"▼ {abs(value)}"
    return "–"


def _format_elo_change(value) -> str:
    if pd.isna(value):
        return "–"
    value = float(value)
    if value > 0:
        return f"+{value:.1f} ▲"
    if value < 0:
        return f"{value:.1f} ▼"
    return "0.0"


def _style_rankings_table(df: pd.DataFrame):
    def color_rank_change(value):
        value = str(value)
        if "▲" in value:
            return "color: #16a34a; font-weight: 700"
        if "▼" in value:
            return "color: #dc2626; font-weight: 700"
        return "color: #6b7280"

    def color_elo_change(value):
        value = str(value)
        if value.startswith("+"):
            return "color: #16a34a; font-weight: 700"
        if "▼" in value or value.startswith("-"):
            return "color: #dc2626; font-weight: 700"
        return "color: #6b7280"

    return (
        df.style
        .map(color_rank_change, subset=["Rank Change"])
        .map(color_elo_change, subset=["ELO Change"])
    )

def _world_cup_live_rankings(state: dict) -> pd.DataFrame:
    """Live World Cup teams ranking vs global Elo table, compared to tournament start."""
    from src.features.team_names import normalize_team_name

    fixtures = state["fixtures"]

    wc_teams = sorted({
        normalize_team_name(team)
        for team in pd.concat([fixtures["team_a"], fixtures["team_b"]]).dropna()
    })

    hist = state["historical_matches"].copy()

    baseline_hist = hist.copy()

    if "source_file" in baseline_hist.columns:
        baseline_hist = baseline_hist[baseline_hist["source_file"] != "live_2026"]

    if "tournament_key" in baseline_hist.columns:
        baseline_hist = baseline_hist[
            baseline_hist["tournament_key"] != "FIFA World Cup_2026"
        ]

    start_points = {}

    for _, row in baseline_hist.sort_values("date").iterrows():
        a = normalize_team_name(row["team_a"])
        b = normalize_team_name(row["team_b"])

        start_points[a] = float(row["rating_a"])
        start_points[b] = float(row["rating_b"])

    current_points = start_points.copy()

    for team, points in state["elo_ratings"].items():
        team_norm = normalize_team_name(team)
        current_points[team_norm] = float(points)

    # Critical fix:
    # Override current points for World Cup teams from the latest historical state,
    # because record_match_result appends canonical live_2026 rows there.
    for _, row in hist.sort_values("date").iterrows():
        a = normalize_team_name(row["team_a"])
        b = normalize_team_name(row["team_b"])

        if a in wc_teams:
            current_points[a] = float(row["rating_a"])
        if b in wc_teams:
            current_points[b] = float(row["rating_b"])

    global_df = pd.DataFrame(
        [{"team": t, "current_points": p} for t, p in current_points.items()]
    )

    start_df = pd.DataFrame(
        [{"team": t, "start_points": p} for t, p in start_points.items()]
    )

    if global_df.empty:
        return global_df

    global_df = (
        global_df
        .sort_values(["current_points", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )
    global_df["current_global_rank"] = global_df.index + 1

    start_df = (
        start_df
        .sort_values(["start_points", "team"], ascending=[False, True])
        .reset_index(drop=True)
    )
    start_df["start_global_rank"] = start_df.index + 1

    result = global_df.merge(
        start_df[["team", "start_points", "start_global_rank"]],
        on="team",
        how="left",
    )

    result = result[result["team"].isin(wc_teams)].copy()

    result["elo_change"] = result["current_points"] - result["start_points"]
    result["rank_change"] = (
        result["start_global_rank"] - result["current_global_rank"]
    )

    return result[
        [
            "current_global_rank",
            "rank_change",
            "team",
            "current_points",
            "elo_change",
            "start_global_rank",
            "start_points",
        ]
    ].sort_values("current_global_rank")



# ---------------------------------------------------------------------------
# Model accuracy tracking
# ---------------------------------------------------------------------------

def _load_all_model_configs() -> list[dict]:
    """Return configs for every model version that has a joblib file on disk."""
    import joblib
    import json
    from functools import partial
    from src.features.build_features import build_pre_match_features, build_pre_match_features_v5
    from src.models.score_conversion import most_likely_score, most_likely_score_v5, most_likely_score_v6

    configs = []

    v4_path = _MODELS_DIR / "production_model_v4.joblib"
    if v4_path.exists():
        configs.append({
            "name": "v4",
            "model": joblib.load(v4_path),
            "feature_fn": build_pre_match_features,
            "score_fn": most_likely_score,
            "csv": MODEL_ACCURACY_CSV_V4,
        })

    v5_path = _MODELS_DIR / "production_model_v5.joblib"
    if v5_path.exists():
        configs.append({
            "name": "v5",
            "model": joblib.load(v5_path),
            "feature_fn": build_pre_match_features_v5,
            "score_fn": most_likely_score_v5,
            "csv": MODEL_ACCURACY_CSV_V5,
        })

    v6_path = _MODELS_DIR / "production_model_v6.joblib"
    if v6_path.exists():
        score_fn_v6 = most_likely_score_v6
        config_path = _MODELS_DIR / "production_config_v6.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as _f:
                    _db = json.load(_f).get("drawband", {})
                score_fn_v6 = partial(
                    most_likely_score_v6,
                    draw_threshold=_db.get("draw_threshold", 0.33),
                    threshold_b=_db.get("threshold_b", 0.5),
                    scale_c=_db.get("scale_c", 0.9992),
                    rho=_db.get("rho", -0.3294),
                )
            except Exception:
                pass
        configs.append({
            "name": "v6",
            "model": joblib.load(v6_path),
            "feature_fn": build_pre_match_features_v5,
            "score_fn": score_fn_v6,
            "csv": MODEL_ACCURACY_CSV_V6,
        })

    return configs


_KO_SLOT_MATCH_ID: dict[str, int] = {
    **{f"R32_{i:02d}": 10000 + i for i in range(1, 17)},
    **{f"R16_{i:02d}": 10100 + i for i in range(1, 9)},
    **{f"QF_{i:02d}": 10200 + i for i in range(1, 5)},
    "SF_01": 10301, "SF_02": 10302,
    "THIRD_PLACE": 10401, "FINAL": 10402,
}


def _ko_score_fns() -> list[tuple[str, Path, object]]:
    """Return (name, csv_path, score_fn) for each model — no joblib loading."""
    import json
    from functools import partial
    from src.models.score_conversion import most_likely_score, most_likely_score_v5, most_likely_score_v6

    configs = []
    if (_MODELS_DIR / "production_model_v4.joblib").exists():
        configs.append(("v4", MODEL_ACCURACY_CSV_V4, most_likely_score))
    if (_MODELS_DIR / "production_model_v5.joblib").exists():
        configs.append(("v5", MODEL_ACCURACY_CSV_V5, most_likely_score_v5))
    if (_MODELS_DIR / "production_model_v6.joblib").exists():
        sfn = most_likely_score_v6
        cp = _MODELS_DIR / "production_config_v6.json"
        if cp.exists():
            try:
                with open(cp, encoding="utf-8") as _f:
                    _db = json.load(_f).get("drawband", {})
                sfn = partial(most_likely_score_v6,
                              draw_threshold=_db.get("draw_threshold", 0.33),
                              threshold_b=_db.get("threshold_b", 0.5),
                              scale_c=_db.get("scale_c", 0.9992),
                              rho=_db.get("rho", -0.3294))
            except Exception:
                pass
        configs.append(("v6", MODEL_ACCURACY_CSV_V6, sfn))
    return configs


def _backfill_model_accuracy(
    base_historical: pd.DataFrame,
    fixtures: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
) -> None:
    """
    Keeps per-model accuracy CSVs in sync with completed matches.

    Two paths:
    - FAST (no model loading): group-stage rows already present, only new KO
      rows need appending. Uses stored pred_la/pred_lb + score_fn (pure fn).
    - SLOW (model loading): group-stage rows missing — full replay needed.
      Only happens on first start or after CSV deletion.
    """
    from src.features.team_names import normalize_team_name

    def _row_count(csv_path: Path) -> int:
        if not csv_path.exists():
            return 0
        try:
            return len(pd.read_csv(csv_path, usecols=["match_id"]))
        except Exception:
            return 0

    def _has_pred_la(csv_path: Path) -> bool:
        if not csv_path.exists():
            return False
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
            return "pred_la" in cols
        except Exception:
            return False

    # ── Cheap row counts ─────────────────────────────────────────────────────
    n_gs = 0
    if UPDATES_CSV.exists():
        try:
            n_gs = int(pd.read_csv(UPDATES_CSV, usecols=["goals_a", "goals_b"]).dropna().shape[0])
        except Exception:
            pass

    ko_rows = pd.DataFrame()
    if KO_RESULTS_CSV.exists():
        try:
            ko_rows = pd.read_csv(KO_RESULTS_CSV).dropna(subset=["goals_a", "goals_b"])
        except Exception:
            pass
    n_ko = len(ko_rows)
    n_total = n_gs + n_ko

    if n_total == 0:
        return

    score_configs = _ko_score_fns()
    if not score_configs:
        return

    # ── Decide which path to take ────────────────────────────────────────────
    acc_csvs = [csv for _, csv, _ in score_configs]
    gs_complete = all(_row_count(p) >= n_gs and _has_pred_la(p) for p in acc_csvs)
    all_complete = all(_row_count(p) >= n_total for p in acc_csvs)

    if all_complete:
        return  # nothing to do

    if gs_complete:
        # ── FAST PATH: only append new KO rows (no model loading) ───────────
        # Find which KO slots are already recorded in the accuracy CSVs
        ko_mid_set: set[int] = set()
        for _, csv, _ in score_configs:
            if csv.exists():
                try:
                    existing = pd.read_csv(csv, usecols=["match_id"])
                    ko_mid_set |= set(existing["match_id"].astype(int))
                except Exception:
                    pass

        new_ko = ko_rows[
            ko_rows["slot"].map(lambda s: _KO_SLOT_MATCH_ID.get(str(s), 19999)).isin(
                set(_KO_SLOT_MATCH_ID.values()) - ko_mid_set
            )
        ]
        if new_ko.empty:
            return

        for name, csv, score_fn in score_configs:
            new_rows = []
            for _, ko in new_ko.iterrows():
                la = ko.get("pred_la")
                lb = ko.get("pred_lb")
                if la is None or lb is None or (isinstance(la, float) and pd.isna(la)):
                    continue
                la, lb = float(la), float(lb)
                mid = _KO_SLOT_MATCH_ID.get(str(ko["slot"]), 19999)
                try:
                    pa, pb = score_fn(la, lb)
                    new_rows.append({
                        "match_id": mid,
                        "team_a": str(ko.get("team_a", "")),
                        "team_b": str(ko.get("team_b", "")),
                        "pred_goals_a": int(pa), "pred_goals_b": int(pb),
                        "actual_a": int(ko["goals_a"]), "actual_b": int(ko["goals_b"]),
                        "pred_la": la, "pred_lb": lb,
                    })
                except Exception:
                    pass
            if new_rows:
                existing_df = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=_MODEL_ACCURACY_COLS)
                combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
                combined.to_csv(csv, index=False)
        return

    # ── SLOW PATH: group-stage rows missing — full rebuild with model loading ─
    model_configs = _load_all_model_configs()
    if not model_configs:
        return

    updates = pd.DataFrame()
    if UPDATES_CSV.exists():
        updates = pd.read_csv(UPDATES_CSV)
        updates["date"] = pd.to_datetime(updates["date"], errors="coerce")
        updates = updates.dropna(subset=["goals_a", "goals_b"]).sort_values("date").reset_index(drop=True)

    state = initialize_live_state(base_historical, fixtures)
    model_records: dict[str, list] = {cfg["name"]: [] for cfg in model_configs}

    for _, row in updates.iterrows():
        fix_df = state["fixtures"]
        match = fix_df[(fix_df["team_a"] == row["team_a"]) & (fix_df["team_b"] == row["team_b"])]
        reversed_order = False
        if match.empty:
            match = fix_df[(fix_df["team_a"] == row["team_b"]) & (fix_df["team_b"] == row["team_a"])]
            reversed_order = True
        if match.empty:
            continue
        fix = match.iloc[0]
        mid = int(fix["match_id"])
        actual_a = int(row["goals_b"] if reversed_order else row["goals_a"])
        actual_b = int(row["goals_a"] if reversed_order else row["goals_b"])
        ta = normalize_team_name(fix["team_a"])
        tb = normalize_team_name(fix["team_b"])
        match_date = pd.to_datetime(fix["date"])
        for cfg in model_configs:
            try:
                feat = cfg["feature_fn"](
                    team_a=ta, team_b=tb, match_date=match_date,
                    team_states=state["team_states"],
                    historical_matches=state["historical_matches"],
                    market_values=market_values, position_values=position_values,
                    elo_ratings=state["elo_ratings"], rankings=state["rankings"],
                )
                raw_pred = cfg["model"].predict(feat.fillna(0))
                la, lb = float(raw_pred[0, 0]), float(raw_pred[0, 1])
                pa, pb = cfg["score_fn"](la, lb)
                model_records[cfg["name"]].append({
                    "match_id": mid, "team_a": fix["team_a"], "team_b": fix["team_b"],
                    "pred_goals_a": int(pa), "pred_goals_b": int(pb),
                    "actual_a": actual_a, "actual_b": actual_b,
                    "pred_la": la, "pred_lb": lb,
                })
            except Exception:
                pass
        try:
            state = record_match_result(state, mid, actual_a, actual_b)
        except Exception:
            pass

    # Append KO rows in the slow path too
    for _, ko in ko_rows.iterrows():
        slot = str(ko.get("slot", ""))
        la, lb = ko.get("pred_la"), ko.get("pred_lb")
        if la is None or lb is None or (isinstance(la, float) and pd.isna(la)):
            continue
        la, lb = float(la), float(lb)
        mid = _KO_SLOT_MATCH_ID.get(slot, 19999)
        actual_a, actual_b = int(ko["goals_a"]), int(ko["goals_b"])
        for cfg in model_configs:
            try:
                pa, pb = cfg["score_fn"](la, lb)
                model_records[cfg["name"]].append({
                    "match_id": mid, "team_a": str(ko.get("team_a", "")),
                    "team_b": str(ko.get("team_b", "")),
                    "pred_goals_a": int(pa), "pred_goals_b": int(pb),
                    "actual_a": actual_a, "actual_b": actual_b,
                    "pred_la": la, "pred_lb": lb,
                })
            except Exception:
                pass

    for cfg in model_configs:
        records = model_records[cfg["name"]]
        if records:
            df = pd.DataFrame(records, columns=_MODEL_ACCURACY_COLS)
            cfg["csv"].parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cfg["csv"], index=False)


def _compute_accuracy_stats(records: list[dict]) -> dict:
    total = len(records)
    if total == 0:
        return {"total": 0, "exact_correct": 0, "result_correct": 0, "exact_pct": 0.0, "result_pct": 0.0}

    def _res(a, b):
        return "W" if a > b else ("L" if a < b else "D")

    exact = sum(
        1 for g in records
        if int(g["pred_goals_a"]) == int(g["actual_a"]) and int(g["pred_goals_b"]) == int(g["actual_b"])
    )
    correct_result = sum(
        1 for g in records
        if _res(int(g["pred_goals_a"]), int(g["pred_goals_b"])) == _res(int(g["actual_a"]), int(g["actual_b"]))
    )
    return {
        "total": total,
        "exact_correct": exact,
        "result_correct": correct_result,
        "exact_pct": exact / total * 100,
        "result_pct": correct_result / total * 100,
    }


def _load_model_accuracy() -> dict[str, dict]:
    """Load per-model accuracy data from CSVs."""
    result = {}
    for label, csv_path in [("V4", MODEL_ACCURACY_CSV_V4), ("V5", MODEL_ACCURACY_CSV_V5), ("V6", MODEL_ACCURACY_CSV_V6)]:
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            records = df.to_dict("records")
            result[label] = {"stats": _compute_accuracy_stats(records), "records": records}
        except Exception:
            pass
    return result


def _load_accuracy_lookup(model_name: str) -> dict:
    """Return {match_id: {pred_a, pred_b, pred_la, pred_lb}} for the given model."""
    csv_map = {"V4": MODEL_ACCURACY_CSV_V4, "V5": MODEL_ACCURACY_CSV_V5, "V6": MODEL_ACCURACY_CSV_V6}
    csv_path = csv_map.get(model_name.upper())
    if csv_path is None or not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            mid = int(row["match_id"])
            result[mid] = {
                "pred_a": int(row["pred_goals_a"]) if pd.notna(row.get("pred_goals_a")) else None,
                "pred_b": int(row["pred_goals_b"]) if pd.notna(row.get("pred_goals_b")) else None,
                "pred_la": float(row["pred_la"]) if "pred_la" in row.index and pd.notna(row.get("pred_la")) else None,
                "pred_lb": float(row["pred_lb"]) if "pred_lb" in row.index and pd.notna(row.get("pred_lb")) else None,
            }
        return result
    except Exception:
        return {}


def _get_score_fns() -> dict:
    """Return per-model score functions without loading model weights."""
    import json
    from functools import partial
    from src.models.score_conversion import most_likely_score, most_likely_score_v5, most_likely_score_v6

    score_fn_v6 = most_likely_score_v6
    config_path = _MODELS_DIR / "production_config_v6.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as _f:
                _db = json.load(_f).get("drawband", {})
            score_fn_v6 = partial(
                most_likely_score_v6,
                draw_threshold=_db.get("draw_threshold", 0.33),
                threshold_b=_db.get("threshold_b", 0.5),
                scale_c=_db.get("scale_c", 0.9992),
                rho=_db.get("rho", -0.3294),
            )
        except Exception:
            pass
    return {"V4": most_likely_score, "V5": most_likely_score_v5, "V6": score_fn_v6}


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _backfill_calibration(
    base_historical: pd.DataFrame,
    fixtures: pd.DataFrame,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn,
    score_fn=None,
) -> dict:
    """
    Replay every completed WC 2026 game in chronological order, computing the
    model's pre-match prediction at each step.  This gives accurate calibration
    data even for games submitted before the calibration system existed.

    The loop mirrors apply_results_from_csv but also runs the model before
    each record_match_result call, so ELO and form at prediction time are
    exactly what they were before that game was played.
    """
    from src.state.live_state import record_match_result
    from src.state.tournament_calibration import (
        add_game, initialize_calibration, load_prior, save_calibration_to_csv,
    )
    from src.features.team_names import normalize_team_name
    from src.features.build_features import build_pre_match_features
    from src.models.score_conversion import most_likely_score

    _feature_fn = feature_fn or build_pre_match_features
    _score_fn   = score_fn   or most_likely_score

    if not UPDATES_CSV.exists():
        prior = load_prior()
        return initialize_calibration(prior)

    updates = pd.read_csv(UPDATES_CSV)
    updates["date"] = pd.to_datetime(updates["date"], errors="coerce")
    updates = (
        updates
        .dropna(subset=["goals_a", "goals_b"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    prior = load_prior()
    calibration = initialize_calibration(prior)

    # Start from a clean slate so pre-match state is accurate at every step
    state = initialize_live_state(base_historical, fixtures)

    for _, row in updates.iterrows():
        # Locate the fixture (try both orderings)
        fix_df = state["fixtures"]
        match = fix_df[
            (fix_df["team_a"] == row["team_a"]) & (fix_df["team_b"] == row["team_b"])
        ]
        reversed_order = False
        if match.empty:
            match = fix_df[
                (fix_df["team_a"] == row["team_b"]) & (fix_df["team_b"] == row["team_a"])
            ]
            reversed_order = True
        if match.empty:
            continue

        fix = match.iloc[0]
        mid = int(fix["match_id"])
        actual_a = int(row["goals_b"] if reversed_order else row["goals_a"])
        actual_b = int(row["goals_a"] if reversed_order else row["goals_b"])

        # Compute model prediction BEFORE recording (pre-match state)
        try:
            ta = normalize_team_name(fix["team_a"])
            tb = normalize_team_name(fix["team_b"])
            feat = _feature_fn(
                team_a=ta,
                team_b=tb,
                match_date=pd.to_datetime(fix["date"]),
                team_states=state["team_states"],
                historical_matches=state["historical_matches"],
                market_values=market_values,
                position_values=position_values,
                elo_ratings=state["elo_ratings"],
                rankings=state["rankings"],
            )
            raw_pred = model.predict(feat.fillna(0))
            la_raw = float(raw_pred[0, 0])
            lb_raw = float(raw_pred[0, 1])
            pa, pb = _score_fn(la_raw, lb_raw)
            calibration = add_game(
                calibration, mid, la_raw, lb_raw, actual_a, actual_b,
                pred_goals_a=pa, pred_goals_b=pb,
            )
        except Exception:
            pass  # skip calibration for this game but still advance state

        # Advance state with the actual result
        try:
            state = record_match_result(state, mid, actual_a, actual_b)
        except Exception:
            pass

    save_calibration_to_csv(calibration, CALIBRATION_CSV)
    return calibration


def _ensure_calibration(
    state: dict,
    base_historical: pd.DataFrame,
    fixtures: pd.DataFrame,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn,
    score_fn=None,
) -> dict:
    """
    If the calibration has fewer games than completed fixtures, backfill.
    Returns the (possibly updated) calibration dict.
    """
    from src.state.tournament_calibration import (
        initialize_calibration, load_prior,
    )

    completed_count = int(state["fixtures"]["is_completed"].sum())
    cal_count = len(state.get("calibration", {}).get("games", []))

    if cal_count >= completed_count:
        return state.get("calibration", initialize_calibration(load_prior()))

    return _backfill_calibration(
        base_historical, fixtures, model, market_values, position_values,
        feature_fn, score_fn=score_fn,
    )


def _build_match_predictions(calibration: dict) -> dict:
    """Build a match_id → {pred_a, pred_b} lookup from calibration games."""
    return {
        g["match_id"]: {
            "pred_a": g.get("pred_goals_a"),
            "pred_b": g.get("pred_goals_b"),
            "pred_la": g["pred_la"],
            "pred_lb": g["pred_lb"],
        }
        for g in calibration.get("games", [])
        if g.get("pred_goals_a") is not None
    }


def _apply_ko_results_to_state(state: dict, ko_results: dict) -> dict:
    """Inject completed KO results into state["historical_matches"] and update ELO/rankings.

    Only appends slots not already present in historical_matches (idempotent).
    KO results do NOT update team_states (those are group-stage standings only).
    """
    from src.state.elo import compute_elo_update
    from src.features.team_names import normalize_team_name
    from src.state.live_state import derive_rankings_from_elo

    # Track (team_a_canon, team_b_canon, date_str) already applied to avoid double-counting.
    hist = state["historical_matches"]
    if "source_file" in hist.columns:
        ko_hist = hist[hist["source_file"] == "ko_2026"]
        existing_keys: set[tuple] = {
            (str(r["team_a"]), str(r["team_b"]), str(r["date"])[:10])
            for _, r in ko_hist.iterrows()
        }
    else:
        existing_keys = set()

    # Process slots in chronological order
    sorted_slots = sorted(
        [(slot, res) for slot, res in ko_results.items()
         if res.get("team_a") and res.get("team_b")],
        key=lambda x: _KO_SLOT_DATES.get(x[0], pd.Timestamp("2099-01-01")),
    )

    new_rows = []
    for slot, res in sorted_slots:
        team_a = res["team_a"]
        team_b = res["team_b"]
        goals_a = int(res["goals_a"])
        goals_b = int(res["goals_b"])
        match_date = _KO_SLOT_DATES.get(slot, pd.Timestamp("2026-07-01"))
        date_str = str(match_date)[:10]

        team_a_c = normalize_team_name(team_a)
        team_b_c = normalize_team_name(team_b)

        if (team_a_c, team_b_c, date_str) in existing_keys:
            continue  # already applied

        rating_a = state["elo_ratings"].get(team_a_c, state["elo_ratings"].get(team_a, 1500.0))
        rating_b = state["elo_ratings"].get(team_b_c, state["elo_ratings"].get(team_b, 1500.0))
        rank_a   = state["rankings"].get(team_a_c, state["rankings"].get(team_a, 0))
        rank_b   = state["rankings"].get(team_b_c, state["rankings"].get(team_b, 0))

        delta_a, delta_b = compute_elo_update(
            rating_a=rating_a, rating_b=rating_b,
            goals_a=goals_a, goals_b=goals_b,
            competition="FIFA World Cup",
            team_a=team_a, team_b=team_b,
            location="neutral", stage=slot, knockout=True,
        )
        rating_a_after = rating_a + delta_a
        rating_b_after = rating_b + delta_b

        new_rows.append({
            "date": match_date,
            "team_a": team_a_c, "team_b": team_b_c,
            "goals_a": goals_a, "goals_b": goals_b,
            "competition": "FIFA World Cup",
            "location": "neutral",
            "rating_change_a": delta_a, "rating_change_b": delta_b,
            "rating_a": rating_a_after, "rating_b": rating_b_after,
            "rating_a_before": rating_a, "rating_b_before": rating_b,
            "rank_a": rank_a, "rank_b": rank_b,
            "rank_a_before": rank_a, "rank_b_before": rank_b,
            "rank_change_a": 0, "rank_change_b": 0,
            "elo_diff": rating_a - rating_b,
            "rank_diff": rank_a - rank_b,
            "source_file": "ko_2026",
            "tournament_year": 2026,
            "tournament_key": "FIFA World Cup_2026",
        })

        # Update ELO immediately so subsequent KO games see updated ratings
        state["elo_ratings"][team_a_c] = rating_a_after
        state["elo_ratings"][team_b_c] = rating_b_after
        if team_a != team_a_c:
            state["elo_ratings"][team_a] = rating_a_after
        if team_b != team_b_c:
            state["elo_ratings"][team_b] = rating_b_after
        state["rankings"] = derive_rankings_from_elo(state["elo_ratings"])
        existing_keys.add((team_a_c, team_b_c, date_str))

    if new_rows:
        state["historical_matches"] = pd.concat(
            [state["historical_matches"], pd.DataFrame(new_rows)],
            ignore_index=True,
        )

    return state


def _init_state(
    historical_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    model=None,
    market_values: pd.DataFrame | None = None,
    position_values: pd.DataFrame | None = None,
    feature_fn=None,
    score_fn=None,
) -> None:
    if "true_state" not in st.session_state:
        from src.state.tournament_calibration import (
            initialize_calibration, load_calibration_from_csv, load_prior,
        )
        state = initialize_live_state(historical_matches, fixtures)
        state = apply_results_from_csv(state, UPDATES_CSV)
        # Apply any completed KO results so ELO/form reflects knockout games
        ko_results = _load_ko_results()
        if ko_results:
            state = _apply_ko_results_to_state(state, ko_results)
            st.session_state.ko_results = ko_results

        prior = load_prior()
        state["calibration"] = initialize_calibration(prior)
        state["calibration"] = load_calibration_from_csv(state["calibration"], CALIBRATION_CSV)

        if model is not None and market_values is not None and position_values is not None:
            state["calibration"] = _ensure_calibration(
                state, historical_matches, fixtures,
                model, market_values, position_values, feature_fn, score_fn=score_fn,
            )

        state["match_predictions"] = _build_match_predictions(state["calibration"])

        if market_values is not None and position_values is not None:
            _backfill_model_accuracy(historical_matches, fixtures, market_values, position_values)

        st.session_state.true_state = state
        st.session_state.sim_state = None
        st.session_state._base_historical = historical_matches
        st.session_state._base_fixtures = fixtures


def _refresh_from_csv(
    model=None,
    market_values: pd.DataFrame | None = None,
    position_values: pd.DataFrame | None = None,
    feature_fn=None,
    score_fn=None,
) -> None:
    from src.state.tournament_calibration import (
        initialize_calibration, load_calibration_from_csv, load_prior,
    )

    state = apply_results_from_csv(st.session_state.true_state, UPDATES_CSV)

    prior = load_prior()
    state["calibration"] = initialize_calibration(prior)
    state["calibration"] = load_calibration_from_csv(state["calibration"], CALIBRATION_CSV)

    if model is not None and market_values is not None and position_values is not None:
        base_hist = st.session_state.get("_base_historical")
        base_fix  = st.session_state.get("_base_fixtures")
        if base_hist is not None and base_fix is not None:
            state["calibration"] = _ensure_calibration(
                state, base_hist, base_fix,
                model, market_values, position_values, feature_fn, score_fn=score_fn,
            )

    state["match_predictions"] = _build_match_predictions(state["calibration"])

    # Re-apply all KO results (refresh rebuilds state from scratch)
    ko_results = _load_ko_results()
    if ko_results:
        state = _apply_ko_results_to_state(state, ko_results)
        st.session_state.ko_results = ko_results

    if market_values is not None and position_values is not None:
        base_hist2 = st.session_state.get("_base_historical")
        base_fix2  = st.session_state.get("_base_fixtures")
        if base_hist2 is not None and base_fix2 is not None:
            _backfill_model_accuracy(base_hist2, base_fix2, market_values, position_values)

    st.session_state.true_state = state
    st.session_state.sim_state = None


def _submit_result(
    match_id: int,
    goals_a: int,
    goals_b: int,
    model=None,
    market_values: pd.DataFrame | None = None,
    position_values: pd.DataFrame | None = None,
    feature_fn=None,
    score_fn=None,
) -> None:
    from src.state.tournament_calibration import (
        add_game, save_calibration_to_csv,
    )
    from src.features.team_names import normalize_team_name
    from src.features.build_features import build_pre_match_features

    state = st.session_state.true_state

    fixture_match = state["fixtures"][state["fixtures"]["match_id"] == int(match_id)]
    if fixture_match.empty:
        st.error(f"Could not find fixture with match_id={match_id}")
        return

    fixture = fixture_match.iloc[0]

    from src.models.score_conversion import most_likely_score

    _feature_fn = feature_fn or build_pre_match_features
    _score_fn   = score_fn   or most_likely_score

    # Compute model prediction BEFORE recording the result (pre-match state).
    la_raw, lb_raw, pred_a, pred_b = None, None, None, None
    if model is not None and market_values is not None and position_values is not None:
        try:
            ta_canon = normalize_team_name(fixture["team_a"])
            tb_canon = normalize_team_name(fixture["team_b"])
            feat = _feature_fn(
                team_a=ta_canon,
                team_b=tb_canon,
                match_date=pd.to_datetime(fixture["date"]),
                team_states=state["team_states"],
                historical_matches=state["historical_matches"],
                market_values=market_values,
                position_values=position_values,
                elo_ratings=state["elo_ratings"],
                rankings=state["rankings"],
            )
            raw_pred = model.predict(feat.fillna(0))
            la_raw = float(raw_pred[0, 0])
            lb_raw = float(raw_pred[0, 1])
            pred_a, pred_b = _score_fn(la_raw, lb_raw)
        except Exception:
            pass  # calibration skipped for this game if prediction fails

    persist_real_result_to_csv(fixture=fixture, goals_a=goals_a, goals_b=goals_b)

    new_state = record_match_result(state, match_id, goals_a, goals_b)

    if la_raw is not None and lb_raw is not None:
        new_state["calibration"] = add_game(
            new_state.get("calibration", state.get("calibration", {})),
            match_id, la_raw, lb_raw, goals_a, goals_b,
            pred_goals_a=pred_a, pred_goals_b=pred_b,
        )
        save_calibration_to_csv(new_state["calibration"], CALIBRATION_CSV)
    elif "calibration" in state:
        new_state["calibration"] = state["calibration"]

    # Update the match_predictions lookup so the completed card shows prediction
    preds = dict(new_state.get("match_predictions", state.get("match_predictions", {})))
    if pred_a is not None:
        preds[int(match_id)] = {
            "pred_a": pred_a, "pred_b": pred_b,
            "pred_la": la_raw, "pred_lb": lb_raw,
        }
    new_state["match_predictions"] = preds

    st.session_state.true_state = new_state
    st.session_state.sim_state = None


# ---------------------------------------------------------------------------
# Match prediction helper
# ---------------------------------------------------------------------------

def _get_prediction(
    model,
    state: dict,
    team_a: str,
    team_b: str,
    match_date,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
    use_calibration: bool = True,
) -> dict | None:
    from src.features.build_features import build_pre_match_features
    from src.features.team_names import normalize_team_name
    from src.models.score_conversion import most_likely_score, win_draw_loss_probs, top_scores

    if feature_fn is None:
        feature_fn = build_pre_match_features
    if score_fn is None:
        score_fn = most_likely_score

    try:
        from src.state.tournament_calibration import calibrate_lambdas, calibrate_win_draw_loss

        feature_row = feature_fn(
            team_a=normalize_team_name(team_a),
            team_b=normalize_team_name(team_b),
            match_date=match_date,
            team_states=state["team_states"],
            historical_matches=state["historical_matches"],
            market_values=market_values,
            position_values=position_values,
            elo_ratings=state["elo_ratings"],
            rankings=state["rankings"],
        )
        pred = model.predict(feature_row.fillna(0))
        la_raw = float(pred[0, 0])
        lb_raw = float(pred[0, 1])

        # Apply tournament calibration only when enabled
        calibration = state.get("calibration") if use_calibration else None
        if calibration is not None:
            la, lb = calibrate_lambdas(la_raw, lb_raw, calibration)
        else:
            la, lb = la_raw, lb_raw

        ga, gb = score_fn(la, lb)
        win_a, draw, win_b = win_draw_loss_probs(la, lb)

        if calibration is not None:
            win_a, draw, win_b = calibrate_win_draw_loss(win_a, draw, win_b, calibration)

        return {
            "lambda_a": la_raw,   # raw, for storing in calibration on submit
            "lambda_b": lb_raw,
            "lambda_a_cal": la,   # calibrated, for display
            "lambda_b_cal": lb,
            "pred_goals_a": ga,
            "pred_goals_b": gb,
            "win_a": win_a,
            "draw": draw,
            "win_b": win_b,
            "top_scores": [
                {
                    "score": f"{a}-{b}",
                    "team_a_goals": a,
                    "team_b_goals": b,
                    "probability_%": round(prob * 100, 2),
                }
                for a, b, prob in top_scores(la, lb, n=10)
            ],
        }
    except Exception as e:
        # Surface the error so it's debuggable from the UI
        return {"_error": str(e)}


# ---------------------------------------------------------------------------
# Match card rendering
# ---------------------------------------------------------------------------

def _render_completed_match(fixture: pd.Series, pred: dict | None = None) -> None:
    ga = int(fixture["goals_a"])
    gb = int(fixture["goals_b"])
    ta, tb = fixture["team_a"], fixture["team_b"]
    time_str = pd.to_datetime(fixture["date"], utc=True).strftime("%H:%M UTC")

    if ga > gb:
        result = f"**{_team_label(ta)}  {ga} – {gb}  {_team_label(tb)}**  ✅ {ta} wins"
    elif gb > ga:
        result = f"**{_team_label(ta)}  {ga} – {gb}  {_team_label(tb)}**  ✅ {tb} wins"
    else:
        result = f"**{_team_label(ta)}  {ga} – {gb}  {_team_label(tb)}**  🤝 Draw"

    pred_str = ""
    if pred and pred.get("pred_a") is not None and pred.get("pred_b") is not None:
        pred_str = f"  ·  *Model predicted: {pred['pred_a']}–{pred['pred_b']}*"

    st.success(f"🕐 {time_str}  |  {result}{pred_str}")


def _render_upcoming_match(
    fixture: pd.Series,
    model,
    state: dict,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
    use_calibration: bool = True,
) -> None:
    ta, tb = fixture["team_a"], fixture["team_b"]
    mid = int(fixture["match_id"])
    match_date = fixture["date"]
    time_str = pd.to_datetime(match_date, utc=True).strftime("%H:%M UTC")

    pred = _get_prediction(model, state, ta, tb, match_date, market_values, position_values, feature_fn=feature_fn, score_fn=score_fn, use_calibration=use_calibration)

    with st.container(border=True):
        header_col, prob_col = st.columns([2, 3])

        with header_col:
            st.markdown(f"**🕐 {time_str}**")
            st.markdown(f"{_team_label(ta)}  vs  {_team_label(tb)}")
            if pred and "_error" not in pred:
                la_cal = pred.get("lambda_a_cal", pred["lambda_a"])
                lb_cal = pred.get("lambda_b_cal", pred["lambda_b"])
                st.markdown(
                    f"Model prediction: **{pred['pred_goals_a']} – {pred['pred_goals_b']}**"
                    f"  *(xG {la_cal:.2f} – {lb_cal:.2f})*"
                )
            elif pred and "_error" in pred:
                st.caption(f"⚠️ Prediction error: {pred['_error']}")
            else:
                st.caption("*Prediction unavailable*")

        with prob_col:
            if pred and "_error" not in pred:
                c1, c2, c3 = st.columns(3)
                c1.metric(_team_label(ta), f"{pred['win_a'] * 100:.0f}%")
                c2.metric("Draw", f"{pred['draw'] * 100:.0f}%")
                c3.metric(_team_label(tb), f"{pred['win_b'] * 100:.0f}%")
        if pred and "_error" not in pred:
            with st.expander("📊 Show most likely scorelines"):
                score_options = pd.DataFrame(pred["top_scores"])
                st.dataframe(
                    score_options,
                    use_container_width=True,
                    hide_index=True,
                )
        with st.expander("Enter actual result"):
            fc1, fc2, fc3 = st.columns([2, 1, 2])
            with fc1:
                ga_input = st.number_input(
                    f"{ta} goals", min_value=0, max_value=20, value=0,
                    key=f"ga_{mid}",
                )
            with fc2:
                st.markdown("<br><div style='text-align:center'>–</div>",
                            unsafe_allow_html=True)
            with fc3:
                gb_input = st.number_input(
                    f"{tb} goals", min_value=0, max_value=20, value=0,
                    key=f"gb_{mid}",
                )
            if st.button("✅ Submit result", key=f"submit_{mid}"):
                _submit_result(
                    mid, int(ga_input), int(gb_input),
                    model=model,
                    market_values=market_values,
                    position_values=position_values,
                    feature_fn=feature_fn,
                    score_fn=score_fn,
                )
                st.success("Result saved, ELO updated, calibration updated.")
                st.rerun()


# ---------------------------------------------------------------------------
# Knockout match rendering (completed + upcoming with full prediction UI)
# ---------------------------------------------------------------------------

def _render_ko_completed_match(
    slot: str,
    team_a: str,
    team_b: str,
    result: dict,
    stage_label: str,
) -> None:
    ga     = result["goals_a"]
    gb     = result["goals_b"]
    winner = result.get("winner")
    pred_a = result.get("pred_a")
    pred_b = result.get("pred_b")

    if ga > gb:
        outcome = f"✅ {team_a} wins"
    elif gb > ga:
        outcome = f"✅ {team_b} wins"
    else:
        if winner:
            outcome = f"🤝 Draw (aet) → ⚽ **{winner}** advances on penalties"
        else:
            outcome = "🤝 Draw (extra time / penalties)"

    pred_str = f"  ·  *Model predicted: {pred_a}–{pred_b}*" if pred_a is not None else ""
    score_col, btn_col = st.columns([9, 1])
    with score_col:
        st.success(
            f"**{_team_label(team_a)}  {ga} – {gb}  {_team_label(team_b)}**"
            f"  {outcome}  ·  {stage_label} · {slot.replace('_', '-')}{pred_str}"
        )
    with btn_col:
        if st.button("🗑️", key=f"ko_remove_{slot}", help="Remove this result and return to prediction view"):
            _remove_ko_result(slot)
            st.rerun()


# ---------------------------------------------------------------------------
# Bracket parent-slot resolution helper
# ---------------------------------------------------------------------------

_R16_PARENTS: dict[str, tuple[str, str]] = {
    "R16_01": ("R32_01", "R32_02"), "R16_02": ("R32_03", "R32_04"),
    "R16_03": ("R32_05", "R32_06"), "R16_04": ("R32_07", "R32_08"),
    "R16_05": ("R32_09", "R32_10"), "R16_06": ("R32_11", "R32_12"),
    "R16_07": ("R32_13", "R32_14"), "R16_08": ("R32_15", "R32_16"),
}
_QF_PARENTS: dict[str, tuple[str, str]] = {
    "QF_01": ("R16_01", "R16_02"), "QF_02": ("R16_03", "R16_04"),
    "QF_03": ("R16_05", "R16_06"), "QF_04": ("R16_07", "R16_08"),
}
_SF_PARENTS: dict[str, tuple[str, str]] = {
    "SF_01": ("QF_01", "QF_02"), "SF_02": ("QF_03", "QF_04"),
}


def _resolve_winner(parent_slot: str, ko_results: dict, r32_lookup: dict) -> str:
    """Return the team that won *parent_slot*, or a descriptive placeholder."""
    if parent_slot in ko_results:
        w = ko_results[parent_slot].get("winner")
        if w:
            return w
        # Result stored but winner field missing (old record without penalty info)
        r = ko_results[parent_slot]
        ga, gb = r["goals_a"], r["goals_b"]
        if ga > gb:
            return r.get("team_a", f"W of {parent_slot.replace('_','-')}")
        if gb > ga:
            return r.get("team_b", f"W of {parent_slot.replace('_','-')}")
    # Not yet played
    if parent_slot.startswith("R32_"):
        num = int(parent_slot[-2:])
        if parent_slot in r32_lookup:
            ta, tb = r32_lookup[parent_slot]
            return f"W of M{num} ({ta} vs {tb})"
        return f"W of M{num}"
    return f"W of {parent_slot.replace('_', '-')}"


def _resolve_loser(parent_slot: str, ko_results: dict) -> str:
    """Return the team that lost *parent_slot*, or a placeholder."""
    if parent_slot in ko_results:
        r = ko_results[parent_slot]
        w = r.get("winner")
        ta, tb = r.get("team_a", ""), r.get("team_b", "")
        if w == ta and tb:
            return tb
        if w == tb and ta:
            return ta
    return f"L of {parent_slot.replace('_', '-')}"


def _resolve_ko_match_teams(
    slot: str,
    ko_results: dict,
    r32_lookup: dict,
) -> tuple[str, str]:
    """Resolve the two teams for any knockout slot using known results where available."""
    if slot.startswith("R32_"):
        return r32_lookup.get(slot, ("TBD", "TBD"))
    if slot in _R16_PARENTS:
        pa, pb = _R16_PARENTS[slot]
        return _resolve_winner(pa, ko_results, r32_lookup), _resolve_winner(pb, ko_results, r32_lookup)
    if slot in _QF_PARENTS:
        pa, pb = _QF_PARENTS[slot]
        return _resolve_winner(pa, ko_results, r32_lookup), _resolve_winner(pb, ko_results, r32_lookup)
    if slot in _SF_PARENTS:
        pa, pb = _SF_PARENTS[slot]
        return _resolve_winner(pa, ko_results, r32_lookup), _resolve_winner(pb, ko_results, r32_lookup)
    if slot == "FINAL":
        return _resolve_winner("SF_01", ko_results, r32_lookup), _resolve_winner("SF_02", ko_results, r32_lookup)
    if slot == "THIRD_PLACE":
        return _resolve_loser("SF_01", ko_results), _resolve_loser("SF_02", ko_results)
    return ("TBD", "TBD")


def _is_real_team(name: str) -> bool:
    """True if name is an actual team, not a placeholder."""
    return bool(name) and not name.startswith(("W of", "L of", "Winner", "Loser", "TBD"))


# ---------------------------------------------------------------------------
# Knockout match rendering (full prediction UI with penalty support)
# ---------------------------------------------------------------------------

def _render_ko_upcoming_match(
    slot: str,
    team_a: str,
    team_b: str,
    date,
    stage_label: str,
    model,
    state: dict,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
    use_calibration: bool = True,
) -> None:
    """Full prediction UI for a knockout match with known teams — mirrors group stage upcoming card."""
    ko_results: dict = st.session_state.get("ko_results", {})
    if slot in ko_results:
        _render_ko_completed_match(slot, team_a, team_b, ko_results[slot], stage_label)
        return

    match_date = date if isinstance(date, pd.Timestamp) else pd.Timestamp(date)

    pred = _get_prediction(
        model, state, team_a, team_b, match_date,
        market_values, position_values,
        feature_fn=feature_fn, score_fn=score_fn, use_calibration=use_calibration,
    )

    with st.container(border=True):
        header_col, prob_col = st.columns([2, 3])

        with header_col:
            st.markdown(f"**{stage_label} · {slot.replace('_', '-')}**")
            st.markdown(f"{_team_label(team_a)}  vs  {_team_label(team_b)}")
            if pred and "_error" not in pred:
                la_cal = pred.get("lambda_a_cal", pred["lambda_a"])
                lb_cal = pred.get("lambda_b_cal", pred["lambda_b"])
                st.markdown(
                    f"Model prediction: **{pred['pred_goals_a']} – {pred['pred_goals_b']}**"
                    f"  *(xG {la_cal:.2f} – {lb_cal:.2f})*"
                )
            elif pred and "_error" in pred:
                st.caption(f"⚠️ Prediction error: {pred['_error']}")
            else:
                st.caption("*Prediction unavailable*")

        with prob_col:
            if pred and "_error" not in pred:
                c1, c2, c3 = st.columns(3)
                c1.metric(_team_label(team_a), f"{pred['win_a'] * 100:.0f}%")
                c2.metric("Draw", f"{pred['draw'] * 100:.0f}%")
                c3.metric(_team_label(team_b), f"{pred['win_b'] * 100:.0f}%")

        if pred and "_error" not in pred:
            with st.expander("📊 Show most likely scorelines"):
                st.dataframe(
                    pd.DataFrame(pred["top_scores"]),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Enter actual result"):
            fc1, fc2, fc3 = st.columns([2, 1, 2])
            with fc1:
                ga_input = st.number_input(
                    f"{team_a} goals", min_value=0, max_value=20, value=0,
                    key=f"ko_ga_{slot}",
                )
            with fc2:
                st.markdown("<br><div style='text-align:center'>–</div>", unsafe_allow_html=True)
            with fc3:
                gb_input = st.number_input(
                    f"{team_b} goals", min_value=0, max_value=20, value=0,
                    key=f"ko_gb_{slot}",
                )

            # Penalty winner selection — only shown on a draw
            pens_winner = None
            if int(ga_input) == int(gb_input):
                st.info("⚽ Draw after 90 min — who advanced on penalties?")
                pens_winner = st.radio(
                    "Penalty winner",
                    options=[team_a, team_b],
                    key=f"ko_pens_{slot}",
                    horizontal=True,
                    label_visibility="collapsed",
                    index=None,
                )

            can_submit = (int(ga_input) != int(gb_input)) or (pens_winner is not None)

            if st.button("✅ Submit result", key=f"ko_submit_{slot}", disabled=not can_submit):
                if int(ga_input) > int(gb_input):
                    winner = team_a
                elif int(gb_input) > int(ga_input):
                    winner = team_b
                else:
                    winner = pens_winner

                pred_a  = pred.get("pred_goals_a") if pred and "_error" not in pred else None
                pred_b  = pred.get("pred_goals_b") if pred and "_error" not in pred else None
                pred_la = pred.get("lambda_a")     if pred and "_error" not in pred else None
                pred_lb = pred.get("lambda_b")     if pred and "_error" not in pred else None
                _save_ko_result(
                    slot=slot, team_a=team_a, team_b=team_b, date=match_date,
                    goals_a=int(ga_input), goals_b=int(gb_input),
                    winner=winner,
                    pred_a=pred_a, pred_b=pred_b, pred_la=pred_la, pred_lb=pred_lb,
                )
                st.success("Knockout result saved!")
                st.rerun()

            if int(ga_input) == int(gb_input) and pens_winner is None:
                st.caption("Select the penalty winner above before submitting.")


# ---------------------------------------------------------------------------
# Smart knockout slot renderer (dispatches completed / full-UI / placeholder)
# ---------------------------------------------------------------------------

def _render_ko_slot(
    slot: str,
    ko_results: dict,
    r32_lookup: dict,
    stage_label: str,
    slot_date,
    model,
    state: dict,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
    use_calibration: bool = True,
) -> None:
    """Render any knockout slot: completed card, full prediction UI, or info placeholder."""
    # Already completed?
    if slot in ko_results:
        result = ko_results[slot]
        ta = result.get("team_a") or r32_lookup.get(slot, ("?", "?"))[0]
        tb = result.get("team_b") or r32_lookup.get(slot, ("?", "?"))[1]
        _render_ko_completed_match(slot, ta, tb, result, stage_label)

        # Draw with no winner yet → show penalty picker so user can fill it in
        ga, gb = result["goals_a"], result["goals_b"]
        if ga == gb and not result.get("winner"):
            st.info("⚽ This match ended in a draw — who advanced on penalties?")
            pens_winner = st.radio(
                "Penalty winner",
                options=[ta, tb],
                key=f"ko_pens_retro_{slot}",
                horizontal=True,
                label_visibility="collapsed",
                index=None,
            )
            if st.button("✅ Confirm penalty winner", key=f"ko_pens_confirm_{slot}",
                         disabled=pens_winner is None):
                _update_ko_winner(slot, pens_winner)
                st.success(f"{pens_winner} advances!")
                st.rerun()
        return

    # Resolve teams for this slot
    ta, tb = _resolve_ko_match_teams(slot, ko_results, r32_lookup)

    if _is_real_team(ta) and _is_real_team(tb):
        # Both teams known → full prediction UI
        _render_ko_upcoming_match(
            slot=slot, team_a=ta, team_b=tb, date=slot_date,
            stage_label=stage_label, model=model, state=state,
            market_values=market_values, position_values=position_values,
            feature_fn=feature_fn, score_fn=score_fn, use_calibration=use_calibration,
        )
    else:
        # Teams not yet determined → info card
        _render_knockout_fixture(slot, ta, tb, stage_label)


# ---------------------------------------------------------------------------
# Fixtures by day tab
# ---------------------------------------------------------------------------

def _show_group_stage(
    state: dict,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
    score_fn=None,
    use_calibration: bool = True,
    model_name: str = "V4",
) -> None:
    from src.models.score_conversion import most_likely_score

    fixtures = state["fixtures"].copy()

    # Load accuracy lookup for the current model (for dynamic completed-match display)
    accuracy_lookup = _load_accuracy_lookup(model_name)
    calibration = state.get("calibration") if use_calibration else None

    fixtures["_date_label"] = pd.to_datetime(
        fixtures["date"], utc=True
    ).dt.strftime("%b %d")
    fixtures["_date_sort"] = pd.to_datetime(fixtures["date"], utc=True).dt.date

    dates_df = (
        fixtures[["_date_label", "_date_sort"]]
        .drop_duplicates()
        .sort_values("_date_sort")
    )
    group_date_labels = dates_df["_date_label"].tolist()

    if not group_date_labels:
        st.info("No fixtures loaded.")
        return

    # Build knockout day labels (deduplicated by date, preserving insertion order)
    ko_date_to_slots: dict[str, list[str]] = {}
    for slot, ts in sorted(_KO_SLOT_DATES.items(), key=lambda x: (x[1], x[0])):
        label = ts.strftime("%b %d")
        ko_date_to_slots.setdefault(label, []).append(slot)

    group_date_set = set(group_date_labels)
    # Dates that appear in BOTH group stage and knockout (e.g. Jun 28)
    overlap_dates = group_date_set & set(ko_date_to_slots.keys())
    # Append only purely-knockout dates (not already in group stage list)
    ko_only_labels = [d for d in ko_date_to_slots if d not in group_date_set]

    all_date_labels = group_date_labels + ko_only_labels

    # Lazy-load knockout results into session state
    if "ko_results" not in st.session_state:
        st.session_state.ko_results = _load_ko_results()

    selected_date = st.selectbox("Select match day", all_date_labels, key="day_selector")

    # ── Group stage day (possibly with knockout fixtures on same calendar day) ──
    day_fixtures = fixtures[fixtures["_date_label"] == selected_date].sort_values("date")

    has_group_games = not day_fixtures.empty
    has_ko_games    = selected_date in ko_date_to_slots

    # Determine labels for the banner
    if has_group_games and has_ko_games:
        # Overlap day — show group stage games first, then knockout
        groups_today    = sorted(day_fixtures["group"].unique())
        matchdays_today = sorted(day_fixtures["matchday"].unique())
        st.markdown(
            f"### {selected_date}  —  Matchday {', '.join(str(m) for m in matchdays_today)}"
            f"  ·  Groups: {', '.join(groups_today)}  ·  +{_KO_STAGE_DISPLAY.get(_KO_ROUND_OF_SLOT.get(ko_date_to_slots[selected_date][0],''),'Knockout')}"
        )
    elif has_group_games:
        groups_today    = sorted(day_fixtures["group"].unique())
        matchdays_today = sorted(day_fixtures["matchday"].unique())
        st.markdown(
            f"### {selected_date}  —  Matchday {', '.join(str(m) for m in matchdays_today)}"
            f"  ·  Groups: {', '.join(groups_today)}"
        )
    elif has_ko_games:
        slots_today    = ko_date_to_slots[selected_date]
        round_key      = _KO_ROUND_OF_SLOT.get(slots_today[0], "")
        stage_label_ko = _KO_STAGE_DISPLAY.get(round_key, round_key)
        st.markdown(f"### {selected_date}  —  {stage_label_ko}")

    # ── Render group-stage fixtures (if any) ────────────────────────────────
    if has_group_games:
        _score_fn = score_fn or most_likely_score

        for _, fix in day_fixtures.iterrows():
            if bool(fix.get("is_completed", False)):
                mid = int(fix["match_id"])
                acc = accuracy_lookup.get(mid)
                if acc and acc.get("pred_a") is not None:
                    if calibration is not None and acc.get("pred_la") is not None:
                        from src.state.tournament_calibration import get_factors
                        fac = get_factors(calibration)
                        la_cal = float(acc["pred_la"]) * fac["goal_scale"]
                        lb_cal = float(acc["pred_lb"]) * fac["goal_scale"]
                        pa, pb = _score_fn(la_cal, lb_cal)
                        pred_display = {"pred_a": pa, "pred_b": pb}
                    else:
                        pred_display = {"pred_a": acc["pred_a"], "pred_b": acc["pred_b"]}
                else:
                    pred_display = state.get("match_predictions", {}).get(mid)
                _render_completed_match(fix, pred=pred_display)
            else:
                _render_upcoming_match(
                    fix, model, state, market_values, position_values,
                    feature_fn=feature_fn, score_fn=score_fn, use_calibration=use_calibration,
                )

        day_completed = day_fixtures[day_fixtures["is_completed"]]
        if not day_completed.empty:
            goals_today = int(day_completed["goals_a"].sum() + day_completed["goals_b"].sum())
            st.caption(f"{len(day_completed)} result(s) recorded today · {goals_today} goals")

        if has_ko_games:
            st.divider()

    # ── Render knockout fixtures (if any) ───────────────────────────────────
    if has_ko_games:
        slots_today = ko_date_to_slots[selected_date]
        round_key   = _KO_ROUND_OF_SLOT.get(slots_today[0], "")
        stage_label_ko = _KO_STAGE_DISPLAY.get(round_key, round_key)

        if has_group_games:
            st.markdown(f"#### {stage_label_ko}")

        # Resolve R32 teams for R32 slots and R16 enrichment
        r32_lookup: dict[str, tuple[str, str]] = {}
        try:
            r32_lookup = _build_r32_teams_lookup(state)
        except Exception:
            pass

        ko_results = st.session_state.get("ko_results", {})

        for slot in slots_today:
            slot_date = _KO_SLOT_DATES.get(slot)
            _render_ko_slot(
                slot=slot,
                ko_results=ko_results,
                r32_lookup=r32_lookup,
                stage_label=stage_label_ko,
                slot_date=slot_date,
                model=model,
                state=state,
                market_values=market_values,
                position_values=position_values,
                feature_fn=feature_fn,
                score_fn=score_fn,
                use_calibration=use_calibration,
            )


# ---------------------------------------------------------------------------
# Knockout schedule constants (dates / team slot descriptions)
# ---------------------------------------------------------------------------

_KO_SLOT_DATES: dict[str, pd.Timestamp] = {
    # Round of 32
    "R32_03": pd.Timestamp("2026-06-28"),
    "R32_01": pd.Timestamp("2026-06-29"), "R32_04": pd.Timestamp("2026-06-29"), "R32_09": pd.Timestamp("2026-06-29"),
    "R32_02": pd.Timestamp("2026-06-30"), "R32_10": pd.Timestamp("2026-06-30"), "R32_11": pd.Timestamp("2026-06-30"),
    "R32_12": pd.Timestamp("2026-07-01"), "R32_07": pd.Timestamp("2026-07-01"), "R32_08": pd.Timestamp("2026-07-01"),
    "R32_05": pd.Timestamp("2026-07-02"), "R32_06": pd.Timestamp("2026-07-02"), "R32_15": pd.Timestamp("2026-07-02"),
    "R32_13": pd.Timestamp("2026-07-03"), "R32_16": pd.Timestamp("2026-07-03"), "R32_14": pd.Timestamp("2026-07-03"),
    # Round of 16
    "R16_01": pd.Timestamp("2026-07-04"), "R16_02": pd.Timestamp("2026-07-04"),
    "R16_05": pd.Timestamp("2026-07-05"), "R16_06": pd.Timestamp("2026-07-05"),
    "R16_03": pd.Timestamp("2026-07-06"), "R16_04": pd.Timestamp("2026-07-06"),
    "R16_07": pd.Timestamp("2026-07-07"), "R16_08": pd.Timestamp("2026-07-07"),
    # Quarter Finals
    "QF_01": pd.Timestamp("2026-07-09"), "QF_02": pd.Timestamp("2026-07-10"),
    "QF_03": pd.Timestamp("2026-07-11"), "QF_04": pd.Timestamp("2026-07-11"),
    # Semi Finals
    "SF_01": pd.Timestamp("2026-07-14"), "SF_02": pd.Timestamp("2026-07-15"),
    # Final Stage
    "THIRD_PLACE": pd.Timestamp("2026-07-18"), "FINAL": pd.Timestamp("2026-07-19"),
}

_KO_R16_TEAMS: dict[str, tuple[str, str]] = {
    "R16_01": ("Winner of R32-01", "Winner of R32-02"),
    "R16_02": ("Winner of R32-03", "Winner of R32-04"),
    "R16_03": ("Winner of R32-05", "Winner of R32-06"),
    "R16_04": ("Winner of R32-07", "Winner of R32-08"),
    "R16_05": ("Winner of R32-09", "Winner of R32-10"),
    "R16_06": ("Winner of R32-11", "Winner of R32-12"),
    "R16_07": ("Winner of R32-13", "Winner of R32-14"),
    "R16_08": ("Winner of R32-15", "Winner of R32-16"),
}

_KO_QF_TEAMS: dict[str, tuple[str, str]] = {
    "QF_01": ("Winner of R16-01", "Winner of R16-02"),
    "QF_02": ("Winner of R16-03", "Winner of R16-04"),
    "QF_03": ("Winner of R16-05", "Winner of R16-06"),
    "QF_04": ("Winner of R16-07", "Winner of R16-08"),
}

_KO_SF_TEAMS: dict[str, tuple[str, str]] = {
    "SF_01": ("Winner of QF-01", "Winner of QF-02"),
    "SF_02": ("Winner of QF-03", "Winner of QF-04"),
}

_KO_FINAL_TEAMS: dict[str, tuple[str, str]] = {
    "THIRD_PLACE": ("Loser of SF-01", "Loser of SF-02"),
    "FINAL":       ("Winner of SF-01", "Winner of SF-02"),
}

_KO_ROUND_OF_SLOT: dict[str, str] = {
    **{f"R32_{i:02d}": "R32" for i in range(1, 17)},
    **{f"R16_{i:02d}": "R16" for i in range(1, 9)},
    **{f"QF_{i:02d}": "QF" for i in range(1, 5)},
    "SF_01": "SF", "SF_02": "SF",
    "THIRD_PLACE": "FINAL_STAGE", "FINAL": "FINAL_STAGE",
}

_KO_STAGE_DISPLAY: dict[str, str] = {
    "R32": "Round of 32", "R16": "Round of 16",
    "QF": "Quarter Finals", "SF": "Semi Finals", "FINAL_STAGE": "Final Stage",
}


def _build_r32_teams_lookup(state: dict) -> dict[str, tuple[str, str]]:
    """Return {match_slot: (team_a, team_b)} for all 16 R32 matches based on current standings."""
    from src.tournament.build_knockout import build_round_of_32_fixtures

    standings = _build_full_standings(state["fixtures"])
    position_map = get_group_position_map(standings)
    r32_df = build_round_of_32_fixtures(standings, position_map)

    result: dict[str, tuple[str, str]] = {}
    for _, row in r32_df.iterrows():
        slot = row["match_slot"]
        ta = str(row["team_a"]) if pd.notna(row.get("team_a")) else row["team_a_slot"]
        tb = str(row["team_b"]) if pd.notna(row.get("team_b")) else row["team_b_slot"]
        result[slot] = (ta, tb)
    return result


def _ko_match_teams(slot: str, r32_lookup: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """Resolve any knockout slot to a (team_a_label, team_b_label) pair."""
    if slot in r32_lookup:
        return r32_lookup[slot]
    if slot in _KO_R16_TEAMS:
        return _KO_R16_TEAMS[slot]
    if slot in _KO_QF_TEAMS:
        return _KO_QF_TEAMS[slot]
    if slot in _KO_SF_TEAMS:
        return _KO_SF_TEAMS[slot]
    if slot in _KO_FINAL_TEAMS:
        return _KO_FINAL_TEAMS[slot]
    return ("TBD", "TBD")


# ---------------------------------------------------------------------------
# Knockout fixture display helper
# ---------------------------------------------------------------------------

def _render_knockout_fixture(slot: str, team_a: str, team_b: str, stage_label: str) -> None:
    """Render one knockout stage fixture card."""
    is_real = not team_a.startswith("W") and not team_a.startswith("L")
    fa = _flag(team_a) if is_real else "⚽"
    fb = _flag(team_b) if is_real else "⚽"

    with st.container():
        st.markdown(
            f"""<div style="border:1px solid #2d3142;border-radius:6px;padding:10px 14px;
            background:#1a1d27;margin-bottom:8px;">
            <div style="font-size:10px;color:#6b6b8a;margin-bottom:6px;text-transform:uppercase;
            letter-spacing:0.5px;">{stage_label} · {slot.replace('_','-')}</div>
            <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
              <div style="flex:1;text-align:right;font-size:14px;color:#d4d4d8;">{fa} {team_a}</div>
              <div style="font-size:13px;color:#6b6b8a;font-weight:600;padding:0 8px;">vs</div>
              <div style="flex:1;font-size:14px;color:#d4d4d8;">{fb} {team_b}</div>
            </div></div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Bracket tab
# ---------------------------------------------------------------------------

def _build_bracket_html(r32_teams: dict[str, tuple[str, str]], ko_results: dict | None = None) -> str:
    """Return HTML string for the full knockout bracket visualization."""
    CARD_W = 140
    CARD_H = 56
    COL_PITCH = 155
    UNIT = 130
    HDR_H = 24

    xs = [i * COL_PITCH for i in range(9)]  # 0,155,310,465,620,775,930,1085,1240
    TOTAL_W = xs[8] + CARD_W  # 1380
    TOTAL_H = HDR_H + 8 * UNIT  # 1064

    def _cy(slot_idx: int) -> int:
        return HDR_H + UNIT * slot_idx + UNIT // 2

    r32_cy = [_cy(i) for i in range(8)]
    r16_cy = [(r32_cy[i * 2] + r32_cy[i * 2 + 1]) // 2 for i in range(4)]
    qf_cy  = [(r16_cy[i * 2] + r16_cy[i * 2 + 1]) // 2 for i in range(2)]
    sf_cy  = (qf_cy[0] + qf_cy[1]) // 2

    LINE_C   = "#3d4166"
    CARD_BG  = "#1a1d27"
    BORD_C   = "#2d3142"
    TEXT_C   = "#d4d4d8"
    DIM_C    = "#6b6b8a"
    HDR_BG   = "#14162a"
    FIN_BG   = "#1e1a2e"
    FIN_BORD = "#5b4fcf"

    def _esc(s: str) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _card(x: int, center_y: int, ta: str, tb: str, label: str, is_final: bool = False) -> str:
        top  = center_y - CARD_H // 2
        bg   = FIN_BG   if is_final else CARD_BG
        bord = FIN_BORD if is_final else BORD_C
        return (
            f'<div style="position:absolute;left:{x}px;top:{top}px;width:{CARD_W}px;height:{CARD_H}px;'
            f'border:1px solid {bord};border-radius:4px;background:{bg};overflow:hidden;">'
            f'<div style="padding:1px 5px;height:21px;line-height:21px;font-size:10px;'
            f'color:{TEXT_C};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'border-bottom:1px solid #22243a;">{_esc(ta)}</div>'
            f'<div style="padding:1px 5px;height:21px;line-height:21px;font-size:10px;'
            f'color:{TEXT_C};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_esc(tb)}</div>'
            f'<div style="height:12px;line-height:12px;font-size:8px;color:{DIM_C};'
            f'background:{HDR_BG};padding:0 4px;text-align:center;">{_esc(label)}</div>'
            f'</div>'
        )

    def _hl(x: int, y: int, w: int) -> str:
        if w <= 0:
            return ""
        return (f'<div style="position:absolute;left:{x}px;top:{y}px;'
                f'width:{w}px;height:1px;background:{LINE_C};"></div>')

    def _vl(x: int, y1: int, y2: int) -> str:
        if y2 <= y1:
            return ""
        return (f'<div style="position:absolute;left:{x}px;top:{y1}px;'
                f'width:1px;height:{y2 - y1}px;background:{LINE_C};"></div>')

    def _lconn(col_x: int, y1: int, y2: int) -> str:
        """Left-to-right: pair of cards at col_x feeds into next column."""
        mid   = (y1 + y2) // 2
        re    = col_x + CARD_W
        vx    = re + 7
        nl    = col_x + COL_PITCH
        return _hl(re, y1, vx - re) + _hl(re, y2, vx - re) + _vl(vx, min(y1, y2), max(y1, y2)) + _hl(vx, mid, nl - vx)

    def _rconn(inner_x: int, outer_x: int, y1: int, y2: int) -> str:
        """Right-to-left: pair of cards at outer_x feeds into inner column."""
        mid   = (y1 + y2) // 2
        ir    = inner_x + CARD_W
        vx    = ir + 7
        return _hl(vx, y1, outer_x - vx) + _hl(vx, y2, outer_x - vx) + _vl(vx, min(y1, y2), max(y1, y2)) + _hl(ir, mid, vx - ir)

    _ko = ko_results or {}

    def _bteam(slot: str) -> tuple[str, str]:
        """Resolve bracket card teams for any slot, using known results where available."""
        ta, tb = _resolve_ko_match_teams(slot, _ko, r32_teams)
        def _fmt(t: str) -> str:
            if _is_real_team(t):
                return f"{_flag(t)} {t}"
            return t
        return _fmt(ta), _fmt(tb)

    parts: list[str] = []

    # Round header labels
    hdr_labels = ["Rd of 32", "Rd of 16", "Qtr Finals", "Semi Finals",
                  "🏆 FINAL", "Semi Finals", "Qtr Finals", "Rd of 16", "Rd of 32"]
    for i, hl in enumerate(hdr_labels):
        parts.append(
            f'<div style="position:absolute;left:{xs[i]}px;top:0;width:{CARD_W}px;height:{HDR_H}px;'
            f'text-align:center;font-size:9px;color:{DIM_C};line-height:{HDR_H}px;'
            f'text-transform:uppercase;letter-spacing:0.4px;">{hl}</div>'
        )

    # ── Left R32 ────────────────────────────────────────────────────────────
    for i, slot in enumerate(["R32_01","R32_02","R32_03","R32_04","R32_05","R32_06","R32_07","R32_08"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[0], r32_cy[i], ta_lbl, tb_lbl, f"M{int(slot[-2:])}"))
    for i in range(4):
        parts.append(_lconn(xs[0], r32_cy[i * 2], r32_cy[i * 2 + 1]))

    # ── Left R16 ────────────────────────────────────────────────────────────
    for i, slot in enumerate(["R16_01","R16_02","R16_03","R16_04"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[1], r16_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    for i in range(2):
        parts.append(_lconn(xs[1], r16_cy[i * 2], r16_cy[i * 2 + 1]))

    # ── Left QF ─────────────────────────────────────────────────────────────
    for i, slot in enumerate(["QF_01","QF_02"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[2], qf_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    parts.append(_lconn(xs[2], qf_cy[0], qf_cy[1]))

    # ── Left SF ─────────────────────────────────────────────────────────────
    sf1_ta, sf1_tb = _bteam("SF_01")
    parts.append(_card(xs[3], sf_cy, sf1_ta, sf1_tb, "SF-01"))
    parts.append(_hl(xs[3] + CARD_W, sf_cy, xs[4] - (xs[3] + CARD_W)))

    # ── Final ───────────────────────────────────────────────────────────────
    fin_ta, fin_tb = _bteam("FINAL")
    parts.append(_card(xs[4], sf_cy, fin_ta, fin_tb, "🏆 FINAL", is_final=True))
    parts.append(_hl(xs[4] + CARD_W, sf_cy, xs[5] - (xs[4] + CARD_W)))

    # ── Right SF ────────────────────────────────────────────────────────────
    sf2_ta, sf2_tb = _bteam("SF_02")
    parts.append(_card(xs[5], sf_cy, sf2_ta, sf2_tb, "SF-02"))
    parts.append(_rconn(xs[5], xs[6], qf_cy[0], qf_cy[1]))

    # ── Right QF ────────────────────────────────────────────────────────────
    for i, slot in enumerate(["QF_03","QF_04"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[6], qf_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    parts.append(_rconn(xs[6], xs[7], r16_cy[0], r16_cy[1]))
    parts.append(_rconn(xs[6], xs[7], r16_cy[2], r16_cy[3]))

    # ── Right R16 ───────────────────────────────────────────────────────────
    for i, slot in enumerate(["R16_05","R16_06","R16_07","R16_08"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[7], r16_cy[i], ta_lbl, tb_lbl, slot.replace("_","-")))
    for i in range(4):
        parts.append(_rconn(xs[7], xs[8], r32_cy[i * 2], r32_cy[i * 2 + 1]))

    # ── Right R32 ───────────────────────────────────────────────────────────
    for i, slot in enumerate(["R32_09","R32_10","R32_11","R32_12","R32_13","R32_14","R32_15","R32_16"]):
        ta_lbl, tb_lbl = _bteam(slot)
        parts.append(_card(xs[8], r32_cy[i], ta_lbl, tb_lbl, f"M{int(slot[-2:])}"))

    return (
        f'<div style="width:{TOTAL_W}px;height:{TOTAL_H}px;position:relative;'
        f'background:#0d0f1a;border-radius:8px;font-family:system-ui,sans-serif;">'
        + "".join(parts)
        + "</div>"
    )


def _show_bracket(state: dict) -> None:
    """Render the knockout bracket based on current group standings."""
    st.markdown("### 🏆 Knockout Bracket")
    st.caption("Round of 32 teams are based on current group standings. Later rounds update as results are entered.")

    # Lazy-load knockout results (may not be loaded if user skipped fixtures tab)
    if "ko_results" not in st.session_state:
        st.session_state.ko_results = _load_ko_results()
    ko_results = st.session_state.ko_results

    try:
        r32_teams = _build_r32_teams_lookup(state)
    except Exception as e:
        st.warning(f"Could not build bracket: {e}")
        return
    html = _build_bracket_html(r32_teams, ko_results=ko_results)
    st.markdown(
        f'<div style="overflow-x:auto;padding-bottom:8px;">{html}</div>',
        unsafe_allow_html=True,
    )

    # R32 slot legend below bracket
    st.divider()
    st.markdown("#### Round of 32 Matchups")
    slots_sorted = sorted(r32_teams.keys(), key=lambda s: int(s[-2:]))
    cols = st.columns(2)
    for idx, slot in enumerate(slots_sorted):
        ta, tb = r32_teams[slot]
        num = int(slot[-2:])
        with cols[idx % 2]:
            st.markdown(
                f"**M{num}** ({slot.replace('_','-')})  —  {_team_label(ta)} vs {_team_label(tb)}"
            )


# ---------------------------------------------------------------------------
# Group standings tab
# ---------------------------------------------------------------------------

def _build_full_standings(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Build standings for ALL groups, including teams with 0 games played."""
    completed = fixtures[fixtures["is_completed"]].copy()

    # Start from completed matches (may be empty)
    if not completed.empty:
        completed["goals_a"] = completed["goals_a"].astype(int)
        completed["goals_b"] = completed["goals_b"].astype(int)
        standings = build_group_standings(
            completed[["group", "team_a", "team_b", "goals_a", "goals_b"]]
        )
    else:
        standings = pd.DataFrame(
            columns=["group", "position", "team", "played", "wins", "draws",
                     "losses", "goals_for", "goals_against", "goal_diff", "points"]
        )

    # Collect every team from all fixtures
    all_teams = pd.concat([
        fixtures[["group", "team_a"]].rename(columns={"team_a": "team"}),
        fixtures[["group", "team_b"]].rename(columns={"team_b": "team"}),
    ]).drop_duplicates()

    played_teams = set(standings["team"]) if not standings.empty else set()
    zero_rows = []
    for _, row in all_teams.iterrows():
        if row["team"] not in played_teams:
            zero_rows.append({
                "group": row["group"], "team": row["team"],
                "position": 0, "played": 0, "wins": 0, "draws": 0,
                "losses": 0, "goals_for": 0, "goals_against": 0,
                "goal_diff": 0, "points": 0,
            })

    if zero_rows:
        standings = pd.concat(
            [standings, pd.DataFrame(zero_rows)], ignore_index=True
        )

    # Re-sort: points desc → goal_diff desc → goals_for desc → team asc
    standings = standings.sort_values(
        ["group", "points", "goal_diff", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    # Re-assign position within each group
    standings["position"] = standings.groupby("group").cumcount() + 1

    return standings


def _show_standings(state: dict) -> None:
    fixtures = state["fixtures"]
    standings = _build_full_standings(fixtures)
    completed = fixtures[fixtures["is_completed"]].copy()

    # Standings tables — always show all groups
    groups_all = sorted(standings["group"].unique())
    for i in range(0, len(groups_all), 3):
        cols = st.columns(3)
        for col, grp in zip(cols, groups_all[i : i + 3]):
            with col:
                st.markdown(f"**{grp}**")
                gdf = standings[standings["group"] == grp][
                    ["position", "team", "played", "wins", "draws",
                     "losses", "goals_for", "goals_against", "goal_diff", "points"]
                ].copy()
                gdf["team"] = gdf["team"].apply(_team_label)
                gdf["position"] = gdf["position"].apply(
                    lambda x: f"🟢 {x}" if x <= 2 else f"⚪ {x}"
                )
                st.dataframe(gdf, use_container_width=True, hide_index=True)

    st.divider()

    # Goals per team + ELO table (only meaningful once some games played)
    col_goals, col_elo = st.columns(2)
    # Goals table
    st.markdown("### Goals scored per team")

    if completed.empty:
        st.caption("No goals yet.")
    else:
        completed["goals_a"] = completed["goals_a"].astype(int)
        completed["goals_b"] = completed["goals_b"].astype(int)

        goals_a = completed.groupby("team_a")["goals_a"].sum().rename("goals")
        goals_b = completed.groupby("team_b")["goals_b"].sum().rename("goals")

        team_goals = (
            pd.concat([goals_a, goals_b])
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        team_goals.columns = ["Team", "Goals"]
        team_goals["Team"] = team_goals["Team"].apply(_team_label)

        st.dataframe(team_goals, use_container_width=True, hide_index=True)

    st.divider()

    # Full-width live ranking table
    st.markdown("### 🌍 Live World Cup Rankings")

    ranking_df = _world_cup_live_rankings(state)

    if ranking_df.empty:
        st.caption("No ranking data available.")
    else:
        display = ranking_df.copy()

        display["Rank"] = display["current_global_rank"].astype(int)
        display["Rank Change"] = display["rank_change"].apply(_format_rank_change)
        display["Team"] = display["team"].apply(_team_label)
        display["ELO"] = display["current_points"].round(1)
        display["ELO Change"] = display["elo_change"].apply(_format_elo_change)
        display["Start Rank"] = display["start_global_rank"].astype("Int64")
        display["Start ELO"] = display["start_points"].round(1)

        display = display[
            [
                "Rank",
                "Rank Change",
                "Team",
                "ELO",
                "ELO Change",
                "Start Rank",
                "Start ELO",
            ]
        ]

        st.dataframe(
            _style_rankings_table(display),
            use_container_width=True,
            hide_index=True,
            height=720,
        )

        st.caption(
            "Rank Change and ELO Change are compared to the start of the tournament."
        )

# ---------------------------------------------------------------------------
# Simulate forward tab
# ---------------------------------------------------------------------------

def _show_simulate_forward(
    state: dict,
    model,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
) -> None:
    unplayed = state["fixtures"][~state["fixtures"]["is_completed"]]
    n = len(unplayed)

    completed_count = int(state["fixtures"]["is_completed"].sum())
    total_count = len(state["fixtures"])
    st.markdown(
        f"**{completed_count}/{total_count}** group-stage matches have real results.  "
        f"**{n}** match(es) will be simulated using the v3 model."
    )

    if n == 0:
        st.success("All group stage matches have been played!")
    else:
        if st.button("🎲 Simulate remaining group matches", type="primary"):
            with st.spinner(f"Simulating {n} match(es)…"):
                st.session_state.sim_state = simulate_forward(
                    state, model, market_values, position_values,
                    feature_fn=feature_fn, score_fn=score_fn,
                )

    sim = st.session_state.get("sim_state")
    if sim is None:
        return

    st.warning("⚠️ SIMULATION — predicted scores, not actual results")

    sim_fixtures = sim["fixtures"]
    sim_completed = sim_fixtures[sim_fixtures["is_completed"]].copy()

    real_ids = set(
        state["fixtures"][state["fixtures"]["is_completed"]]["match_id"].tolist()
    )
    sim_only = sim_completed[~sim_completed["match_id"].isin(real_ids)].copy()

    if not sim_only.empty:
        st.markdown("### Simulated results")
        sim_only["goals_a"] = sim_only["goals_a"].astype(int)
        sim_only["goals_b"] = sim_only["goals_b"].astype(int)
        for _, fix in sim_only.sort_values("date").iterrows():
            ga, gb = int(fix["goals_a"]), int(fix["goals_b"])
            ta, tb = fix["team_a"], fix["team_b"]
            st.markdown(
                f"- {_team_label(ta)} **{ga}–{gb}** {_team_label(tb)}"
            )

    # Simulated standings
    st.markdown("### Simulated group standings")
    sim_completed["goals_a"] = sim_completed["goals_a"].astype(int)
    sim_completed["goals_b"] = sim_completed["goals_b"].astype(int)

    sim_standings = build_group_standings(
        sim_completed[["group", "team_a", "team_b", "goals_a", "goals_b"]]
    )
    groups_all = sorted(sim_standings["group"].unique())
    for i in range(0, len(groups_all), 3):
        cols = st.columns(3)
        for col, grp in zip(cols, groups_all[i : i + 3]):
            with col:
                st.markdown(f"**{grp}**")
                gdf = sim_standings[sim_standings["group"] == grp][
                    ["position", "team", "played", "wins", "draws", "losses",
                     "goals_for", "goals_against", "goal_diff", "points"]
                ].copy()
                gdf["team"] = gdf["team"].apply(_team_label)
                gdf["position"] = gdf["position"].apply(
                    lambda x: f"🟢 {x}" if x <= 2 else f"⚪ {x}"
                )
                st.dataframe(gdf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Team inspector tab
# ---------------------------------------------------------------------------

_FEATURE_LABELS: dict[str, str] = {
    "elo_diff": "ELO difference (A − B)",
    "rating_a_before": "ELO — Team A",
    "rating_b_before": "ELO — Team B",
    "rank_diff": "FIFA rank difference (A − B)",
    "form_diff_last5": "Form diff last 5 (pts avg)",
    "weighted_goals_for_diff_last5": "Weighted goals scored diff (last 5)",
    "weighted_goals_against_diff_last5": "Weighted goals conceded diff (last 5)",
    "opponent_strength_diff_last5": "Opponent ELO diff (last 5)",
    "rating_change_diff_last5": "ELO change diff (last 5)",
    "team_a_matches_played_before": "Career matches played — A",
    "team_b_matches_played_before": "Career matches played — B",
    "team_a_days_since_last_match": "Days since last match — A",
    "team_b_days_since_last_match": "Days since last match — B",
    "days_since_match_diff": "Rest diff (days, A − B)",
    "rest_diff": "Rest diff (days, A − B)",
    "tournament_points_diff": "Tournament points diff (A − B)",
    "tournament_goal_diff_diff": "Tournament GD diff (A − B)",
    "team_a_tournament_matches_played": "WC matches played — A",
    "team_b_tournament_matches_played": "WC matches played — B",
    "avg_player_value_diff": "Avg player value diff (€M)",
    "market_value_rel_mean_diff": "Market value vs year mean (diff)",
    "defender_share_diff": "Defender squad share (diff)",
    "goalkeeper_share_diff": "GK squad share (diff)",
    "competition_importance": "Competition importance (K-weight)",
}


def _last_n_games(team_canon: str, historical_matches: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the last N games for a team as a display DataFrame."""
    hist = historical_matches.copy()
    dates = pd.to_datetime(hist["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    hist["_date_naive"] = dates

    mask = (hist["team_a"] == team_canon) | (hist["team_b"] == team_canon)
    team_hist = hist[mask].sort_values("_date_naive").tail(n)

    rows = []
    for _, row in team_hist.iterrows():
        is_a = row["team_a"] == team_canon
        gf = int(row["goals_a"]) if is_a else int(row["goals_b"])
        ga = int(row["goals_b"]) if is_a else int(row["goals_a"])
        opponent = row["team_b"] if is_a else row["team_a"]
        elo_key = "rating_change_a" if is_a else "rating_change_b"
        elo_change = float(row.get(elo_key, 0) or 0)

        result = "W" if gf > ga else ("L" if gf < ga else "D")
        result_label = {"W": "✅ W", "D": "🤝 D", "L": "❌ L"}[result]

        rows.append({
            "Date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
            "Opponent": opponent,
            "Score": f"{gf}–{ga}",
            "Result": result_label,
            "ELO Δ": f"{elo_change:+.0f}",
            "Competition": str(row.get("competition", "")),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _show_team_inspector(
    state: dict,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    feature_fn=None,
) -> None:
    from src.features.build_features import build_pre_match_features
    from src.features.team_names import normalize_team_name

    if feature_fn is None:
        feature_fn = build_pre_match_features

    fixtures = state["fixtures"]
    all_teams = sorted(set(fixtures["team_a"]) | set(fixtures["team_b"]))

    sel_col, _ = st.columns([2, 3])
    with sel_col:
        team = st.selectbox("Select team", all_teams, key="team_inspector_selector")

    if not team:
        return

    team_canon = normalize_team_name(team)

    st.subheader(f"{_flag(team)} {team}")

    # --- Last 5 games ---
    st.markdown("#### Last 5 Games")

    games_df = _last_n_games(team_canon, state["historical_matches"])
    if games_df.empty and team != team_canon:
        games_df = _last_n_games(team, state["historical_matches"])

    if games_df.empty:
        st.caption("No match history found.")
    else:
        st.dataframe(games_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Current standing stats ---
    st.markdown("#### Current Snapshot")

    elo = state["elo_ratings"].get(team_canon, state["elo_ratings"].get(team))
    rank = state["rankings"].get(team_canon, state["rankings"].get(team))
    ts = state["team_states"].get(team_canon, state["team_states"].get(team, {}))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("ELO", f"{elo:.0f}" if elo is not None else "–")
    c2.metric("Global Rank", f"#{rank}" if rank is not None else "–")
    c3.metric("WC Pts", ts.get("points", 0))
    c4.metric("WC GD", f"{ts.get('goal_diff', 0):+d}")
    c5.metric("GF", ts.get("goals_for", 0))
    c6.metric("GA", ts.get("goals_against", 0))

    st.divider()

    # --- Features for next match ---
    unplayed = fixtures[~fixtures["is_completed"]].sort_values("date")
    next_fix_rows = unplayed[
        (unplayed["team_a"] == team) | (unplayed["team_b"] == team)
    ].head(1)

    if next_fix_rows.empty:
        st.caption("No upcoming group stage fixture.")
        return

    fix = next_fix_rows.iloc[0]
    opponent = fix["team_b"] if fix["team_a"] == team else fix["team_a"]
    match_date = pd.to_datetime(fix["date"])
    time_str = match_date.strftime("%b %d, %H:%M UTC")

    ta_canon = normalize_team_name(fix["team_a"])
    tb_canon = normalize_team_name(fix["team_b"])

    st.markdown(
        f"#### Features for Next Match: {_team_label(fix['team_a'])} vs {_team_label(fix['team_b'])}"
        f"  —  {time_str}"
    )
    st.caption(
        f"Features are computed with **{fix['team_a']}** as Team A and "
        f"**{fix['team_b']}** as Team B (fixture order). "
        f"Difference features = A − B."
    )

    try:
        feat_row = feature_fn(
            team_a=ta_canon,
            team_b=tb_canon,
            match_date=match_date,
            team_states=state["team_states"],
            historical_matches=state["historical_matches"],
            market_values=market_values,
            position_values=position_values,
            elo_ratings=state["elo_ratings"],
            rankings=state["rankings"],
        )

        feat_records = []
        for col in feat_row.columns:
            val = feat_row[col].iloc[0]
            feat_records.append({
                "Feature": col,
                "Description": _FEATURE_LABELS.get(col, ""),
                "Value": round(float(val), 4) if pd.notna(val) else None,
            })

        feat_display = pd.DataFrame(feat_records)
        st.dataframe(feat_display, use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"Could not compute features: {e}")


# ---------------------------------------------------------------------------
# Calibration status display
# ---------------------------------------------------------------------------

def _show_calibration_status(state: dict, use_calibration: bool = True) -> None:
    """Collapsible panel showing current Bayesian calibration factors."""
    from src.state.tournament_calibration import get_factors

    calibration = state.get("calibration")
    if calibration is None:
        return

    f = get_factors(calibration)
    n = f["n_games"]

    active_icon = "🎯" if use_calibration else "⏸️"
    active_note = "" if use_calibration else "  *(disabled — predictions use raw model)*"
    label = (
        f"{active_icon} Tournament Calibration  —  "
        f"goal scale **{f['goal_scale']:.3f}** · "
        f"draw adj **{f['draw_adj']:.3f}** · "
        f"{n} game{'s' if n != 1 else ''} calibrated"
        f"{active_note}"
    )

    with st.expander(label, expanded=False):
        if not use_calibration:
            st.warning(
                "Calibration is **disabled** — predictions are from the raw model without "
                "tournament adjustments. The calibration data is still being collected and "
                "will apply again when you re-enable it."
            )
        st.caption(
            "Bayesian adjustments applied to every prediction based on how this "
            "tournament is unfolding vs what the model expected.  "
            f"Prior fitted from WC 2006-2022 + Euro/Copa 2024 "
            f"({f['prior_n']} effective-game weight)."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Goal scale",
            f"{f['goal_scale']:.3f}",
            help="Multiplier on model λ. >1 = tournament is higher-scoring than model predicts.",
        )
        c2.metric(
            "Draw adjustment",
            f"{f['draw_adj']:.3f}",
            help="Multiplier on draw probability. >1 = more draws than model predicts.",
        )
        c3.metric(
            "Games calibrated",
            n,
            help="WC 2026 games where model predictions were stored before result submission.",
        )
        c4.metric(
            "Prior weight",
            f["prior_n"],
            help="How many historical games the prior counts for. Equal weight with observed at this many WC 2026 games.",
        )

        st.divider()

        col_goals, col_draws = st.columns(2)

        with col_goals:
            st.markdown("**Goals / game**")
            rows_g = [
                {"": "Prior (historical)", "Avg goals/game": f"{f['prior_goals']:.2f}"},
                {"": "Model predicted (this WC)", "Avg goals/game": f"{f['pred_goals_per_game']:.2f}" if f["pred_goals_per_game"] is not None else "–"},
                {"": "Actual (this WC)", "Avg goals/game": f"{f['obs_goals_per_game']:.2f}" if f["obs_goals_per_game"] is not None else "–"},
            ]
            st.dataframe(pd.DataFrame(rows_g), use_container_width=True, hide_index=True)

        with col_draws:
            st.markdown("**Draw rate**")
            rows_d = [
                {"": "Prior (historical)", "Draw rate": f"{f['prior_draw_rate']:.1%}"},
                {"": "Model predicted (this WC)", "Draw rate": f"{f['model_draw_rate']:.1%}"},
                {"": "Actual (this WC)", "Draw rate": f"{f['obs_draw_rate']:.1%}" if f["obs_draw_rate"] is not None else "–"},
                {"": "Posterior draw rate", "Draw rate": f"{f['posterior_draw_rate']:.1%}"},
            ]
            st.dataframe(pd.DataFrame(rows_d), use_container_width=True, hide_index=True)

        if n < 5:
            st.info(
                f"Only {n} game(s) calibrated so far — prior dominates. "
                "Calibration strengthens as more results are submitted."
            )


# ---------------------------------------------------------------------------
# Model accuracy tab
# ---------------------------------------------------------------------------

def _show_model_accuracy(state: dict) -> None:
    """Show per-model prediction accuracy for all completed WC 2026 matches."""
    completed = state["fixtures"][state["fixtures"]["is_completed"]]
    n_gs = int(len(completed))

    # Also count completed knockout matches
    ko_results: dict = st.session_state.get("ko_results", {})
    n_ko = len(ko_results)
    n_completed = n_gs + n_ko

    if n_completed == 0:
        st.info("No completed matches yet. Submit results to track model accuracy.")
        return

    accuracy_data = _load_model_accuracy()

    if not accuracy_data:
        st.info(
            "Model accuracy data is being computed in the background. "
            "Try clicking **🔄 Refresh from CSV** or reload the page."
        )
        return

    st.markdown(f"### Prediction Accuracy — {n_completed} match(es) completed")
    st.caption(
        "**Exact Score**: the model's predicted integer scoreline matches the actual result exactly.  "
        "**Correct Result**: the predicted outcome (Win / Draw / Loss) matches the actual outcome."
    )

    # Calibration factors for the calibrated rows
    calibration = state.get("calibration")
    cal_goal_scale = None
    if calibration is not None:
        from src.state.tournament_calibration import get_factors
        fac = get_factors(calibration)
        cal_goal_scale = fac["goal_scale"]

    score_fns = _get_score_fns()

    def _compute_calibrated_stats(records: list[dict], model_label: str) -> tuple[dict, list[dict]]:
        """Re-score records using current calibration goal_scale; return (stats, game_rows)."""
        sfn = score_fns.get(model_label)
        if sfn is None or cal_goal_scale is None:
            return None, []

        def _res(a, b):
            return "W" if a > b else ("L" if a < b else "D")

        cal_records = []
        for g in records:
            la = g.get("pred_la")
            lb = g.get("pred_lb")
            if la is None or lb is None:
                continue
            pa, pb = sfn(float(la) * cal_goal_scale, float(lb) * cal_goal_scale)
            aa, ab = int(g["actual_a"]), int(g["actual_b"])
            cal_records.append({
                "team_a": g["team_a"], "team_b": g["team_b"],
                "pa": pa, "pb": pb, "aa": aa, "ab": ab,
                "exact_ok": pa == aa and pb == ab,
                "result_ok": _res(pa, pb) == _res(aa, ab),
            })

        total = len(cal_records)
        if total == 0:
            return None, []

        exact = sum(1 for r in cal_records if r["exact_ok"])
        correct = sum(1 for r in cal_records if r["result_ok"])
        stats = {
            "total": total, "exact_correct": exact, "result_correct": correct,
            "exact_pct": exact / total * 100, "result_pct": correct / total * 100,
        }
        return stats, cal_records

    # Summary table — calibrated rows first, then uncalibrated
    summary_rows = []
    for model_label in ["V4", "V5", "V6"]:
        if model_label not in accuracy_data:
            continue
        records = accuracy_data[model_label]["records"]
        cal_stats, _ = _compute_calibrated_stats(records, model_label)
        if cal_stats is not None:
            s = cal_stats
            summary_rows.append({
                "Model": f"{model_label} + calibration",
                "Games": s["total"],
                "Exact Score": f"{s['exact_correct']} / {s['total']}",
                "Exact %": f"{s['exact_pct']:.1f}%",
                "Correct W/D/L": f"{s['result_correct']} / {s['total']}",
                "Result %": f"{s['result_pct']:.1f}%",
            })

    for model_label in ["V4", "V5", "V6"]:
        if model_label not in accuracy_data:
            continue
        s = accuracy_data[model_label]["stats"]
        summary_rows.append({
            "Model": model_label,
            "Games": s["total"],
            "Exact Score": f"{s['exact_correct']} / {s['total']}",
            "Exact %": f"{s['exact_pct']:.1f}%",
            "Correct W/D/L": f"{s['result_correct']} / {s['total']}",
            "Result %": f"{s['result_pct']:.1f}%",
        })

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No model accuracy data available yet.")
        return

    st.divider()
    st.markdown("#### Game by game")
    st.caption("🟢 Exact score correct · 🟠 Correct W/D/L · _(no color)_ Wrong result")

    def _res(a: int, b: int) -> str:
        return "W" if a > b else ("L" if a < b else "D")

    # Build per-match-id prediction lookup for all 6 configs
    pred_lookup: dict[str, dict[int, tuple]] = {}
    for model_label in ["V4", "V5", "V6"]:
        if model_label not in accuracy_data:
            continue
        records = accuracy_data[model_label]["records"]
        sfn = score_fns.get(model_label)

        pred_lookup[model_label] = {
            int(g["match_id"]): (int(g["pred_goals_a"]), int(g["pred_goals_b"]))
            for g in records
        }

        if cal_goal_scale is not None and sfn is not None:
            cal_preds = {}
            for g in records:
                la, lb = g.get("pred_la"), g.get("pred_lb")
                if la is not None and lb is not None:
                    pa, pb = sfn(float(la) * cal_goal_scale, float(lb) * cal_goal_scale)
                    cal_preds[int(g["match_id"])] = (pa, pb)
            pred_lookup[f"{model_label}+cal"] = cal_preds

    # Ordered columns (V4, V4+cal, V5, V5+cal, V6, V6+cal — only those with data)
    ordered_cols = [c for c in ["V4", "V4+cal", "V5", "V5+cal", "V6", "V6+cal"] if c in pred_lookup]

    # Collect all match IDs in order from the first available model
    all_match_ids: list[int] = []
    match_info: dict[int, dict] = {}
    for model_label in ["V4", "V5", "V6"]:
        if model_label not in accuracy_data:
            continue
        for g in accuracy_data[model_label]["records"]:
            mid = int(g["match_id"])
            if mid not in match_info:
                match_info[mid] = {
                    "team_a": g["team_a"], "team_b": g["team_b"],
                    "actual_a": int(g["actual_a"]), "actual_b": int(g["actual_b"]),
                }
                all_match_ids.append(mid)
        break

    rows = []
    color_rows = []
    for mid in all_match_ids:
        info = match_info[mid]
        aa, ab = info["actual_a"], info["actual_b"]
        row = {
            "Match": f"{_flag(info['team_a'])} {info['team_a']} vs {info['team_b']} {_flag(info['team_b'])}",
            "Actual": f"{aa}–{ab}",
        }
        color_row = {"Match": "", "Actual": ""}

        for col_label in ordered_cols:
            preds = pred_lookup.get(col_label, {})
            if mid in preds:
                pa, pb = preds[mid]
                row[col_label] = f"{pa}–{pb}"
                if pa == aa and pb == ab:
                    color_row[col_label] = "background-color: #16a34a; color: white"
                elif _res(pa, pb) == _res(aa, ab):
                    color_row[col_label] = "background-color: #fed7aa"
                else:
                    color_row[col_label] = ""
            else:
                row[col_label] = "–"
                color_row[col_label] = ""

        rows.append(row)
        color_rows.append(color_row)

    if rows:
        df_game = pd.DataFrame(rows)
        df_color = pd.DataFrame(color_rows)
        styled = df_game.style.apply(lambda _: df_color, axis=None)
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def show_live_tournament(
    model,
    fixtures: pd.DataFrame,
    historical_matches: pd.DataFrame,
    market_values: pd.DataFrame,
    position_values: pd.DataFrame,
    score_fn=None,
    feature_fn=None,
    model_name: str = "V4",
) -> None:
    """Render the full Live Tournament page."""
    _init_state(
        historical_matches, fixtures,
        model=model,
        market_values=market_values,
        position_values=position_values,
        feature_fn=feature_fn,
        score_fn=score_fn,
    )

    state = st.session_state.true_state

    st.header("⚽ WC 2026 — Live Tournament")

    # Header bar
    completed_count = int(state["fixtures"]["is_completed"].sum())
    total_count = len(state["fixtures"])
    hdr_col, cal_col, refresh_col, reset_col, clear_col = st.columns([3, 2, 1, 1, 1])

    with hdr_col:
        st.caption(f"{completed_count}/{total_count} group stage matches with real results")

    with cal_col:
        use_calibration = st.toggle(
            "🎯 Calibration",
            value=st.session_state.get("use_calibration", True),
            key="use_calibration",
            help=(
                "When ON: predictions are adjusted using Bayesian calibration from WC 2026 results so far. "
                "When OFF: raw model output is used directly. "
                "Calibration data is always collected regardless of this setting."
            ),
        )

    with refresh_col:
        if st.button("🔄 Refresh from CSV"):
            _refresh_from_csv(
                model=model,
                market_values=market_values,
                position_values=position_values,
                feature_fn=feature_fn,
                score_fn=score_fn,
            )
            st.rerun()

    with reset_col:
        if st.button("🧹 Reset state"):
            for key in ("true_state", "sim_state"):
                st.session_state.pop(key, None)
            st.rerun()

    with clear_col:
        if st.button("🗑️ Clear results"):
            clear_saved_results_csv()
            for key in ("true_state", "sim_state"):
                st.session_state.pop(key, None)
            st.rerun()

    _show_calibration_status(state, use_calibration=use_calibration)

    tab_gs, tab_standings, tab_bracket, tab_sim, tab_inspect, tab_accuracy = st.tabs(
        ["📅 Fixtures by Day", "📊 Group Standings", "🏆 Bracket", "🎲 Simulate Forward", "🔍 Team Inspector", "📈 Model Accuracy"]
    )

    with tab_gs:
        _show_group_stage(state, model, market_values, position_values, feature_fn=feature_fn, score_fn=score_fn, use_calibration=use_calibration, model_name=model_name)

    with tab_standings:
        _show_standings(state)

    with tab_bracket:
        _show_bracket(state)

    with tab_sim:
        _show_simulate_forward(state, model, market_values, position_values, score_fn=score_fn, feature_fn=feature_fn)

    with tab_inspect:
        _show_team_inspector(state, market_values, position_values, feature_fn=feature_fn)

    with tab_accuracy:
        _show_model_accuracy(state)
