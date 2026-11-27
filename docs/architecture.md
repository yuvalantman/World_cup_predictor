# Architecture

## Data flow

1. **Raw inputs** (`data/raw/`): yearly Elo result files, FIFA ranking, historical results, Transfermarkt squad and position values, 2026 fixtures, live match-result updates.
2. **Dataset build** (notebooks `05_build_model_dataset`, `10_append_new_matches`): produces `data/processed/model_dataset.csv` and `updated_model_dataset.csv` (about 21.7k matches, 2004 to 2026-06-08).
3. **Feature construction** (`src/features/`): one pre-match row per fixture. Column groups include Elo and rank differences, market and position value shares, opponent-adjusted rolling form over the last five matches, rest days, competition importance and in-tournament points and goal difference. `feature_columns.py` holds the feature lists for V4 (21), and V5/V6 (20).
4. **Training** (`src/models/`, `scripts/train_v5.py`, `scripts/train_v6.py`): LightGBM and XGBoost regressors with a Poisson objective, mirror-augmented rows, competition and temporal-decay sample weights, ensemble weights 0.9 / 0.1. Artifacts and configs are saved under `models/` per version.
5. **Scoreline selection** (`src/models/score_conversion.py`, `score_grid.py`): converts the two goal rates into a score and W/D/L probabilities.
   - V4: floor(λ + 0.1) per team.
   - V5: conditional floor for team B.
   - V6: V5 rule, but predict the most likely draw when the calibrated Dixon-Coles grid has draw mass of at least 0.33.
6. **Live state** (`src/state/`): `initialize_live_state`, `record_match_result`, `apply_results_from_csv` and `simulate_forward` maintain Elo, rankings, team windows and group tables. State is copied before simulation, never mutated in place.
7. **Tournament logic** (`src/tournament/`): group standings, third-place combinations, Round of 32 bracket, knockout simulation and Monte Carlo.
8. **App** (`src/app/`): `streamlit_app.py` with pages for single-match prediction, the 2026 simulation and the live tournament view. The model version is selected in the sidebar.

## Model versions

| Version | Features | Decision rule | Artifact |
|---|---|---|---|
| V4 | 21 | floor(λ + 0.1) | `models/production_model_v4.joblib` |
| V5 | 20 | conditional floor | `models/production_model_v5.joblib` |
| V6 | 20 (corrected mirror) | drawband | `models/production_model_v6.joblib`, `production_config_v6.json` |

Earlier artifacts (`production_model.joblib`, `_v2`, `_v3`) are kept unchanged for reproducibility. `production_model_v5.joblib` is preserved as trained, including a known mirror-feature inconsistency that V6 corrects.

## Elo update

`Rn = Ro + K · G · (W − We)` with `We = 1 / (10^(−dr/400) + 1)` and a home-advantage offset in `dr`. K is 60 for World Cup matches, 50 for continental tournaments, 40 for qualifiers, 20 for friendlies and 30 otherwise. G is 1.0 for a one-goal margin, 1.5 for two goals and (11 + diff) / 8 for three or more.

## Tests

`tests/` covers feature leakage, metrics, score conversion and grid behaviour, state updates, model smoke tests and the pipeline. They were not run during the documentation update.
