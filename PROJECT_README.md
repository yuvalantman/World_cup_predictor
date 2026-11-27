# World Cup Match Predictor

## Overview

This project is an end-to-end machine learning system that predicts football match scores and outcomes, purpose-built and tuned for the FIFA World Cup. It goes beyond a single trained model: it's a full pipeline covering data ingestion, feature engineering from dynamic team state, model training and evaluation, tournament simulation, and a live-updating web app that tracks a real tournament (World Cup 2026) as it happens.

The core problem is harder than standard supervised learning because team strength isn't static — it evolves match by match. A team's rating, form, and context change *during* the tournament being predicted. The system was designed around that: team state (Elo ratings, rolling form, standings) is a first-class, continuously updated object that feeds the model, rather than a fixed snapshot computed once at training time.

## Why this approach

- **Score prediction, not just win/draw/loss.** Predicting exact goals (via goal-rate models) is strictly harder than classification, but it's what lets the system also derive outcome probabilities, expected goal difference, and full tournament simulations (group standings, bracket progression) for free.
- **Dynamic team state instead of static features.** Elo ratings and form windows are recomputed as results come in, so the model's inputs reflect the tournament's actual trajectory instead of pre-tournament assumptions.
- **Chronological evaluation only.** All backtesting is done by holding out entire World Cups in time order (e.g., train on everything before 2022, test on 2022) to avoid leakage — a model that "predicts the past" is not useful for a live tournament.
- **Iterative model versioning (v1 → v6).** Each version isolates one hypothesis (which features matter, which model family, how to handle draws) and is evaluated against the same held-out World Cup before being promoted to production, so improvements are attributable rather than accidental.

## System architecture

### High level

```
Historical match data + Elo/rankings + squad market value
          │
          ▼
  Feature engineering (team state, form, rank/rating diffs, tournament context)
          │
          ▼
  Model zoo (Poisson regression / gradient-boosted trees / ensembles)
          │
          ▼
  Score distribution → outcome probabilities → tournament simulation
          │
          ▼
  Live tournament state (Elo updates, standings, bracket) ── Streamlit app
```

### Low level — key modules

- **`src/data/`** — loaders and validators for historical results, Elo/ranking sources, and fixture lists; establishes a consistent schema across data vintages.
- **`src/features/`** — feature construction: pre-tournament priors, rolling recent-form windows, market-value and squad-position indicators, match context (home advantage, tournament stage), and rank/rating differentials between opposing teams.
- **`src/models/`** — the model zoo: a Poisson goal-rate model, gradient-boosted tree models (LightGBM/XGBoost), a baseline model, an ensembling layer, hyperparameter tuning (Optuna), and score-grid/score-conversion utilities that turn predicted goal rates into full scorelines and outcome probabilities.
- **`src/state/`** — the dynamic team-state engine: an Elo rating updater (K-factor by competition tier, goal-margin multiplier, home advantage), group-table and standings logic, and the live-state object that gets mutated (via defined transitions, never in place) as real results come in.
- **`src/tournament/`** — bracket construction and knockout-stage simulation logic, used to project a full tournament forward from any point.
- **`src/evaluation/`** — chronological backtesting harness, metrics (exact-score accuracy, outcome accuracy, goal MAE/RMSE), and evaluation reports comparing model versions on identical held-out tournaments.
- **`src/experiments/`** — isolated, reproducible experiment code (feature sets, alternate scoring rules) used to validate a hypothesis before it's folded into a production model version.
- **`src/app/`** — a Streamlit application with two surfaces: a what-if tournament simulator, and a live tournament tracker that ingests real match results, updates ratings/standings, and simulates the remainder of the tournament forward from the live state.

## Development process

1. **Baseline.** Started with a simple Poisson goal-rate model using static pre-tournament features (rankings, market value) to establish a reference point.
2. **Feature iteration.** Progressively added dynamic signals — recent form, live Elo, rank/rating differentials — and measured each addition's effect via chronological backtests, discarding features that didn't survive out-of-time validation.
3. **Model family comparison.** Evaluated tree-based models and ensembles alongside the Poisson baseline to see whether nonlinear feature interactions (e.g., form vs. rank conflicts) improved predictions; kept components only when they generalized to held-out tournaments, not just training folds.
4. **Draw-handling refinement.** Because goal-rate models underrepresent draws, added a calibrated draw-correction rule (Dixon-Coles-style adjustment) that intervenes when the model's own draw-mass estimate signals a likely draw.
5. **Bug-hunting via mirrored data.** Used team-swapped ("mirrored") training data as a symmetry check — if swapping which team is "home" in the data changes the prediction asymmetrically, that flags a real feature-engineering bug. This caught and fixed an inconsistency in how a feature was constructed for the production model at the time.
6. **Versioned promotion.** Each model iteration (v1 through v6) was only promoted to "production" after outperforming the current production model on the same untouched holdout tournament, with configs and artifacts versioned alongside the model.
7. **From offline to live.** Extended the offline backtest/simulation pipeline into a live system: an Elo/state engine that ingests real results as they're reported, updates ratings and standings incrementally, and re-simulates the rest of the tournament — exposed through a Streamlit app for interactive use during the actual 2026 World Cup.

## Models used

- **Poisson regression** on team-level goal rates (statistical baseline, interpretable, natural fit for scorelines).
- **Gradient-boosted trees** (LightGBM / XGBoost) for nonlinear feature interactions.
- **Ensembling** of the above to combine statistical and tree-based signal.
- **Elo rating system** (custom implementation with competition-tier K-factors and goal-margin scaling) as the core dynamic team-strength signal.
- **Dixon-Coles-style draw correction** applied at the score-distribution level to correct for goal-model underestimation of draws.

## Tech stack

Python, pandas/numpy, scikit-learn, LightGBM, XGBoost, Optuna (hyperparameter tuning), Streamlit (live app), pytest (testing).
