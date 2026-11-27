# Model Evaluation

## Protocol

- **Tuning folds:** WC 2014, WC 2018, Euro 2024, Copa America 2024 (211 matches). Each fold is trained on all other data except WC 2022.
- **Holdout:** WC 2022 (64 matches), evaluated once per final candidate and logged.
- **Calibration:** scale and Dixon-Coles rho are fit on out-of-fold goal rates and cross-fitted across folds. V6 uses `scale_c = 0.999`, `rho = -0.329`.
- **Metrics:** raw unweighted exact-score accuracy, outcome accuracy, goal-difference accuracy, predicted vs actual draw share, both-teams-to-score share and RPS.

Detailed results and the full experiment log are in [`outputs/experiments/v6/V6_REPORT.md`](../outputs/experiments/v6/V6_REPORT.md), with per-fold and per-match files in the same directory. The design of V5 is in [PLAN_V5.md](PLAN_V5.md).

## WC 2022 holdout

| Model | Exact | Outcome | Goal diff | Predicted / actual draws |
|---|---|---|---|---|
| V4 | 31.2% | 85.9% | 48.4% | 14 / 23 |
| V5 | 34.4% | 85.9% | 59.4% | 14 / 23 |
| V6 | 39.1% | 90.6% | 62.5% | 33 / 23 |

On the 64-match holdout the V6 gain over V5 is about three matches; the fold ranking (drawband best) is the stronger evidence.

## Tuning folds (211 matches)

| Variant | Exact | Outcome |
|---|---|---|
| V5-like (conditional floor) | 22.3% | 80.1% |
| V6 drawband | 24.6% | 86.7% |
| V6 + knockout-gated posterior blend (k = 8) | 28.0% | 87.2% |

## Live 2026 predictions (100 matches)

Per-match predicted and actual scores are stored in `data/raw/world_cup_updates/model_accuracy_v4.csv`, `_v5.csv` and `_v6.csv`; group matches have ids below 10000, knockout matches 10000 and above.

| Model | All (exact / outcome) | Group, 72 (exact / outcome) | Knockout, 28 (exact / outcome) |
|---|---|---|---|
| V4 | 10.0% / 64.0% | 12.5% / 58.3% | 3.6% / 78.6% |
| V5 | 9.0% / 64.0% | 11.1% / 58.3% | 3.6% / 78.6% |
| V6 | 9.0% / 58.0% | 9.7% / 55.6% | 7.1% / 64.3% |

Computed directly from the CSVs: a prediction counts as correct when the predicted scoreline matches exactly, or when the predicted result (win, draw, loss) matches. These are tracking numbers from a small sample, and the 2026 ordering is different from the holdout ordering. Likely contributors are sample size, an unusually high-scoring tournament with many large margins, and V6's draw-heavy decision rule; none of these was tested here.

## Known issues

- Draw predictions overshoot on the holdout (33% vs 23%).
- The posterior form blend lifted tuning-fold knockout exact accuracy (20.0% to 32.7%) but lowered holdout knockout exact accuracy (25% to 12.5%); it is disabled by default.
- Removing `rank_diff` collapses outcome accuracy (80% to 48%), so rank and Elo jointly carry the strength signal.
- A capped-Elo feature set won on folds but failed on the holdout and was rejected.
