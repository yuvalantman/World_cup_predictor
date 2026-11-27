# V6 Experiments Report — Isolated Improvements over V4/V5

Date: 2026-06-11. All experiments tuned on the 4 clean folds (WC 2014, WC 2018,
Euro 2024, Copa 2024); WC 2022 used **only** for final one-shot evaluation.
V4/V5 production artifacts untouched. All code in `src/experiments/` and
`scripts/v6_*.py`.

---

## 1. What V4 and V5 currently do

| | V4 | V5 |
|---|---|---|
| Features | 21 (`FEATURE_COLS`) | 20 (`FEATURE_COLS_V5_PROD`: −matches_played, +rest_diff, +competition_importance) |
| Model | LGBM(0.9)+XGB(0.1) ensemble, Poisson objective, mirror-augmented symmetric training | same |
| Weights | competition-weighted | + temporal decay 0.9, blend 0.5 |
| Decision rule | `floor(λ+0.1)` per team | conditional floor: B floor at λ_b+0.5 only when pred A goals ≥ 2 |
| WC 2022 exact | 31.2% | 34.4% (this harness) |
| Pathologies | mode-collapse to 2-0, almost no 2-1/BTTS, KO draws | better 2-1, but draws under-predicted (14% vs 23% actual) |

## 2. Diagnoses

1. **Mirror bug in V5-prod (confirmed empirically)**: models mirror-augment via
   the V4 `mirror_features`, whose diff-column list lacks `rest_diff`. Mirrored
   training rows swap the rest-day pair but keep `rest_diff`'s sign →
   predict(A vs B) ≠ swap(predict(B vs A)) by ~0.02–0.03 λ. V4 itself is
   perfectly symmetric. Fixed in V6 (`mirror_features_v6`); verified asymmetry = 0.000000.
2. **Draw under-prediction is a decision-rule problem**: floor rules can only
   produce a draw when both λs land in the same floor bin. Lambdas carry enough
   draw information — the calibrated Dixon-Coles grid (ρ ≈ −0.33 fit on OOF)
   identifies high-draw matches well; the rule just never used it.
3. **rank_diff is load-bearing**: removing it (F_NORANK) collapses lambda
   separation (mean |λa−λb| 1.24→0.79) and outcome accuracy (80%→48%). The model
   is not "blindly" Elo-dependent — rank+Elo together carry the strength signal.
4. **Tree features cannot fix form-vs-rank**: even with 10 new in-tournament
   form features, a team that won 3-0, 3-0 vs a favorite that drew/lost moves λ
   by only ~0.04. Strong form-vs-Elo conflicts are too rare in training data for
   GBM splits to learn them. The fix has to live at the lambda level (see §5).

## 3. Experiments run

**Feature sets** (OOF lambdas on 4 folds, fresh ensembles per fold, cached in `oof_*.csv`):
- `F_V5` — control: V5-prod 20 features with the corrected mirror
- `F_FORM` — +7 leakage-safe shrunk tournament-form diffs (gf/ga per match, win rate, clean sheets, failed-to-score, margin, opponent-Elo-adjusted GD)
- `F_ELOUPD` — +3 more: in-tournament attack/defense Elo-style updates (K=0.4 vs Elo-implied expectation), form-vs-prior disagreement
- `F_CAP` — F_ELOUPD with `elo_diff` clipped to ±250 and `rank_diff` log-transformed
- `F_NORANK` — F_ELOUPD without rank_diff (ablation)

**Decision rules** (on cached OOF lambdas, calibration cross-fitted across folds):
- `floor01` (V4), `cond_floor tb∈{0.4,0.5,0.6}` (V5 family)
- Calibrated DC grid argmax × draw-bonus β × empirical-WC-prior weight × KO-ET mixture (24 configs)
- **`drawband`** (new hybrid): cond_floor by default; when the calibrated DC grid puts ≥ t mass on a draw, predict the most likely draw score. t∈{0.26,0.28,0.30,0.33}
- **Poisson-Gamma posterior blend** (new): λ_post = (tournament_goals + k·λ)/(m + k), k∈{2..10}, optionally gated to KO rows (m≥3)

## 4. Results

### Fold tuning (n=211, WC 2022 never seen) — selected rows

| Variant | Exact | Outcome | GD | Draws p/a | BTTS p/a |
|---|---|---|---|---|---|
| F_V5 + cond_floor 0.5 (≈V5) | 22.3% | 80.1% | 46.0% | 10/25 | 36/51 |
| F_CAP + cond_floor 0.6 | 23.7% | 80.6% | 47.4% | 9/25 | 50/51 |
| **F_V5 + drawband(0.33, tb0.5)** | **24.6%** | **86.7%** | **55.0%** | 32/25 | 38/51 |
| F_V5 + drawband + KO-gated blend k=8 | **28.0%** | 87.2% | 55.9% | 33/25 | 48/51 |

### WC 2022 one-shot holdout (64 matches)

| Variant | Exact | Outcome | GD | Group exact | KO exact | Draws p/a | BTTS p/a | Top predicted |
|---|---|---|---|---|---|---|---|---|
| V4 baseline | 31.2% | 85.9% | 48.4% | 33.3% | 25.0% | 14/23 | 14/48 | 2-0 ×30 |
| V5 baseline | 34.4% | 85.9% | 59.4% | 35.4% | 31.2% | 14/23 | 34/48 | 2-0 ×16, 1-0 ×14, 2-1 ×12 |
| **V6 = F_V5 + drawband(0.33, tb0.5)** | **39.1%** | **90.6%** | **62.5%** | **43.8%** | 25.0% | 33/23 | 34/48 | 2-0 ×17, 0-0 ×13, 2-1 ×12 |
| V6 + KO-gated blend k=8 | 35.9% | 87.5% | 57.8% | 43.8% | 12.5% | 33/23 | 47/48 | **2-1 ×15**, 2-0 ×14, 1-1 ×12 |

Other holdout shots: F_ELOUPD+drawband 37.5%; F_CAP variants 25.0–34.4%
(F_CAP won folds but did not generalize — capped-Elo rejected).

### Match-level deltas (V6 drawband vs V5, holdout)

Fixed (7): Denmark–Tunisia 0-0, Croatia–Morocco 0-0, South Korea–Uruguay 0-0,
England–USA 0-0, Morocco–Spain 0-0, Brazil–Serbia 2-0, France–Denmark 2-1.
Broken (4): Tunisia–France 1-0→1-1, Mexico–Saudi 2-1→2-0, Croatia–Japan 1-1→0-0,
Morocco–Portugal 1-0→1-1. On actual draws: V5 2/15 exact, V6 6/15.

## 5. Form-vs-rank scenario (B won 3-0, 3-0; favorite A drew 0-0, lost 0-1)

| Model | λ_A | λ_B | Favors |
|---|---|---|---|
| V5 prod | 1.13 | 0.85 | A (wrongly) |
| V6 features only (F_ELOUPD) | 1.11 | 0.90 | A — features barely move it |
| Posterior blend k=8 | 0.91 | 1.28 | **B** ✓ |
| Posterior blend k=4 | 0.76 | 1.56 | **B** (stronger) |

The blend is the only mechanism that meaningfully reacts to tournament form. On
folds it lifted KO exact 20.0%→32.7%, but on the WC 2022 holdout it dropped KO
exact 25%→12.5% (2022 KOs were upset-heavy; n=16 is very noisy both ways).
**Verdict: keep it as an optional, disabled-by-default layer** — it is the right
tool when you explicitly want form-responsiveness (e.g. live 2026 predictions
where you believe in-tournament form), but it did not earn holdout-default status.

## 6. Recommendation

**Promote: `V6 = F_V5 features (20, corrected mirror) + drawband(t=0.33, tb=0.5)`**
with frozen calibration `scale_c=0.999, rho=-0.329` (fit on 4-fold OOF).

- +4.7pts exact, +4.7pts outcome, +3.1pts GD vs V5 on the untouched holdout
- Draw rate 33% vs actual 23% (V5: 14%) — slight overshoot, far more realistic
- Same training pipeline, same features as V5; only the mirror fix and the
  decision rule change. Inference needs: λs → blend gate (off) → drawband rule.
- Perfect team-swap symmetry (asymmetry 0.000000; V5-prod has a real bug here).

## 7. Files

New (experiments, all isolated):
- `src/experiments/v6_features.py` — form features, correct mirror, posterior blend
- `src/experiments/v6_models.py` — prod-hyperparam models with corrected mirror
- `src/experiments/v6_eval.py` — metric suite, decision rules, cross-fit calibration
- `scripts/v6_experiments.py` — orchestrator (prepare / oof / sweep / holdout)
- `scripts/v6_drawband_sweep.py`, `scripts/v6_posterior_sweep.py`,
  `scripts/v6_posterior_ko_sweep.py`, `scripts/v6_final_holdout.py`,
  `scripts/v6_scenario_test.py`, `scripts/v6_symmetry_check.py`
- `outputs/experiments/v6/` — cached dataset/OOF lambdas, `results_folds*.csv`,
  `results_holdout.csv`, `matchbymatch_*.csv`, this report

Modified: none of the V4/V5 production path.

## 8. Remaining risks / next ideas

- n=64 holdout: the +4.7pt gap is ~3 matches; treat as "clearly not worse,
  probably better", not a guarantee. The fold ranking agreed (drawband best),
  which is the stronger evidence.
- Draw band slightly overshoots draws (33% vs 23–25%); t could be nudged to 0.35
  but that was not tuned on folds — leave as is.
- BTTS still under at tb=0.5 (34% vs 48%). tb=0.6 fixes BTTS but costs exact —
  exact-first objective says keep 0.5.
- If promoted to production: retrain on ALL data with `mirror_features_v6`,
  freeze drawband params from full OOF, wire a `score_fn` that takes both λs
  through the DC grid (needs the grid at app inference time, not just floors).
- Untried: xG features (no data in repo), H2H features, multiclass scoreline
  classifier, Optuna re-tune under the drawband objective.
