// Mirrors backend/survivor/export/json_bundle.py's output shapes exactly.
// Hand-kept in sync rather than generated from a shared schema (a
// deliberate scope cut) -- `make export` regenerates the JSON files, so a
// shape mismatch surfaces at review/build time rather than silently.

export type Method = "vegas" | "montecarlo" | "blend" | "ml" | "blend3";

export interface SeasonMeta {
  season: number;
  current_week: number;
  generated_at: string;
  primary_method: Method;
  tie_counts_as_loss: boolean;
  used_teams: string[];
}

export interface BoardGame {
  game_id: string;
  home_team: string;
  away_team: string;
  gameday: string;
  spread_line: number | null;
  // Null when no line has been posted for this game yet — common for
  // late-season weeks. Not a 50/50 prediction; an absence of one.
  vegas_home_win_prob: number | null;
  vegas_away_win_prob: number | null;
  vegas_tie_prob: number | null;
  mc_home_win_prob: number | null;
  mc_away_win_prob: number | null;
  mc_tie_prob: number | null;
  mc_simulated_spread: number | null;
  // Method 3. Null when the model couldn't be fit for this week (e.g. the
  // pipeline ran without play-by-play reaching back to its training floor).
  ml_home_win_prob: number | null;
  ml_away_win_prob: number | null;
  ml_tie_prob: number | null;
}

export interface WeekBoard {
  season: number;
  week: number;
  games: BoardGame[];
  recommended_pick: string | null;
  cost_of_deviation: Record<string, number>;
  // How many of this week's games have a posted line yet, and therefore
  // whether the recommendation is market-driven or model-driven. Optional:
  // bundles exported before these existed simply won't carry them.
  n_games_with_line?: number;
  recommendation_source?: "vegas" | "mixed" | "model";
}

export interface WeekPick {
  week: number;
  team: string;
  survival_prob: number;
}

export interface SeasonPlanAlternate {
  total_survival_prob: number;
  picks: WeekPick[];
}

export interface SeasonPlan {
  season: number;
  current_week: number;
  recommended_sequence: WeekPick[];
  total_survival_prob: number | null;
  alternates: SeasonPlanAlternate[];
}

export interface ReliabilityBin {
  bin_low: number;
  bin_high: number;
  n: number;
  mean_predicted: number;
  empirical_rate: number;
}

export interface ConfidenceSlice {
  threshold_low: number;
  threshold_high: number;
  n: number;
  brier: number | null;
  log_loss: number | null;
  empirical_win_rate: number | null;
  mean_predicted: number | null;
}

export interface WeekBucketMetrics {
  n: number;
  brier: number;
  log_loss: number;
}

export interface MethodCalibration {
  brier: number;
  log_loss: number;
  reliability: ReliabilityBin[];
  high_confidence_slice: ConfidenceSlice;
  by_week_bucket: Record<string, WeekBucketMetrics>;
}

export interface BootstrapComparison {
  point_estimate: number;
  ci_low: number;
  ci_high: number;
  mc_significantly_better?: boolean;
  vegas_significantly_better?: boolean;
  blend_significantly_better?: boolean;
  ml_significantly_better?: boolean;
}

export interface BlendCalibration {
  n_games: number;
  brier: number;
  log_loss: number;
  vegas_on_same_subset_brier: number;
  blend_vs_vegas_bootstrap: BootstrapComparison;
}

export interface Calibration {
  n_games: number;
  vegas: MethodCalibration;
  montecarlo: MethodCalibration;
  mc_vs_vegas_brier_skill_score: number;
  mc_vs_vegas_bootstrap: BootstrapComparison;
  // Method 3 -- optional because a bundle exported before it existed, or
  // without play-by-play deep enough to fit it, simply won't carry these.
  ml?: MethodCalibration;
  ml_vs_vegas_brier_skill_score?: number;
  ml_vs_vegas_bootstrap?: BootstrapComparison;
  // The comparison that decides the season plan: future weeks have no Vegas
  // line, so Method 3's real competition is Monte Carlo.
  ml_vs_mc_bootstrap?: BootstrapComparison;
  blend?: BlendCalibration;
}

export interface BacktestMethodSummary {
  method: string;
  cheating: boolean;
  median_weeks_survived: number;
  mean_weeks_survived: number;
  p_survive_to_week10: number;
  p_survive_full_season: number;
  n_seasons: number;
}

export interface BacktestSeasonRow {
  season: number;
  method: string;
  cheating: boolean;
  n_picks: number;
  week_eliminated: number | null;
  survived_full_season: boolean;
}

export interface Backtest {
  summary: Record<string, BacktestMethodSummary>;
  seasons: BacktestSeasonRow[];
}

export interface StrategyOutcomeSummary {
  median_weeks_survived: number;
  mean_weeks_survived: number;
  p_survive_full_run: number;
  survive_to_week_curve: Record<string, number>;
}

export type Simulation = Record<string, StrategyOutcomeSummary>;

// --- survival_matrix.json: the what-if planner's only pipeline input ---
//
// The full (week, team) -> survival probability table for every remaining
// week, exported *unfiltered* (used teams included). Availability is applied
// in the browser from localStorage, so the planner is never stale relative to
// whatever USED= happened to be passed to `make export`.

export interface SurvivalEntry {
  week: number;
  team: string;
  opponent: string;
  is_home: boolean;
  game_id: string;
  survival_prob: number;
  weeks_ahead: number;
  /** "vegas" for the current week's posted lines, "model" for the MC+ML blend. */
  source: "vegas" | "model";
}

export interface SurvivalMatrix {
  season: number;
  current_week: number;
  /** Horizon decay already applied to `survival_prob`; carried for display. */
  decay_rate: number;
  tie_counts_as_loss: boolean;
  weeks: number[];
  teams: string[];
  /** A (week, team) pair absent from this list is unavailable, not zero. */
  entries: SurvivalEntry[];
}

// --- localStorage-persisted user state (not part of the pipeline export) ---

export interface UserPick {
  season: number;
  week: number;
  team: string;
  outcome: "pending" | "win" | "loss" | "tie";
}

export interface UserState {
  version: 1;
  picks: UserPick[];
}

/** A what-if commitment: "whatever else happens, I take this team this week." */
export interface ScenarioLock {
  season: number;
  week: number;
  team: string;
}

export interface ScenarioState {
  version: 1;
  locks: ScenarioLock[];
}
