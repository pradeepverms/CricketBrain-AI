"""
CricketBrain AI — Monte Carlo Simulation Engine
10,000+ match simulations | Full probability distributions | CI analysis
"""

import numpy as np
import pandas as pd
import os, json, warnings, joblib
warnings.filterwarnings("ignore")

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")

N_SIMULATIONS = 10_000

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
def load_model():
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def load_features():
    path = os.path.join(MODEL_DIR, "feature_names.pkl")
    if not os.path.exists(path):
        return []
    return joblib.load(path)

# ─────────────────────────────────────────────────────────
# SIMULATION CORE
# ─────────────────────────────────────────────────────────
def simulate_match(base_prob: float, n_sim: int = N_SIMULATIONS,
                   uncertainty: float = 0.12) -> dict:
    """
    Simulate a match n_sim times using a base win probability.
    Adds team-specific noise to model real-world variability.
    Returns full distribution stats.
    """
    np.random.seed(42)

    # Team 1 win probability with beta-distributed noise
    alpha = base_prob / uncertainty
    beta_ = (1 - base_prob) / uncertainty
    alpha = max(alpha, 0.5)
    beta_ = max(beta_, 0.5)

    # Per-simulation probabilities (uncertainty in the estimate itself)
    sim_probs = np.random.beta(alpha, beta_, n_sim)
    sim_probs = np.clip(sim_probs, 0.01, 0.99)

    # Simulate outcomes
    outcomes = np.random.binomial(1, sim_probs)
    team1_wins = outcomes.sum()

    # Simulate scores (empirical IPL distributions)
    t1_scores = simulate_scores(n_sim, mean=165, std=28, low=90, high=230)
    t2_scores = simulate_scores(n_sim, mean=161, std=27, low=88, high=225)

    ci_low  = np.percentile(sim_probs, 2.5)
    ci_high = np.percentile(sim_probs, 97.5)

    # Scenario breakdown
    dominant  = (sim_probs > 0.70).sum() / n_sim * 100
    contested = ((sim_probs >= 0.45) & (sim_probs <= 0.55)).sum() / n_sim * 100
    upset     = (sim_probs < 0.40).sum() / n_sim * 100

    # Score distribution
    t1_hist, edges = np.histogram(t1_scores, bins=20)
    t2_hist, _     = np.histogram(t2_scores, bins=edges)

    return {
        "n_simulations":    n_sim,
        "team1_win_pct":    round(team1_wins / n_sim * 100, 2),
        "team2_win_pct":    round((n_sim - team1_wins) / n_sim * 100, 2),
        "base_probability": round(base_prob, 4),
        "mean_sim_prob":    round(float(sim_probs.mean()), 4),
        "std_sim_prob":     round(float(sim_probs.std()), 4),
        "ci_95_low":        round(float(ci_low), 4),
        "ci_95_high":       round(float(ci_high), 4),
        "scenario_dominant_pct":  round(dominant, 2),
        "scenario_contested_pct": round(contested, 2),
        "scenario_upset_pct":     round(upset, 2),
        "team1_avg_score":   round(float(t1_scores.mean()), 1),
        "team2_avg_score":   round(float(t2_scores.mean()), 1),
        "team1_score_std":   round(float(t1_scores.std()), 1),
        "team2_score_std":   round(float(t2_scores.std()), 1),
        "team1_score_p10":   round(float(np.percentile(t1_scores, 10)), 1),
        "team1_score_p90":   round(float(np.percentile(t1_scores, 90)), 1),
        "team2_score_p10":   round(float(np.percentile(t2_scores, 10)), 1),
        "team2_score_p90":   round(float(np.percentile(t2_scores, 90)), 1),
        "score_hist_edges":  [round(e, 1) for e in edges.tolist()],
        "team1_score_hist":  t1_hist.tolist(),
        "team2_score_hist":  t2_hist.tolist(),
        "raw_probs":         sim_probs[:200].tolist(),  # sample for frontend
    }

def simulate_scores(n, mean=165, std=25, low=80, high=240):
    """Simulate IPL-realistic scores using truncated normal"""
    scores = np.random.normal(mean, std, n)
    return np.clip(scores, low, high).astype(int)

# ─────────────────────────────────────────────────────────
# PLAYER PERFORMANCE SIMULATION
# ─────────────────────────────────────────────────────────
def simulate_player_performance(player_avg_runs: float, player_sr: float,
                                 n_sim: int = N_SIMULATIONS) -> dict:
    """Simulate a batsman's innings across n_sim matches"""
    np.random.seed(42)
    # Balls faced distribution
    avg_balls = (player_avg_runs / max(player_sr, 1)) * 100
    balls_sim = np.random.poisson(avg_balls, n_sim)
    balls_sim = np.clip(balls_sim, 0, 120)
    # Runs from balls
    sr_sim = np.random.normal(player_sr, player_sr * 0.15, n_sim)
    sr_sim = np.clip(sr_sim, 30, 250)
    runs_sim = (balls_sim * sr_sim / 100).astype(int)

    return {
        "avg_runs_sim":   round(float(runs_sim.mean()), 1),
        "std_runs_sim":   round(float(runs_sim.std()), 1),
        "p25_runs":       round(float(np.percentile(runs_sim, 25)), 1),
        "p50_runs":       round(float(np.percentile(runs_sim, 50)), 1),
        "p75_runs":       round(float(np.percentile(runs_sim, 75)), 1),
        "prob_30plus":    round(float((runs_sim >= 30).mean() * 100), 1),
        "prob_50plus":    round(float((runs_sim >= 50).mean() * 100), 1),
        "prob_duck":      round(float((runs_sim == 0).mean() * 100), 1),
        "hist_runs":      np.histogram(runs_sim, bins=15)[0].tolist(),
        "hist_edges":     [round(e, 0) for e in np.histogram(runs_sim, bins=15)[1].tolist()],
    }

def simulate_bowler_performance(avg_economy: float, wicket_rate: float,
                                  n_sim: int = N_SIMULATIONS) -> dict:
    """Simulate a bowler's spell across n_sim matches"""
    np.random.seed(42)
    # Economy variation
    econ_sim = np.random.normal(avg_economy, avg_economy * 0.15, n_sim)
    econ_sim = np.clip(econ_sim, 3, 20)
    # Wickets (Poisson with expected λ)
    expected_wkts = wicket_rate * 4  # in ~4 overs
    wkts_sim = np.random.poisson(max(expected_wkts, 0.1), n_sim)
    wkts_sim = np.clip(wkts_sim, 0, 6)

    return {
        "avg_economy_sim":  round(float(econ_sim.mean()), 2),
        "std_economy_sim":  round(float(econ_sim.std()), 2),
        "avg_wickets_sim":  round(float(wkts_sim.mean()), 2),
        "prob_0_wkts":      round(float((wkts_sim == 0).mean() * 100), 1),
        "prob_1plus_wkts":  round(float((wkts_sim >= 1).mean() * 100), 1),
        "prob_2plus_wkts":  round(float((wkts_sim >= 2).mean() * 100), 1),
        "prob_3plus_wkts":  round(float((wkts_sim >= 3).mean() * 100), 1),
    }

# ─────────────────────────────────────────────────────────
# NATURAL LANGUAGE EXPLANATION
# ─────────────────────────────────────────────────────────
def explain_simulation(result: dict, team1: str, team2: str) -> str:
    t1_pct  = result["team1_win_pct"]
    ci_low  = round(result["ci_95_low"] * 100, 1)
    ci_high = round(result["ci_95_high"] * 100, 1)
    dominant  = result["scenario_dominant_pct"]
    contested = result["scenario_contested_pct"]
    upset     = result["scenario_upset_pct"]

    lines = [
        f"🎲 **CricketBrain AI simulated this match {result['n_simulations']:,} times.**",
        f"",
        f"📊 **{team1}** wins in **{t1_pct:.1f}%** of simulations "
        f"(95% CI: {ci_low}%–{ci_high}%)",
        f"📊 **{team2}** wins in **{result['team2_win_pct']:.1f}%** of simulations",
        f"",
        f"🔍 **Scenario Breakdown:**",
        f"• Dominant {team1} victory: {dominant:.1f}% of simulations",
        f"• Closely contested match: {contested:.1f}% of simulations",
        f"• Upset potential for {team2}: {upset:.1f}% of simulations",
        f"",
        f"🎯 **Predicted Scores:**",
        f"• {team1}: {result['team1_avg_score']} runs "
        f"(range: {result['team1_score_p10']}–{result['team1_score_p90']})",
        f"• {team2}: {result['team2_avg_score']} runs "
        f"(range: {result['team2_score_p10']}–{result['team2_score_p90']})",
    ]

    if t1_pct > 65:
        verdict = f"🏆 {team1} is the **strong favourite** based on current form and historical data."
    elif t1_pct > 55:
        verdict = f"🏆 {team1} holds a **moderate advantage**, but {team2} can upset."
    elif t1_pct > 45:
        verdict = f"⚖️ This is a **near coin-flip match** — both teams evenly matched."
    else:
        verdict = f"🏆 {team2} is the **favourite** in this contest."

    lines.append(f"\n{verdict}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────
def run_simulation(team1: str, team2: str, base_prob: float,
                   n_sim: int = N_SIMULATIONS) -> dict:
    result = simulate_match(base_prob, n_sim)
    result["team1"] = team1
    result["team2"] = team2
    result["explanation"] = explain_simulation(result, team1, team2)
    return result

if __name__ == "__main__":
    # Quick test
    r = run_simulation("Chennai Super Kings", "Mumbai Indians", 0.58)
    print(r["explanation"])
    print(f"\nCI: {r['ci_95_low']:.3f} – {r['ci_95_high']:.3f}")
