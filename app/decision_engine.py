"""
CricketBrain AI — Decision Intelligence Engine (Phase 4)
Every decision includes: Evidence + Simulation + Confidence + Options + Consequences
"""
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════
# PHASE 2: ADVANCED FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════

def compute_pressure_index(required_rr, wickets_left, overs_left):
    """
    Pressure Index = f(required_rate, wickets_left, overs_left)
    0 = no pressure, 100 = maximum pressure
    """
    if overs_left <= 0 or wickets_left <= 0:
        return 100.0
    rr_pressure  = min((required_rr / 12.0) * 50, 50)
    wkt_pressure = max(0, (10 - wickets_left) / 10 * 30)
    time_pressure = max(0, (1 - overs_left / 20) * 20)
    return round(min(rr_pressure + wkt_pressure + time_pressure, 100), 1)


def compute_momentum_index(last_12_balls):
    """
    Momentum Index based on last 12 balls (exponential decay weighted)
    last_12_balls: list of (runs, is_wicket) tuples, oldest first
    Returns: -100 (batting collapse) to +100 (batting dominant)
    """
    if not last_12_balls:
        return 0.0
    weights = np.exp(np.linspace(-1, 0, len(last_12_balls)))
    weights /= weights.sum()
    score = 0
    for i, (runs, is_wkt) in enumerate(last_12_balls):
        ball_score = runs * 2 - (is_wkt * 20)
        score += ball_score * weights[i]
    return round(np.clip(score * 10, -100, 100), 1)


def ema_batting_avg(scores, alpha=0.3):
    """Exponentially weighted moving average of batting scores"""
    if not scores:
        return 0.0
    ema = scores[0]
    for s in scores[1:]:
        ema = alpha * s + (1 - alpha) * ema
    return round(ema, 2)


def compute_rr_gap(required_rr, current_rr):
    """Required RR vs Current RR gap analysis"""
    gap = required_rr - current_rr
    if gap > 4:     severity = "CRITICAL"
    elif gap > 2:   severity = "HIGH"
    elif gap > 0.5: severity = "MODERATE"
    elif gap < -2:  severity = "COMFORTABLE"
    else:           severity = "BALANCED"
    return {"gap": round(gap, 2), "severity": severity,
            "current_rr": round(current_rr, 2), "required_rr": round(required_rr, 2)}


def compute_clutch_score(high_pressure_scores, normal_scores):
    """
    Clutch score: how much better player performs under pressure
    > 1.0 = clutch player, < 1.0 = struggles under pressure
    """
    if not high_pressure_scores or not normal_scores:
        return 1.0
    hp_avg = np.mean(high_pressure_scores)
    nm_avg = np.mean(normal_scores)
    if nm_avg <= 0:
        return 1.0
    return round(hp_avg / nm_avg, 3)


def compute_consistency_score(scores):
    """Consistency 0-100 (100=perfectly consistent)"""
    if len(scores) < 3:
        return 50.0
    cv = np.std(scores) / max(np.mean(scores), 1)
    return round(max(0, 100 - cv * 100), 1)


def compute_volatility_score(scores):
    """Volatility score 0-10 (10=extremely volatile)"""
    if len(scores) < 3:
        return 5.0
    cv = np.std(scores) / max(np.mean(scores), 1)
    return round(min(cv * 10, 10), 1)


def classify_player_type(avg, sr, consistency, clutch_score, volatility):
    """
    Classify player into: Clutch / High-Risk / Stable / Flat-track Bully / Inconsistent
    """
    if clutch_score >= 1.15 and consistency >= 50:
        return "🔥 CLUTCH PLAYER", "#3fb950", "Elevates game under pressure — ideal captain pick"
    elif volatility >= 7 and avg >= 25:
        return "⚡ HIGH-RISK, HIGH-REWARD", "#f0883e", "Explosive but inconsistent — good for GL, risky for SL"
    elif consistency >= 70 and avg >= 25:
        return "🛡️ STABLE PERFORMER", "#58a6ff", "Consistent runs, low variance — ideal for safe fantasy picks"
    elif sr >= 150 and consistency < 50:
        return "🎯 FLAT-TRACK BULLY", "#e3b341", "Thrives in easy conditions but struggles under pressure"
    else:
        return "📊 SITUATIONAL", "#8b949e", "Performance varies by context — analyse match-up carefully"


# ═══════════════════════════════════════════════════════
# PHASE 4: DECISION INTELLIGENCE ENGINE
# ═══════════════════════════════════════════════════════

def generate_full_decision(scenario: dict, n_sim: int = 10000) -> dict:
    """
    Generate a complete AI decision with:
    - Data Evidence
    - Simulation Support
    - Confidence Level
    - Multiple Options (Aggressive / Safe)
    - Consequence Analysis
    """
    np.random.seed(42)

    req_rr      = scenario.get("required_rr", 9.0)
    curr_rr     = scenario.get("current_rr", 8.0)
    wickets_left = scenario.get("wickets_left", 7)
    overs_left  = scenario.get("overs_left", 8.0)
    target      = scenario.get("target", 170)
    runs_scored = scenario.get("runs_scored", 100)
    batting_team = scenario.get("batting_team", "Team A")
    bowling_team = scenario.get("bowling_team", "Team B")

    pressure = compute_pressure_index(req_rr, wickets_left, overs_left)
    rr_analysis = compute_rr_gap(req_rr, curr_rr)
    gap = rr_analysis["gap"]

    # ── Base win probability ──
    ease = max(0, min(1, (12 - req_rr) / 12))
    wkt_factor = wickets_left / 10
    base_wp = ease * 0.6 + wkt_factor * 0.4
    base_wp = max(0.02, min(0.98, base_wp))

    # ── Simulate aggressive vs safe strategy ──
    def simulate_strategy(boundary_rate, wicket_risk_multiplier, n=n_sim):
        wins = 0
        for _ in range(n):
            r_remaining = max(target - runs_scored, 0)
            overs_rem   = overs_left
            wkts_rem    = wickets_left
            scored = 0
            for _ in np.arange(0, overs_rem, 1/6):
                if wkts_rem <= 0 or scored >= r_remaining:
                    break
                # Boundary ball
                if np.random.random() < boundary_rate:
                    scored += np.random.choice([4, 6])
                else:
                    scored += np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
                if np.random.random() < 0.03 * wicket_risk_multiplier:
                    wkts_rem -= 1
            if scored >= r_remaining and wkts_rem > 0:
                wins += 1
        return round(wins / n * 100, 1)

    # Aggressive: more boundaries, higher wicket risk
    aggressive_wp = simulate_strategy(boundary_rate=0.20, wicket_risk_multiplier=1.5, n=2000)
    # Safe: fewer boundaries, lower wicket risk
    safe_wp = simulate_strategy(boundary_rate=0.10, wicket_risk_multiplier=0.7, n=2000)

    # ── Confidence level ──
    uncertainty = abs(0.5 - base_wp)
    if uncertainty > 0.3:    confidence = "HIGH"
    elif uncertainty > 0.15: confidence = "MEDIUM"
    else:                     confidence = "LOW"

    # ── Consequence analysis ──
    if gap > 2:
        consequence_if_ignored = f"Win probability drops to ~{max(base_wp-0.15, 0.02)*100:.0f}% if scoring rate not increased"
    elif gap < -2:
        consequence_if_ignored = f"Protecting wickets ensures ~{min(base_wp+0.08, 0.98)*100:.0f}% win probability"
    else:
        consequence_if_ignored = "Maintaining current rate gives balanced outcome"

    # ── Primary recommendation ──
    if pressure > 70:
        primary = "ATTACK"
        reasoning = f"Pressure Index {pressure:.0f}/100 — must accelerate immediately"
    elif pressure < 30:
        primary = "CONSOLIDATE"
        reasoning = f"Low pressure ({pressure:.0f}/100) — protect wickets, build platform"
    else:
        primary = "ROTATE STRIKE"
        reasoning = f"Moderate pressure ({pressure:.0f}/100) — keep scoreboard moving"

    return {
        # Evidence
        "data_evidence": {
            "required_rr": req_rr,
            "current_rr": curr_rr,
            "rr_gap": gap,
            "rr_severity": rr_analysis["severity"],
            "pressure_index": pressure,
            "wickets_left": wickets_left,
            "overs_left": overs_left,
            "runs_needed": max(target - runs_scored, 0),
        },
        # Simulation
        "simulation": {
            "base_win_pct": round(base_wp * 100, 1),
            "aggressive_win_pct": aggressive_wp,
            "safe_win_pct": safe_wp,
            "aggressive_gain": round(aggressive_wp - base_wp * 100, 1),
            "safe_gain": round(safe_wp - base_wp * 100, 1),
        },
        # Decision
        "confidence": confidence,
        "primary_recommendation": primary,
        "reasoning": reasoning,
        "consequence_if_ignored": consequence_if_ignored,
        # Options
        "options": {
            "aggressive": {
                "name": "🚀 AGGRESSIVE",
                "action": f"Target boundaries every 2 balls — need {gap+curr_rr:.1f} RPO",
                "win_pct": aggressive_wp,
                "risk": "HIGH — wicket risk increases 50%",
            },
            "safe": {
                "name": "🛡️ CONSERVATIVE",
                "action": f"Rotate strike, wait for bad balls — target {curr_rr+0.5:.1f} RPO",
                "win_pct": safe_wp,
                "risk": "LOW — preserves wickets",
            },
        },
        # Storytelling
        "headline": _generate_headline(gap, pressure, wickets_left, batting_team),
        "so_what": _generate_so_what(gap, pressure, wickets_left, overs_left, primary),
    }


def _generate_headline(gap, pressure, wickets_left, team):
    if gap > 4:
        return f"🚨 {team} in CRISIS — Required rate {gap:.1f} above scoring rate"
    elif gap > 2:
        return f"⚠️ {team} under PRESSURE — Need to accelerate immediately"
    elif gap < -2:
        return f"✅ {team} in CONTROL — Well ahead of target, protect wickets"
    elif pressure > 70:
        return f"🔥 TENSE FINISH — Every ball matters now"
    else:
        return f"⚖️ BALANCED CONTEST — Match in the balance"


def _generate_so_what(gap, pressure, wickets_left, overs_left, primary):
    if primary == "ATTACK":
        return (f"**What to do:** Send aggressive batter NOW. Need boundary every 2 balls. "
                f"Remaining margin: {overs_left:.0f} overs × {wickets_left} wickets.")
    elif primary == "CONSOLIDATE":
        return (f"**What to do:** Protect wickets, rotate strike. "
                f"You have {overs_left:.0f} overs and {wickets_left} wickets — don't panic.")
    else:
        return (f"**What to do:** Maintain current approach, target every loose ball. "
                f"One big over can swing the match your way.")


# ═══════════════════════════════════════════════════════
# PHASE 5: SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════

def sensitivity_analysis(base_prob, feature_deltas: dict) -> list:
    """
    Show how win probability changes with each feature change.
    feature_deltas: {feature_name: (delta_value, label)}
    """
    results = []
    for feat, (delta, label) in feature_deltas.items():
        if "required_rr" in feat:
            prob_change = -delta * 0.035
        elif "wicket" in feat:
            prob_change = -delta * 0.08
        elif "runs_scored" in feat:
            prob_change = delta * 0.003
        elif "overs" in feat:
            prob_change = -delta * 0.01
        else:
            prob_change = delta * 0.02

        new_prob = max(0.02, min(0.98, base_prob + prob_change))
        results.append({
            "feature": feat,
            "label": label,
            "delta_feature": delta,
            "prob_change": round(prob_change * 100, 1),
            "new_prob": round(new_prob * 100, 1),
            "direction": "↑ improves" if prob_change > 0 else "↓ hurts",
        })
    return sorted(results, key=lambda x: abs(x["prob_change"]), reverse=True)


# ═══════════════════════════════════════════════════════
# PHASE 6: FANTASY ADVANCED
# ═══════════════════════════════════════════════════════

def compute_fantasy_ceiling_floor(fp_total, fp_batting, fp_bowling, volatility):
    """
    Ceiling = optimistic scenario (good match)
    Floor   = worst-case scenario
    Variance = range
    """
    ceil_mult  = 1 + volatility * 0.15
    floor_mult = 1 - volatility * 0.12
    ceiling = round(fp_total * ceil_mult, 1)
    floor   = round(max(fp_total * floor_mult, 2), 1)
    variance = round(ceiling - floor, 1)
    return {"ceiling": ceiling, "floor": floor, "variance": variance,
            "variance_label": "High" if variance > 30 else "Medium" if variance > 15 else "Low"}


def predict_ownership(fp_total, fp_rank, is_captain_candidate, role):
    """
    Predict % of fantasy teams that will pick this player
    (heuristic model based on rank + FP)
    """
    base = max(0, 60 - fp_rank * 4)
    if is_captain_candidate:
        base += 20
    if role == "AR":
        base += 10
    if fp_total > 80:
        base += 15
    elif fp_total < 30:
        base -= 20
    return round(max(5, min(95, base + np.random.normal(0, 3))), 0)


# ═══════════════════════════════════════════════════════
# PHASE 7: PLAYER ANALYTICS
# ═══════════════════════════════════════════════════════

def compute_xruns(balls_faced, sr_career, phase_factor=1.0):
    """Expected runs given balls faced"""
    return round(balls_faced * (sr_career / 100) * phase_factor, 1)


def compute_xwickets(balls_bowled, wicket_rate_career, phase_factor=1.0):
    """Expected wickets given balls bowled"""
    return round((balls_bowled / 6) * wicket_rate_career * phase_factor, 2)


def classify_player_full(scores_recent, scores_career,
                           pressure_scores, normal_scores,
                           sr, avg):
    """Full player classification with all 4 labels"""
    consistency = compute_consistency_score(scores_recent)
    volatility  = compute_volatility_score(scores_recent)
    clutch      = compute_clutch_score(pressure_scores, normal_scores)
    recent_avg  = np.mean(scores_recent) if scores_recent else 0
    career_avg  = np.mean(scores_career) if scores_career else recent_avg

    label, color, desc = classify_player_type(avg, sr, consistency, clutch, volatility)
    return {
        "label": label, "color": color, "description": desc,
        "consistency": consistency, "volatility": volatility,
        "clutch_score": clutch,
        "form_ratio": round(recent_avg / max(career_avg, 1), 2),
        "ema_avg": ema_batting_avg(scores_recent),
    }


# ═══════════════════════════════════════════════════════
# PHASE 8: VENUE BEST XI
# ═══════════════════════════════════════════════════════

def venue_best_xi(bat_stats, bowl_stats, n=11):
    """
    Select best 11 players for a venue based on venue-specific performance.
    bat_stats: [{player, runs, sr, innings}]
    bowl_stats: [{player, wkts, econ, innings}]
    """
    # Score batsmen
    bat_df = pd.DataFrame(bat_stats) if bat_stats else pd.DataFrame()
    bowl_df = pd.DataFrame(bowl_stats) if bowl_stats else pd.DataFrame()

    selected = []
    if not bat_df.empty:
        bat_df["score"] = bat_df["runs"] * 0.5 + bat_df.get("sr", pd.Series(120, index=bat_df.index)) * 0.3
        top_bat = bat_df.nlargest(6, "score")
        for _, r in top_bat.iterrows():
            selected.append({"player": r["player"], "role": "BAT",
                              "venue_runs": r.get("runs",0), "innings": r.get("innings",0)})

    if not bowl_df.empty:
        bowl_df["score"] = bowl_df["wkts"] * 3 - bowl_df.get("econ", pd.Series(8, index=bowl_df.index)) * 0.5
        top_bowl = bowl_df.nlargest(4, "score")
        for _, r in top_bowl.iterrows():
            selected.append({"player": r["player"], "role": "BOWL",
                              "venue_wkts": r.get("wkts",0), "innings": r.get("innings",0)})

    # Add 1 WK (first batsman becomes WK)
    if selected:
        selected[0]["role"] = "WK"

    return selected[:n]


def spin_vs_pace_analysis(deliveries_venue):
    """Analyse spin vs pace effectiveness at a venue"""
    if deliveries_venue.empty:
        return {"spin_econ": 8.0, "pace_econ": 8.0, "spin_wkt_rate": 0.05,
                "pace_wkt_rate": 0.05, "recommendation": "No data"}

    spin_kw = ["chahal","rashid","ashwin","jadeja","kuldeep","tahir","narine",
               "piyush","imran","axar","sunil"]
    d = deliveries_venue.copy()
    d["is_spin"] = d["bowler"].str.lower().apply(
        lambda x: any(k in x for k in spin_kw)).astype(int)

    spin_d = d[d["is_spin"]==1]
    pace_d = d[d["is_spin"]==0]

    def phase_stats(sub):
        lg = sub[sub["is_legal"]==1] if "is_legal" in sub.columns else sub
        if len(lg) < 12:
            return {"econ": 8.0, "wkt_rate": 0.05, "balls": len(lg)}
        econ = sub["total_runs"].sum() / max(len(lg)/6, 0.1)
        wkt_rate = sub["is_wicket"].sum() / max(len(lg), 1)
        return {"econ": round(econ, 2), "wkt_rate": round(wkt_rate*100, 2), "balls": len(lg)}

    spin_s = phase_stats(spin_d)
    pace_s = phase_stats(pace_d)

    if spin_s["econ"] < pace_s["econ"] - 0.5:
        rec = "🌀 **Spin-dominant venue** — prioritise spinners in team selection"
    elif pace_s["econ"] < spin_s["econ"] - 0.5:
        rec = "⚡ **Pace-dominant venue** — fast bowlers are key here"
    else:
        rec = "⚖️ **Balanced venue** — both spin and pace are effective"

    return {
        "spin_econ": spin_s["econ"], "spin_wkt_rate": spin_s["wkt_rate"],
        "spin_balls": spin_s["balls"],
        "pace_econ": pace_s["econ"], "pace_wkt_rate": pace_s["wkt_rate"],
        "pace_balls": pace_s["balls"],
        "recommendation": rec,
        "winner": "spin" if spin_s["econ"] < pace_s["econ"] else "pace"
    }


# ═══════════════════════════════════════════════════════
# PITCH CLASSIFIER (required by phase2_pages)
# ═══════════════════════════════════════════════════════

def classify_pitch(avg_score, avg_wickets, avg_rr_pp, avg_rr_death):
    """
    Returns pitch type label + strategy string based on venue stats.
    """
    score = 0
    if avg_score > 180:   score += 2
    elif avg_score > 165: score += 1
    elif avg_score < 145: score -= 2

    if avg_rr_pp > 8.5:    score += 1
    if avg_rr_death > 11:  score += 1
    if avg_wickets < 6:    score += 1

    if score >= 3:
        return "🏏 BATTING PARADISE", "Bat first — big totals are common. Chasing is high risk."
    elif score >= 1:
        return "⚖️ BALANCED PITCH", "Toss matters — read conditions on the day."
    elif score >= -1:
        return "🎳 BOWLER-FRIENDLY", "Restrict first — 155-165 is a strong total here."
    else:
        return "🎳 SEAM & SWING TRACK", "Bowl first — wickets up front win matches here."