"""
CricketBrain AI — Insight Generation & Decision Engine
Rule-based + Statistical hybrid insight engine
Toss advisor | Bowling strategy | Batting order suggestions
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED  = os.path.join(DATA_DIR, "cleaned")

def load_data():
    m = pd.read_csv(os.path.join(CLEANED, "matches.csv"), parse_dates=["date"])
    d = pd.read_csv(os.path.join(CLEANED, "deliveries.csv"))
    m = m.sort_values("date", na_position="last").reset_index(drop=True)
    d["legal_ball"] = (d.get("is_wide",0) + d.get("is_noball",0) == 0).astype(int)
    return m, d

# ─────────────────────────────────────────────────────────
# TEAM FORM INSIGHTS
# ─────────────────────────────────────────────────────────
def team_form_insights(team: str, matches: pd.DataFrame, last_n: int = 10) -> list:
    tm = matches[(matches["team1"] == team) | (matches["team2"] == team)].copy()
    tm = tm.sort_values("date", na_position="last").tail(last_n)
    if len(tm) == 0:
        return [f"No recent data for {team}"]

    wins    = (tm["winner"] == team).sum()
    losses  = len(tm) - wins
    win_pct = wins / len(tm) * 100
    streak  = 0
    streak_type = ""
    for _, row in tm[::-1].iterrows():
        if row["winner"] == team:
            if streak_type in ("", "W"):
                streak += 1; streak_type = "W"
            else:
                break
        else:
            if streak_type in ("", "L"):
                streak += 1; streak_type = "L"
            else:
                break

    insights = []
    if win_pct >= 70:
        insights.append(f"🔥 {team} is in **superb form** — won {wins}/{len(tm)} last matches ({win_pct:.0f}%)")
    elif win_pct >= 50:
        insights.append(f"✅ {team} has **decent form** — won {wins}/{len(tm)} matches recently")
    else:
        insights.append(f"⚠️ {team} in **poor form** — only {wins}/{len(tm)} wins recently ({win_pct:.0f}%)")

    if streak >= 3:
        emoji = "🔥" if streak_type == "W" else "❌"
        label = "winning" if streak_type == "W" else "losing"
        insights.append(f"{emoji} {team} is on a **{streak}-match {label} streak**")

    return insights

# ─────────────────────────────────────────────────────────
# TOSS ADVANTAGE ADVISOR
# ─────────────────────────────────────────────────────────
def toss_advisor(team1: str, team2: str, venue: str,
                  matches: pd.DataFrame) -> dict:
    venue_m = matches[matches["venue"] == venue]
    all_insights = []
    recommendation = "bat"
    confidence = 50.0

    if len(venue_m) >= 5:
        bat_first_wins = (venue_m["bat_first_won"] == 1).sum() if "bat_first_won" in venue_m else 0
        total = len(venue_m)
        bat_win_rate = bat_first_wins / total * 100
        chase_win_rate = 100 - bat_win_rate

        if bat_win_rate >= 55:
            recommendation = "bat"
            confidence = bat_win_rate
            all_insights.append(f"📊 At **{venue}**, batting first teams win **{bat_win_rate:.0f}%** of matches ({total} games)")
            all_insights.append(f"💡 **Recommendation: Bat first** — venue favours setting a total")
        elif chase_win_rate >= 55:
            recommendation = "field"
            confidence = chase_win_rate
            all_insights.append(f"📊 At **{venue}**, chasing teams win **{chase_win_rate:.0f}%** of matches ({total} games)")
            all_insights.append(f"💡 **Recommendation: Field first** — venue favours chasing")
        else:
            all_insights.append(f"📊 At **{venue}**, results are **evenly split** (bat first: {bat_win_rate:.0f}%)")
            all_insights.append("💡 **Recommendation: Depend on pitch reading & team strengths**")
    else:
        all_insights.append(f"⚠️ Limited data for {venue} — general IPL trend suggests **fielding first** gives slight edge")
        recommendation = "field"
        confidence = 52.0

    # Team toss record
    for team, opp in [(team1, team2), (team2, team1)]:
        team_toss = matches[matches["toss_winner"] == team]
        if len(team_toss) >= 10:
            team_toss_wins = (team_toss["winner"] == team).sum()
            toss_win_pct = team_toss_wins / len(team_toss) * 100
            if toss_win_pct >= 60:
                all_insights.append(f"🎯 {team} wins **{toss_win_pct:.0f}%** of matches when winning the toss — good toss winners")

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 1),
        "insights": all_insights,
    }

# ─────────────────────────────────────────────────────────
# BOWLING STRATEGY ENGINE
# ─────────────────────────────────────────────────────────
def bowling_strategy(batting_team: str, deliveries: pd.DataFrame,
                      matches: pd.DataFrame) -> dict:
    d = deliveries[deliveries["batting_team"] == batting_team].copy()
    insights = []

    # Phase vulnerability
    for phase in ["powerplay","middle","death"]:
        sub  = d[d["phase"] == phase]
        slg  = sub[sub["legal_ball"] == 1]
        if len(slg) < 20:
            continue
        rr   = sub["total_runs"].sum() / (len(slg)/6)
        wkts = sub["is_wicket"].sum()
        wkt_per_over = wkts / max(len(slg)/6, 1)

        if rr < 7.0 and phase == "powerplay":
            insights.append(f"🎯 Attack in **powerplay** — {batting_team} scores at only {rr:.1f}/over")
        if rr > 10.0 and phase == "death":
            insights.append(f"⚠️ Protect the **death overs** — {batting_team} is explosive ({rr:.1f}/over)")
        if wkt_per_over > 0.5 and phase == "middle":
            insights.append(f"✅ Middle overs bring wickets — {wkt_per_over:.2f} wkts/over vs {batting_team}")

    # Best bowlers against this team
    bowl_vs = (
        d.groupby("bowler")
        .agg(wkts=("is_wicket","sum"), balls=("legal_ball","sum"), runs=("total_runs","sum"))
        .reset_index()
    )
    bowl_vs = bowl_vs[bowl_vs["balls"] >= 24]
    bowl_vs["econ"] = bowl_vs["runs"] / (bowl_vs["balls"]/6)
    bowl_vs["wkt_rate"] = bowl_vs["wkts"] / bowl_vs["balls"] * 6

    top_wkt = bowl_vs.sort_values("wkts", ascending=False).head(3)
    if len(top_wkt):
        names = ", ".join(top_wkt["bowler"].tolist())
        insights.append(f"🏆 Top wicket-takers vs {batting_team}: **{names}**")

    # Spin vs Pace analysis
    spin_kw = ["chahal","rashid","ashwin","jadeja","kuldeep","tahir","narine","piyush","imran","imad"]
    d["is_spin"] = d["bowler"].str.lower().apply(lambda x: any(k in x for k in spin_kw)).astype(int)

    spin_d = d[d["is_spin"]==1]
    pace_d = d[d["is_spin"]==0]

    if len(spin_d[spin_d["legal_ball"]==1]) >= 30 and len(pace_d[pace_d["legal_ball"]==1]) >= 30:
        spin_rr = spin_d["total_runs"].sum() / max(spin_d["legal_ball"].sum()/6, 1)
        pace_rr = pace_d["total_runs"].sum() / max(pace_d["legal_ball"].sum()/6, 1)

        if spin_rr < pace_rr - 0.5:
            insights.append(f"🌀 {batting_team} struggles vs **spin** — use spinners! (Spin: {spin_rr:.1f} vs Pace: {pace_rr:.1f} RPO)")
        elif pace_rr < spin_rr - 0.5:
            insights.append(f"🏃 {batting_team} struggles vs **pace** — use fast bowlers! (Pace: {pace_rr:.1f} vs Spin: {spin_rr:.1f} RPO)")
        else:
            insights.append(f"⚖️ {batting_team} handles both spin and pace equally well")

    return {"team": batting_team, "strategy_insights": insights}

# ─────────────────────────────────────────────────────────
# PLAYER FORM INSIGHTS
# ─────────────────────────────────────────────────────────
def player_form_insights(player: str, deliveries: pd.DataFrame,
                          matches: pd.DataFrame, role: str = "bat") -> list:
    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce") if "date" in dm.columns else pd.NaT
    dm = dm.sort_values("date", na_position="last")
    insights = []

    if role == "bat":
        pdf = dm[dm["batter"] == player].groupby("match_id").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"),
            date=("date","max")
        ).reset_index().sort_values("date", na_position="last")

        if len(pdf) < 3:
            return [f"Insufficient batting data for {player}"]

        recent = pdf.tail(5)["runs"]
        career_avg = pdf["runs"].mean()
        recent_avg = recent.mean()

        if recent_avg > career_avg * 1.25:
            insights.append(f"🔥 {player} is in **peak batting form** — averaging {recent_avg:.0f} runs in last 5 innings (career: {career_avg:.0f})")
        elif recent_avg < career_avg * 0.7:
            insights.append(f"📉 {player} is in **poor form** — {recent_avg:.0f} runs/innings recently vs career {career_avg:.0f}")
        else:
            insights.append(f"📊 {player} is in **decent form** — {recent_avg:.0f} runs/innings (career avg: {career_avg:.0f})")

        # Consistency
        cv = pdf["runs"].std() / max(pdf["runs"].mean(), 1)
        if cv < 0.6:
            insights.append(f"✅ **Consistent performer** — low variance in scores (CV: {cv:.2f})")
        elif cv > 1.2:
            insights.append(f"⚠️ **Inconsistent** — high variance in performance (CV: {cv:.2f})")

    elif role == "bowl":
        pdf = dm[dm["bowler"] == player].groupby("match_id").agg(
            wkts=("is_wicket","sum"), runs=("total_runs","sum"),
            balls=("legal_ball","sum"), date=("date","max")
        ).reset_index().sort_values("date", na_position="last")
        pdf["econ"] = pdf["runs"] / (pdf["balls"]/6).replace(0,1)

        if len(pdf) < 3:
            return [f"Insufficient bowling data for {player}"]

        recent_wkts = pdf.tail(5)["wkts"].mean()
        recent_econ = pdf.tail(5)["econ"].mean()
        career_wkts = pdf["wkts"].mean()
        career_econ = pdf["econ"].mean()

        if recent_wkts > career_wkts * 1.2:
            insights.append(f"🔥 {player} on a **wicket-taking roll** — {recent_wkts:.1f} wkts/match recently")
        if recent_econ < career_econ * 0.9:
            insights.append(f"🎯 {player} is **economical** — Economy {recent_econ:.1f} recently")
        elif recent_econ > career_econ * 1.1:
            insights.append(f"⚠️ {player} is **expensive** lately — Economy {recent_econ:.1f}")

    return insights

# ─────────────────────────────────────────────────────────
# MATCH PREVIEW INSIGHTS
# ─────────────────────────────────────────────────────────
def match_preview_insights(team1: str, team2: str, venue: str,
                             matches: pd.DataFrame, deliveries: pd.DataFrame) -> dict:
    t1_form = team_form_insights(team1, matches)
    t2_form = team_form_insights(team2, matches)
    toss    = toss_advisor(team1, team2, venue, matches)
    bowl1   = bowling_strategy(team2, deliveries, matches)  # team1 bowling against team2
    bowl2   = bowling_strategy(team1, deliveries, matches)  # team2 bowling against team1

    # H2H
    h2h = matches[
        ((matches["team1"]==team1) & (matches["team2"]==team2)) |
        ((matches["team1"]==team2) & (matches["team2"]==team1))
    ]
    t1_h2h_wins = (h2h["winner"] == team1).sum()
    t2_h2h_wins = (h2h["winner"] == team2).sum()
    h2h_insights = []
    if len(h2h) > 0:
        h2h_insights.append(f"📜 **H2H Record:** {team1} {t1_h2h_wins} — {t2_h2h_wins} {team2} (last {len(h2h)} meetings)")
        if t1_h2h_wins > t2_h2h_wins * 1.5:
            h2h_insights.append(f"🏆 {team1} **historically dominates** this fixture")
        elif t2_h2h_wins > t1_h2h_wins * 1.5:
            h2h_insights.append(f"🏆 {team2} **historically dominates** this fixture")

    return {
        f"{team1}_form":      t1_form,
        f"{team2}_form":      t2_form,
        "toss_advice":        toss,
        f"how_to_bowl_vs_{team2}": bowl1["strategy_insights"],
        f"how_to_bowl_vs_{team1}": bowl2["strategy_insights"],
        "h2h_insights":       h2h_insights,
    }

# ─────────────────────────────────────────────────────────
# GENERATE SHAREABLE VIRAL CONTENT
# ─────────────────────────────────────────────────────────
def viral_insight(team1: str, team2: str, sim_result: dict) -> str:
    t1_pct = sim_result.get("team1_win_pct", 50)
    t2_pct = sim_result.get("team2_win_pct", 50)
    n      = sim_result.get("n_simulations", 10000)

    lines = [
        f"🤖 CricketBrain AI ran {n:,} simulations of {team1} vs {team2}",
        f"",
        f"📊 Results:",
        f"  {team1}: {t1_pct:.1f}% wins",
        f"  {team2}: {t2_pct:.1f}% wins",
        f"",
        f"🎯 Predicted Score Range:",
        f"  {team1}: {sim_result.get('team1_score_p10',140)}–{sim_result.get('team1_score_p90',195)}",
        f"  {team2}: {sim_result.get('team2_score_p10',138)}–{sim_result.get('team2_score_p90',192)}",
        f"",
        f"#IPL #CricketAnalytics #CricketBrainAI #DataScience",
    ]
    return "\n".join(lines)

if __name__ == "__main__":
    m, d = load_data()
    print(team_form_insights("Chennai Super Kings", m))
    print("---")
    print(toss_advisor("Chennai Super Kings", "Mumbai Indians", "MA Chidambaram Stadium", m))
    print("---")
    print(bowling_strategy("Royal Challengers Bengaluru", d, m))