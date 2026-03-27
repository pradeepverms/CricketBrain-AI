"""
CricketBrain AI — Player Weakness & Matchup Engine
Detects batting/bowling weaknesses | Phase analysis | Bowler matchups
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED  = os.path.join(DATA_DIR, "cleaned")
MIN_BALLS = 20   # Minimum balls for reliable analysis

def load_data():
    d = pd.read_csv(os.path.join(CLEANED, "deliveries.csv"))
    m = pd.read_csv(os.path.join(CLEANED, "matches.csv"), parse_dates=["date"])
    d["legal_ball"] = (d.get("is_wide", 0) + d.get("is_noball", 0) == 0).astype(int)
    return d, m

# ─────────────────────────────────────────────────────────
# BATSMAN WEAKNESS ANALYSIS
# ─────────────────────────────────────────────────────────
def batsman_weakness(player: str, deliveries: pd.DataFrame, matches: pd.DataFrame) -> dict:
    d = deliveries[deliveries["batter"] == player].copy()
    if len(d) < MIN_BALLS:
        return {"error": f"Insufficient data for {player} (< {MIN_BALLS} balls)"}

    result = {"player": player, "total_balls": len(d), "weaknesses": [], "strengths": []}

    # 1. Phase performance
    phase_stats = []
    for phase in ["powerplay", "middle", "death"]:
        sub = d[d["phase"] == phase]
        legal = sub[sub["legal_ball"] == 1]
        if len(legal) < 5:
            continue
        sr  = sub["batsman_runs"].sum() / len(legal) * 100
        avg = sub["batsman_runs"].sum() / max(sub["is_wicket"].sum(), 1)
        phase_stats.append({"phase": phase, "sr": round(sr, 1), "avg": round(avg, 1), "balls": len(legal)})

    career_sr = d["batsman_runs"].sum() / max(d["legal_ball"].sum(), 1) * 100
    for p in phase_stats:
        if p["sr"] < career_sr * 0.75 and p["balls"] >= MIN_BALLS:
            result["weaknesses"].append(
                f"⚠️ Struggles in **{p['phase']}** overs — SR {p['sr']:.0f} vs career {career_sr:.0f}"
            )
        elif p["sr"] > career_sr * 1.1 and p["balls"] >= MIN_BALLS:
            result["strengths"].append(
                f"✅ Excellent in **{p['phase']}** overs — SR {p['sr']:.0f}"
            )
    result["phase_stats"] = phase_stats

    # 2. Specific bowler types (if bowler name contains hints)
    spin_keywords   = ["chahal","rashid","ashwin","jadeja","kuldeep","tahir","narine","piyush","rahul","sunil","sai","imran"]
    pace_keywords   = ["bumrah","malinga","rabada","boult","ishant","umesh","shami","cummins","steyn","strc","harshal","natarajan"]
    left_arm_keywords = ["boult","zaheer","axar","starc","jaydev","t natarajan","arshdeep"]

    spin_d     = d[d["bowler"].str.lower().apply(lambda x: any(k in x for k in spin_keywords))]
    pace_d     = d[d["bowler"].str.lower().apply(lambda x: any(k in x for k in pace_keywords))]
    la_pace_d  = d[d["bowler"].str.lower().apply(lambda x: any(k in x for k in left_arm_keywords))]

    def sr_analysis(sub, label):
        legal = sub[sub["legal_ball"] == 1]
        if len(legal) < MIN_BALLS:
            return
        sr_val  = sub["batsman_runs"].sum() / len(legal) * 100
        dis_rate = sub["is_wicket"].sum() / len(legal) * 100
        if sr_val < career_sr * 0.8:
            result["weaknesses"].append(f"⚠️ Struggles vs **{label}** — SR {sr_val:.0f} (career {career_sr:.0f})")
        elif sr_val > career_sr * 1.1:
            result["strengths"].append(f"✅ Dominates **{label}** — SR {sr_val:.0f}")
        return {"label": label, "sr": round(sr_val, 1), "dismissal_rate": round(dis_rate, 2), "balls": len(legal)}

    result["vs_spin"]     = sr_analysis(spin_d, "Spin")
    result["vs_pace"]     = sr_analysis(pace_d, "Pace")
    result["vs_la_pace"]  = sr_analysis(la_pace_d, "Left-arm Pace")

    # 3. Dismissal patterns
    if "dismissal_kind" in d.columns:
        dismissed = d[d["is_wicket"] == 1]
        if len(dismissed) > 3:
            top_dismiss = dismissed["dismissal_kind"].value_counts().head(3)
            result["dismissal_patterns"] = top_dismiss.to_dict()
            if "bowled" in top_dismiss.index or "lbw" in top_dismiss.index:
                result["weaknesses"].append("⚠️ Susceptible to **bowled/LBW** — possible straight-ball vulnerability")

    # 4. Death over pressure index
    death_d = d[d["phase"] == "death"]
    if len(death_d[death_d["legal_ball"] == 1]) >= 10:
        death_sr = death_d["batsman_runs"].sum() / death_d["legal_ball"].sum() * 100
        death_dis = death_d["is_wicket"].sum() / max(death_d["legal_ball"].sum(), 1) * 100
        result["death_sr"]    = round(death_sr, 1)
        result["death_wicket_rate"] = round(death_dis, 2)

    # 5. Consistency score
    match_scores = d.groupby("match_id")["batsman_runs"].sum()
    result["consistency_score"] = round(float(match_scores.std() / max(match_scores.mean(), 1)), 3) if len(match_scores) > 3 else None

    return result

# ─────────────────────────────────────────────────────────
# BOWLER WEAKNESS ANALYSIS
# ─────────────────────────────────────────────────────────
def bowler_weakness(player: str, deliveries: pd.DataFrame, matches: pd.DataFrame) -> dict:
    d = deliveries[deliveries["bowler"] == player].copy()
    legal = d[d["legal_ball"] == 1]
    if len(legal) < MIN_BALLS:
        return {"error": f"Insufficient data for {player}"}

    result = {"player": player, "total_balls": len(legal), "weaknesses": [], "strengths": []}

    career_econ = d["total_runs"].sum() / (len(legal) / 6)

    # Phase economy
    phase_stats = []
    for phase in ["powerplay", "middle", "death"]:
        sub  = d[d["phase"] == phase]
        slg  = sub[sub["legal_ball"] == 1]
        if len(slg) < 6:
            continue
        overs = len(slg) / 6
        econ  = sub["total_runs"].sum() / max(overs, 0.01)
        wkt_r = sub["is_wicket"].sum() / max(len(slg), 1) * 100
        phase_stats.append({"phase": phase, "economy": round(econ, 2), "wicket_rate": round(wkt_r, 2), "balls": len(slg)})

        if econ > career_econ * 1.15 and len(slg) >= 12:
            result["weaknesses"].append(f"⚠️ Expensive in **{phase}** overs — Economy {econ:.1f}")
        elif econ < career_econ * 0.9 and len(slg) >= 12:
            result["strengths"].append(f"✅ Excellent in **{phase}** overs — Economy {econ:.1f}")

    result["phase_stats"] = phase_stats

    # Top batsmen who score well against this bowler
    bat_vs = (
        d.groupby("batter")
        .agg(runs=("batsman_runs","sum"), balls=("legal_ball","sum"), wkts=("is_wicket","sum"))
        .reset_index()
    )
    bat_vs = bat_vs[bat_vs["balls"] >= 10]
    bat_vs["sr"] = bat_vs["runs"] / bat_vs["balls"] * 100
    bat_vs = bat_vs.sort_values("sr", ascending=False)

    result["difficult_batsmen"] = bat_vs.head(5)[["batter","runs","balls","sr","wkts"]].to_dict("records")
    result["wickets_taken_vs"]  = bat_vs.sort_values("wkts", ascending=False).head(5)[["batter","wkts"]].to_dict("records")

    return result

# ─────────────────────────────────────────────────────────
# BATSMAN vs BOWLER MATCHUP MATRIX
# ─────────────────────────────────────────────────────────
def matchup_matrix(batsman: str, bowler: str, deliveries: pd.DataFrame) -> dict:
    d = deliveries[(deliveries["batter"] == batsman) & (deliveries["bowler"] == bowler)].copy()
    legal = d[d["legal_ball"] == 1]

    if len(legal) < 5:
        return {
            "batsman": batsman, "bowler": bowler,
            "balls": len(legal), "result": "Insufficient data (< 5 balls)",
            "verdict": "no_data"
        }

    runs  = d["batsman_runs"].sum()
    wkts  = d["is_wicket"].sum()
    sr    = runs / len(legal) * 100
    avg   = runs / max(wkts, 1)

    # Phase breakdown
    phase_data = {}
    for phase in ["powerplay","middle","death"]:
        sub = d[d["phase"] == phase]
        sl  = sub[sub["legal_ball"] == 1]
        if len(sl) >= 3:
            phase_data[phase] = {
                "runs": int(sub["batsman_runs"].sum()),
                "balls": len(sl),
                "sr": round(sub["batsman_runs"].sum() / len(sl) * 100, 1),
                "wkts": int(sub["is_wicket"].sum()),
            }

    # Dismissal types
    dismissals = {}
    if "dismissal_kind" in d.columns:
        dismissals = d[d["is_wicket"] == 1]["dismissal_kind"].value_counts().to_dict()

    # Verdict
    if sr >= 160:
        verdict = f"🏏 {batsman} **dominates** {bowler} — SR {sr:.0f}, {wkts} wickets in {len(legal)} balls"
    elif sr >= 120:
        verdict = f"✅ {batsman} scores **comfortably** vs {bowler} — SR {sr:.0f}"
    elif sr >= 80:
        verdict = f"⚖️ **Even contest** between {batsman} and {bowler}"
    elif wkts >= 3:
        verdict = f"⚠️ {bowler} **dominates** {batsman} — {wkts} dismissals in {len(legal)} balls"
    else:
        verdict = f"⚠️ {batsman} **struggles** vs {bowler} — SR only {sr:.0f}"

    return {
        "batsman": batsman, "bowler": bowler,
        "balls": len(legal), "runs": int(runs),
        "wickets": int(wkts), "strike_rate": round(sr, 1),
        "average": round(avg, 1), "verdict": verdict,
        "phase_breakdown": phase_data,
        "dismissal_types": dismissals,
    }

# ─────────────────────────────────────────────────────────
# TOP MATCHUPS (for a given match)
# ─────────────────────────────────────────────────────────
def key_matchups(batting_team_players: list, bowling_team_players: list,
                 deliveries: pd.DataFrame, top_n: int = 5) -> list:
    matchups = []
    for bat in batting_team_players:
        for bowl in bowling_team_players:
            m = matchup_matrix(bat, bowl, deliveries)
            if m.get("balls", 0) >= 6:
                matchups.append(m)

    matchups.sort(key=lambda x: x.get("balls", 0), reverse=True)
    return matchups[:top_n]

# ─────────────────────────────────────────────────────────
# TEAM WEAKNESS AGAINST OPPOSITION
# ─────────────────────────────────────────────────────────
def team_weakness_report(team: str, deliveries: pd.DataFrame, matches: pd.DataFrame) -> dict:
    # Bowling team is the one not batting
    d = deliveries[deliveries["batting_team"] == team].copy()
    legal = d[d.get("legal_ball", 1) == 1]

    report = {"team": team, "weaknesses": [], "strengths": []}

    # Phase analysis
    for phase in ["powerplay","middle","death"]:
        sub = d[d["phase"] == phase]
        sl  = sub[sub.get("legal_ball", True) == 1] if "legal_ball" in sub else sub
        if len(sl) < 20:
            continue
        rr = sub["total_runs"].sum() / (len(sl) / 6)
        wkt_rate = sub["is_wicket"].sum() / len(sl) * 100

        if rr < 7.5 and phase == "death":
            report["weaknesses"].append(f"⚠️ Batting struggles in **death overs** — RR {rr:.1f}")
        elif rr > 10 and phase == "death":
            report["strengths"].append(f"✅ Explosive death batting — RR {rr:.1f}")

        if rr < 6.5 and phase == "powerplay":
            report["weaknesses"].append(f"⚠️ Slow powerplay start — RR {rr:.1f}")
        elif rr > 9 and phase == "powerplay":
            report["strengths"].append(f"✅ Explosive powerplay — RR {rr:.1f}")

    return report

if __name__ == "__main__":
    d, m = load_data()
    print(batsman_weakness("V Kohli", d, m))
    print("---")
    print(bowler_weakness("JJ Bumrah", d, m))
    print("---")
    print(matchup_matrix("V Kohli", "JJ Bumrah", d))