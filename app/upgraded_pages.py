"""
CricketBrain AI — All Upgraded Page Functions
Implements every improvement from the product upgrade prompt.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Intelligence functions (inlined to avoid import issues) ──

def classify_form(recent_avg, career_avg):
    if career_avg <= 0: return "⚪ Insufficient Data", 0
    ratio = recent_avg / career_avg
    if ratio >= 1.25:   return "🔥 HOT FORM", ratio
    elif ratio >= 1.05: return "✅ GOOD FORM", ratio
    elif ratio >= 0.85: return "📊 AVERAGE FORM", ratio
    else:               return "📉 DECLINING", ratio

def risk_score(scores):
    if len(scores) < 3: return 5.0
    cv = np.std(scores) / max(np.mean(scores), 1)
    return round(min(cv * 10, 10), 1)

def trend_prediction(scores):
    if len(scores) < 4: return "stable", 0
    x = np.arange(len(scores))
    slope = np.polyfit(x, scores, 1)[0]
    if slope > 1.5:   return "improving", slope
    elif slope < -1.5: return "declining", slope
    return "stable", slope

def form_vs_career(recent_avg, career_avg):
    if career_avg <= 0: return 0
    return round((recent_avg - career_avg) / career_avg * 100, 1)

def detect_turning_points(over_probs, threshold=0.12):
    turning = []
    for i in range(1, len(over_probs)):
        shift = over_probs[i][1] - over_probs[i-1][1]
        if abs(shift) >= threshold:
            turning.append({
                "over": over_probs[i][0],
                "shift": round(shift * 100, 1),
                "direction": "📈 Batting team gained" if shift > 0 else "📉 Bowling team gained",
                "magnitude": "MAJOR" if abs(shift) > 0.20 else "SIGNIFICANT"
            })
    return turning

def why_prob_changed(over_data, prev_prob, curr_prob):
    reasons = []
    wkts  = over_data.get("wickets", 0)
    runs  = over_data.get("runs", 0)
    rr    = over_data.get("run_rate", 0)
    req_rr = over_data.get("required_rr", 0)
    if wkts >= 2:    reasons.append(f"💀 **{wkts} wickets fell** — batting collapse")
    elif wkts == 1:  reasons.append("💀 **Key wicket** changed the balance")
    if req_rr > 0 and rr > 0:
        if req_rr > rr + 3:   reasons.append(f"⚡ Required RR ({req_rr:.1f}) outpacing scoring ({rr:.1f})")
        elif rr > req_rr + 2: reasons.append(f"🏏 Batting team ahead — scoring {rr:.1f} vs needed {req_rr:.1f}")
    if runs >= 15:    reasons.append(f"💥 **Big over** — {runs} runs, pressure released")
    elif runs <= 3 and wkts == 0: reasons.append("🔒 **Dot ball over** — pressure building")
    if not reasons:   reasons.append("📊 Gradual match position shift")
    return reasons

def what_to_do_next(required_rr, current_rr, wickets_left, overs_left, batting_first=False):
    decisions = []
    if not batting_first:
        gap = required_rr - current_rr
        if gap > 3:
            decisions.append("🚨 **URGENT:** Need a boundary every 2 balls — send in big hitters now")
            decisions.append(f"⚡ Required RR is {required_rr:.1f} — attack immediately")
        elif gap > 1.5:
            decisions.append(f"⚠️ Required RR rising to {required_rr:.1f} — must accelerate")
        elif gap < -1:
            decisions.append("✅ Ahead of target — protect wickets, accelerate in last 4 overs")
        else:
            decisions.append("⚖️ Evenly poised — rotate strike, wait for bad ball")
        if wickets_left <= 3 and overs_left > 5:
            decisions.append("💀 **LOW WICKETS WARNING** — protect tail, play smart cricket")
    else:
        if required_rr > 10:
            decisions.append(f"🔒 **DEFEND MODE** — chasing needs {required_rr:.1f} RPO. Bowl full & straight.")
        elif required_rr < 7:
            decisions.append("⚡ Chase is on — attack with wicket-taking deliveries now")
    return decisions

def playoff_probability(team, current_pts, matches_played, total_matches, all_teams_pts, n_sim=2000):
    np.random.seed(42)
    remaining = max(total_matches - matches_played, 0)
    qualify_count = 0
    other_teams = {t: p for t, p in all_teams_pts.items() if t != team}
    for _ in range(n_sim):
        sim_pts = current_pts + np.random.binomial(remaining, 0.5) * 2
        others_sim = {}
        for t, p in other_teams.items():
            t_rem = max(0, min(remaining, total_matches - (p // 2)))
            others_sim[t] = p + np.random.binomial(t_rem, 0.5) * 2
        all_pts = {team: sim_pts, **others_sim}
        ranking = sorted(all_pts, key=lambda x: all_pts[x], reverse=True)
        if team in ranking[:4]:
            qualify_count += 1
    return round(qualify_count / n_sim * 100, 1)

def classify_pitch(avg_score, avg_wickets, avg_rr_pp, avg_rr_death):
    score = 0
    if avg_score > 180: score += 2
    elif avg_score > 165: score += 1
    elif avg_score < 145: score -= 2
    if avg_rr_pp > 8.5: score += 1
    if avg_rr_death > 11: score += 1
    if avg_wickets < 6: score += 1
    if score >= 3:    return "🏏 BATTING PARADISE", "Bat first — big totals common. Chase is high risk."
    elif score >= 1:  return "⚖️ BALANCED PITCH", "Toss matters — winner should read conditions."
    elif score >= -1: return "🎳 BOWLER-FRIENDLY", "Restrict first — 160 is a strong total here."
    else:             return "🎳 SEAM & SWING TRACK", "Bowl first — wickets up front win here."

def impact_score(runs, sr, avg, wickets=0, economy=None, matches=1):
    bat_impact  = min((runs / max(avg, 1)) * (sr / 120) * 40, 50)
    bowl_impact = 0
    if wickets > 0 and economy is not None:
        bowl_impact = min(wickets * 10 + max(0, (8 - economy) * 3), 50)
    return round(min(bat_impact + bowl_impact, 100), 1)

def sustainability_score(recent_scores, career_avg):
    if len(recent_scores) < 3: return 50
    recent_avg = np.mean(recent_scores)
    cv = np.std(recent_scores) / max(recent_avg, 1)
    above_career = (recent_avg - career_avg) / max(career_avg, 1)
    return round(max(0, 100 - cv*100)*0.6 + min(above_career*50, 50)*0.4, 1)

def why_breakout(curr_sr, prev_sr, curr_avg, prev_avg, curr_innings, prev_innings):
    reasons = []
    if curr_sr - prev_sr > 15:    reasons.append(f"⚡ Strike rate surged +{curr_sr-prev_sr:.0f}")
    if curr_avg - prev_avg > 8:   reasons.append(f"📈 Average improved +{curr_avg-prev_avg:.0f}")
    if curr_innings > prev_innings * 1.3: reasons.append("🏏 More opportunities / higher batting position")
    return reasons if reasons else ["📊 Consistent improvement across metrics"]

def classify_play_style(sr, avg, boundary_pct):
    if sr >= 160 and boundary_pct >= 0.5:  return "💥 AGGRESSIVE STRIKER", "Explosive boundary-hitter"
    elif sr >= 145 and avg >= 30:           return "⚡ POWER HITTER", "Consistent high-scorer"
    elif avg >= 35 and sr < 130:            return "🛡️ ANCHOR BATTER", "Builds innings, high average"
    elif sr >= 130 and avg >= 25:           return "⚖️ BALANCED BATTER", "Adaptable to any situation"
    else:                                   return "🎯 SITUATIONAL PLAYER", "Role varies by context"

def fantasy_player_reasoning(player_row, captain, vice_captain):
    p = player_row["player"]; fp = player_row["fp_total"]
    fp_bat = player_row.get("fp_batting", 0); fp_bowl = player_row.get("fp_bowling", 0)
    lines = []
    if p == captain:      lines.append(f"👑 **Captain** — highest FP ({fp:.0f} pts → {fp*2:.0f} with 2×)")
    elif p == vice_captain: lines.append(f"🥈 **Vice-captain** — {fp:.0f} pts → {fp*1.5:.0f} with 1.5×")
    if fp_bat > 40:  lines.append(f"🏏 Strong batting form — {fp_bat:.0f} batting pts")
    if fp_bowl > 20: lines.append(f"🎳 Wicket-taking threat — {fp_bowl:.0f} bowling pts")
    return lines if lines else [f"📊 Consistent performer — {fp:.0f} expected pts"]

def diversity_score(team_df):
    roles = team_df["role"].value_counts()
    teams = team_df["team"].value_counts()
    role_entropy = -(roles/roles.sum() * np.log(roles/roles.sum() + 1e-9)).sum()
    team_balance = 1 - abs(teams.iloc[0] - teams.iloc[-1]) / 11 if len(teams) > 1 else 0.5
    return round((role_entropy + team_balance) * 10, 1)

def detect_run_anomalies(over_stats, league_avg_rr=8.5):
    anomalies = []
    if over_stats.empty: return anomalies
    mean_rr = over_stats["rr"].mean(); std_rr = over_stats["rr"].std()
    for _, row in over_stats.iterrows():
        z = (row["rr"] - mean_rr) / max(std_rr, 0.1)
        if abs(z) > 1.8:
            over_num = int(row.get("over_bin", row.get("over_num", 0)))
            anomalies.append({
                "over": over_num, "rr": round(row["rr"], 1), "z": round(z, 2),
                "type": "spike" if z > 0 else "collapse",
                "label": f"Over {over_num+1}: {row['rr']:.1f} RPO — {'abnormal spike 🚀' if z>0 else 'scoring collapse 💀'}"
            })
    return anomalies


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight_card(text, kind="info"):
    cls = {"info":"insight-card","warning":"warning-card","success":"success-card"}.get(kind,"insight-card")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def decision_card(text):
    st.markdown(f'<div style="background:#0d2137;border-left:4px solid #f0883e;border-radius:8px;padding:0.8rem 1rem;margin:0.4rem 0;font-size:0.95rem;">{text}</div>', unsafe_allow_html=True)

def headline_card(title, subtitle, color="#58a6ff"):
    st.markdown(f'''<div style="background:linear-gradient(135deg,#1a2a3a,#0d1117);border:1px solid {color};border-radius:12px;padding:1rem 1.5rem;margin:0.5rem 0;">
    <div style="font-size:1.3rem;font-weight:800;color:{color};">{title}</div>
    <div style="font-size:0.9rem;color:#8b949e;margin-top:0.3rem;">{subtitle}</div>
    </div>''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 9 UPGRADED: WIN PROBABILITY TRACKER
# ═══════════════════════════════════════════════════════
def page_win_probability_v2(matches, deliveries):
    st.title("📈 Win Probability Tracker")
    headline_card(
        "🧠 AI-Powered Match Intelligence",
        "Live win probability • Turning point detection • Decision recommendations • Confidence scoring"
    )

    teams = sorted(set(matches["team1"].dropna().tolist()+matches["team2"].dropna().tolist()))
    c1,c2 = st.columns(2)
    t1 = c1.selectbox("Team 1",[""] + teams, key="wp1")
    t2 = c2.selectbox("Team 2",[""] + teams, key="wp2")
    if not t1 or not t2:
        st.info("Select both teams to analyse a match.")
        return

    h2h = matches[
        ((matches["team1"]==t1)&(matches["team2"]==t2))|
        ((matches["team1"]==t2)&(matches["team2"]==t1))
    ].sort_values("date")
    if h2h.empty:
        st.warning("No matches found between these teams.")
        return

    match_id = st.selectbox("Select Match", h2h["match_id"].tolist(),
        format_func=lambda x: f"{str(h2h[h2h['match_id']==x]['date'].values[0])[:10]} — {t1} vs {t2}")

    dm   = deliveries[deliveries["match_id"]==match_id].copy()
    mr   = h2h[h2h["match_id"]==match_id].iloc[0]
    if dm.empty:
        st.warning("No ball-by-ball data for this match.")
        return

    winner = str(mr.get("winner","N/A"))

    # ── Summary headline ──
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Winner", winner)
    c2.metric("Balls Played", len(dm))
    c3.metric("Total Runs", int(dm["total_runs"].sum()))
    c4.metric("Total Wickets", int(dm["is_wicket"].sum()))

    # ── Build over-by-over stats ──
    dm["over_num"] = pd.to_numeric(dm.get("over_num", 0), errors="coerce").fillna(0).astype(int)
    over_stats = dm.groupby(["batting_team","over_num"]).agg(
        runs=("total_runs","sum"),
        balls=("is_legal","sum"),
        wickets=("is_wicket","sum")
    ).reset_index()
    over_stats["rr"] = over_stats["runs"] / over_stats["balls"].replace(0,1) * 6

    # ── Simulate over-by-over win probability (heuristic) ──
    # Based on run accumulation and wickets
    teams_in_match = dm["batting_team"].unique()
    if len(teams_in_match) < 1:
        st.warning("Insufficient data.")
        return

    bat_first = teams_in_match[0]
    bat_second = teams_in_match[1] if len(teams_in_match) > 1 else t2

    # Innings 1 & 2 data
    inn1 = dm[dm["batting_team"]==bat_first].copy()
    inn2 = dm[dm["batting_team"]==bat_second].copy() if len(teams_in_match)>1 else pd.DataFrame()

    inn1_total = int(inn1["total_runs"].sum())
    target = inn1_total + 1

    # ── Cumulative runs chart ──
    section("📊 Ball-by-Ball Cumulative Runs")
    fig = go.Figure()
    dm["ball_idx"] = range(len(dm))
    for team, col in [(bat_first,"#58a6ff"), (bat_second,"#f0883e")]:
        sub = dm[dm["batting_team"]==team].copy()
        if sub.empty: continue
        sub = sub.reset_index(drop=True)
        sub["ball_idx2"] = range(len(sub))
        sub["cum_r"] = sub["total_runs"].cumsum()
        fig.add_trace(go.Scatter(x=sub["ball_idx2"], y=sub["cum_r"],
                                  name=str(team), mode="lines", line=dict(width=2.5)))
    fig.update_layout(template="plotly_dark", height=320,
                       xaxis_title="Ball", yaxis_title="Cumulative Runs",
                       title="Scoring Progression")
    st.plotly_chart(fig, use_container_width=True)

    # ── Over-by-over win probability (heuristic WP) ──
    section("🎯 Win Probability by Over")
    if not inn2.empty and inn1_total > 0:
        inn2["over_num"] = pd.to_numeric(inn2.get("over_num",0), errors="coerce").fillna(0).astype(int)
        wp_data = []
        cum_runs = 0
        cum_wkts = 0
        for ov in sorted(inn2["over_num"].unique()):
            sub = inn2[inn2["over_num"]==ov]
            cum_runs += sub["total_runs"].sum()
            cum_wkts += sub["is_wicket"].sum()
            overs_done = ov + 1
            overs_left = max(20 - overs_done, 0.1)
            runs_needed = target - cum_runs
            wkts_left = 10 - cum_wkts
            rr_req = runs_needed / overs_left if overs_left > 0 else 99

            # Heuristic WP for batting second team
            if cum_runs >= target:
                wp = 0.99
            elif wkts_left <= 0:
                wp = 0.01
            else:
                ease = max(0, min(1, (12 - rr_req) / 12))
                wkt_factor = wkts_left / 10
                wp = ease * 0.6 + wkt_factor * 0.4
                wp = max(0.02, min(0.98, wp))

            wp_data.append({
                "over": overs_done,
                "wp_bat_second": wp,
                "wp_bat_first": 1-wp,
                "runs_scored": int(cum_runs),
                "wickets": int(cum_wkts),
                "req_rr": round(rr_req, 1),
                "curr_rr": round(cum_runs/overs_done, 1) if overs_done>0 else 0
            })

        if wp_data:
            wp_df = pd.DataFrame(wp_data)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=wp_df["over"], y=wp_df["wp_bat_second"]*100,
                                       name=f"{bat_second} (Chasing)",
                                       line=dict(color="#58a6ff",width=2.5), fill="tozeroy",
                                       fillcolor="rgba(88,166,255,0.1)"))
            fig2.add_trace(go.Scatter(x=wp_df["over"], y=wp_df["wp_bat_first"]*100,
                                       name=f"{bat_first} (Defending)",
                                       line=dict(color="#f0883e",width=2.5)))
            fig2.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig2.update_layout(template="plotly_dark", height=320,
                                yaxis_title="Win Probability %",
                                xaxis_title="Over",
                                title="Win Probability Progression")
            st.plotly_chart(fig2, use_container_width=True)

            # ── TURNING POINT DETECTION ──
            section("🚨 Match Turning Points")
            wp_tuples = [(r["over"], r["wp_bat_second"]) for _, r in wp_df.iterrows()]
            turning = detect_turning_points(wp_tuples, threshold=0.12)
            if turning:
                for tp in turning[:5]:
                    color = "warning" if tp["shift"] < 0 else "success"
                    insight_card(
                        f"**Over {tp['over']}** — {tp['direction']} by **{abs(tp['shift']):.0f}%** ({tp['magnitude']} shift)",
                        color
                    )
            else:
                insight_card("📊 No major turning points — match progressed steadily", "info")

            # ── WHY DID PROBABILITY CHANGE ──
            section("🔍 Why Did Win Probability Change?")
            # Find over with biggest shift
            if len(wp_df) > 1:
                wp_df["shift"] = wp_df["wp_bat_second"].diff().abs()
                biggest_shift = wp_df.nlargest(1, "shift").iloc[0]
                ov = int(biggest_shift["over"])
                sub_ov = inn2[inn2["over_num"]==ov-1]
                reasons = why_prob_changed(
                    {"wickets": int(sub_ov["is_wicket"].sum()),
                     "runs": int(sub_ov["total_runs"].sum()),
                     "run_rate": float(sub_ov["total_runs"].sum()/max(len(sub_ov),1)*6),
                     "required_rr": float(biggest_shift["req_rr"])},
                    float(wp_df[wp_df["over"]==ov-1]["wp_bat_second"].values[0] if ov > 1 else 0.5),
                    float(biggest_shift["wp_bat_second"])
                )
                st.markdown(f"**🔄 Biggest shift in Over {ov}:**")
                for r in reasons:
                    insight_card(r, "warning" if "wicket" in r.lower() or "pressure" in r.lower() else "info")

            # ── WHAT SHOULD TEAM DO NEXT ──
            section("🎯 Decision Engine — What To Do Next")
            last = wp_df.iloc[-1]
            decisions = what_to_do_next(
                required_rr=last["req_rr"],
                current_rr=last["curr_rr"],
                wickets_left=10-last["wickets"],
                overs_left=20-last["over"],
                batting_first=False
            )
            for d in decisions:
                decision_card(d)

            # ── Confidence score ──
            max_shift = wp_df["shift"].max() if "shift" in wp_df.columns else 0
            confidence = max(30, min(85, 75 - max_shift * 100))
            st.metric("🤖 Model Confidence", f"{confidence:.0f}%",
                       help="Based on match volatility — lower confidence in high-variance matches")
    else:
        insight_card("⚠️ Only 1 innings data available — full WP tracker requires both innings", "warning")


# ═══════════════════════════════════════════════════════
# PAGE 11 UPGRADED: RUN HEATMAP
# ═══════════════════════════════════════════════════════
def page_run_heatmap_v2(matches, deliveries):
    st.title("🌡️ Run Scoring Heatmap")
    headline_card(
        "📊 Phase Intelligence + Anomaly Detection",
        "Powerplay · Middle · Death analysis • Anomaly detection • vs League Average comparison"
    )

    teams   = sorted(set(matches["team1"].dropna().tolist()+matches["team2"].dropna().tolist()))
    c1,c2   = st.columns(2)
    team    = c1.selectbox("Team",[""] + teams, key="rh_t")
    seasons = sorted(matches["season"].dropna().unique().tolist())
    season  = c2.selectbox("Season",["All"] + [str(int(s)) for s in seasons], key="rh_s")
    if not team:
        st.info("Select a team.")
        return

    dm = deliveries.copy()
    dm["season"] = pd.to_numeric(dm.get("season",2024), errors="coerce")
    if season != "All":
        dm = dm[dm["season"]==int(season)]
    dm = dm[dm["batting_team"]==team]
    if dm.empty:
        st.warning("No data.")
        return

    dm["over_bin"] = pd.to_numeric(dm.get("over_num",0), errors="coerce").fillna(0).astype(int)
    heat = dm.groupby("over_bin").agg(
        runs=("total_runs","sum"), balls=("is_legal","sum"), wkts=("is_wicket","sum")
    ).reset_index()
    heat["rr"] = heat["runs"]/heat["balls"].replace(0,1)*6

    # ── League average (all teams same season) ──
    all_dm = deliveries.copy()
    if season != "All":
        all_dm = all_dm[all_dm["season"]==int(season)] if "season" in all_dm.columns else all_dm
    all_dm["over_bin"] = pd.to_numeric(all_dm.get("over_num",0), errors="coerce").fillna(0).astype(int)
    league = all_dm.groupby("over_bin").agg(
        runs=("total_runs","sum"), balls=("is_legal","sum")
    ).reset_index()
    league["league_rr"] = league["runs"]/league["balls"].replace(0,1)*6

    heat = heat.merge(league[["over_bin","league_rr"]], on="over_bin", how="left")
    heat["vs_league"] = ((heat["rr"] - heat["league_rr"]) / heat["league_rr"].replace(0,1) * 100).round(1)

    # ── Summary stats ──
    c1,c2,c3 = st.columns(3)
    c1.metric("Avg Run Rate", f"{heat['rr'].mean():.2f}")
    c2.metric("Best Over (avg)", f"Over {heat.nlargest(1,'rr')['over_bin'].values[0]+1} ({heat.nlargest(1,'rr')['rr'].values[0]:.1f} RPO)")
    c3.metric("Most Wickets Over", f"Over {heat.nlargest(1,'wkts')['over_bin'].values[0]+1}")

    # ── Phase-wise insights ──
    section("📊 Phase Intelligence — Powerplay · Middle · Death")
    pp   = heat[heat["over_bin"] <= 5]
    mid  = heat[(heat["over_bin"] >= 6) & (heat["over_bin"] <= 14)]
    dth  = heat[heat["over_bin"] >= 15]

    c1,c2,c3 = st.columns(3)
    def phase_metric(col, phase_df, name, color):
        if phase_df.empty: return
        rr  = phase_df["runs"].sum()/phase_df["balls"].sum()*6
        wkts= int(phase_df["wkts"].sum())
        lg  = phase_df["league_rr"].mean() if "league_rr" in phase_df else rr
        diff= rr - lg
        col.metric(f"{name}", f"{rr:.2f} RPO",
                    delta=f"{diff:+.2f} vs league avg",
                    delta_color="normal")
        col.caption(f"💀 {wkts} wkts lost in phase")

    phase_metric(c1, pp,  "⚡ Powerplay (Ov 1-6)",  "#58a6ff")
    phase_metric(c2, mid, "🎯 Middle (Ov 7-15)",    "#e3b341")
    phase_metric(c3, dth, "💥 Death (Ov 16-20)",    "#f0883e")

    # Phase insights NL
    pp_rr  = pp["runs"].sum()/pp["balls"].sum()*6   if not pp.empty  else 0
    dth_rr = dth["runs"].sum()/dth["balls"].sum()*6 if not dth.empty else 0
    mid_rr = mid["runs"].sum()/mid["balls"].sum()*6 if not mid.empty else 0

    if dth_rr > 11:
        insight_card(f"💥 {team} is **explosive in death overs** ({dth_rr:.1f} RPO) — power-hitters excel here", "success")
    elif dth_rr < 8.5:
        insight_card(f"⚠️ {team} **struggles in death overs** ({dth_rr:.1f} RPO) — batting weakness identified", "warning")
    if pp_rr > 9:
        insight_card(f"🚀 {team} gets **fast starts in powerplay** ({pp_rr:.1f} RPO) — openers are destructive", "success")
    if mid_rr < 7:
        insight_card(f"🔒 {team} **consolidates in middle overs** ({mid_rr:.1f} RPO) — conservative approach", "info")

    # ── Run Rate Chart with phase markers ──
    section("⚾ Run Rate by Over")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=heat["over_bin"]+1, y=heat["rr"],
                          name=f"{team} RPO",
                          marker_color=[
                              "#58a6ff" if o<=5 else "#e3b341" if o<=14 else "#f0883e"
                              for o in heat["over_bin"]
                          ], opacity=0.85))
    if "league_rr" in heat.columns:
        fig.add_trace(go.Scatter(x=heat["over_bin"]+1, y=heat["league_rr"],
                                  name="League Average",
                                  line=dict(color="white", width=1.5, dash="dash"), opacity=0.6))
    fig.add_vline(x=6.5,  line_dash="dash", line_color="#58a6ff", opacity=0.4, annotation_text="PP End")
    fig.add_vline(x=15.5, line_dash="dash", line_color="#f0883e", opacity=0.4, annotation_text="Death Start")
    fig.update_layout(template="plotly_dark", height=340, xaxis_title="Over", yaxis_title="Run Rate")
    st.plotly_chart(fig, use_container_width=True)

    # ── vs League Average chart ──
    section("📈 Performance vs League Average by Over")
    heat["color"] = heat["vs_league"].apply(lambda x: "#3fb950" if x >= 0 else "#e84040")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=heat["over_bin"]+1, y=heat["vs_league"],
                           marker_color=heat["color"],
                           name="vs League Avg %"))
    fig2.add_hline(y=0, line_color="white", line_width=1, opacity=0.5)
    fig2.update_layout(template="plotly_dark", height=280,
                        xaxis_title="Over", yaxis_title="% vs League Avg",
                        title=f"{team} — Above/Below League Average Run Rate")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Anomaly detection ──
    section("🚨 Anomaly Detection — Unusual Scoring Overs")
    anomalies = detect_run_anomalies(heat)
    if anomalies:
        for a in anomalies:
            kind = "warning" if a["type"]=="spike" else "success"
            insight_card(f"**{a['label']}** (Z-score: {a['z']:+.1f})", kind)
        insight_card("💡 These overs represent **match-shift moments** — plan bowling changes around them", "info")
    else:
        insight_card("✅ No major scoring anomalies detected — consistent scoring pattern", "success")

    # Death vs league summary
    if not dth.empty and "league_rr" in dth.columns:
        dth_league = dth["league_rr"].mean()
        pct_diff   = (dth_rr - dth_league) / max(dth_league, 1) * 100
        if abs(pct_diff) > 5:
            insight_card(
                f"📊 {team} scores **{abs(pct_diff):.0f}% {'more' if pct_diff>0 else 'less'}** than league average in death overs",
                "success" if pct_diff > 0 else "warning"
            )


# ═══════════════════════════════════════════════════════
# PAGE 10 UPGRADED: SEASON POINTS TABLE
# ═══════════════════════════════════════════════════════
def page_season_table_v2(matches):
    st.title("📋 Season Points Table")
    headline_card(
        "🏆 AI Playoff Intelligence",
        "Points table • Playoff probability • Form trend • Monte Carlo qualification simulation"
    )

    seasons = sorted(matches["season"].dropna().unique().tolist(), reverse=True)
    season  = st.selectbox("Select Season", [int(s) for s in seasons])
    sm      = matches[matches["season"]==season]
    all_t   = list(set(sm["team1"].dropna().tolist()+sm["team2"].dropna().tolist()))
    total_matches_per_team = len(all_t) - 1  # round robin

    rows = []
    for t in all_t:
        tm     = sm[(sm["team1"]==t)|(sm["team2"]==t)]
        played = len(tm)
        wins   = int((tm["winner"]==t).sum())
        nr     = int((tm.get("result","")=="no result").sum()) if "result" in tm else 0
        losses = played - wins - nr
        pts    = wins * 2 + nr
        recent_5 = tm.sort_values("date").tail(5)
        form_str = "".join(["W" if r["winner"]==t else "L" for _,r in recent_5.iterrows()])
        rows.append({
            "Team":t, "P":played, "W":wins, "L":losses, "NR":nr,
            "Pts":pts, "Win%":round(wins/max(played,1)*100,1),
            "Form":form_str
        })

    tdf = pd.DataFrame(rows).sort_values("Pts",ascending=False).reset_index(drop=True)
    tdf.index += 1

    # ── Compute playoff probability ──
    pts_dict = dict(zip(tdf["Team"], tdf["Pts"]))
    tdf["Playoff %"] = tdf.apply(lambda r:
        playoff_probability(r["Team"], r["Pts"], r["P"],
                             total_matches_per_team, pts_dict, n_sim=2000), axis=1)

    # ── Expected position ──
    tdf["Exp. Pos."] = tdf["Pts"].rank(ascending=False, method="min").astype(int)

    # ── Form badge ──
    def form_badge(f):
        return f.replace("W","🟢").replace("L","🔴")
    tdf["Form"] = tdf["Form"].apply(form_badge)

    # ── Color playoff prob ──
    section("📊 Full Points Table with Playoff Intelligence")
    st.dataframe(
        tdf[["Team","P","W","L","Pts","Win%","Playoff %","Form","Exp. Pos."]].style
            .background_gradient(subset=["Pts"], cmap="Blues")
            .background_gradient(subset=["Playoff %"], cmap="Greens")
            .background_gradient(subset=["Win%"], cmap="YlOrRd"),
        use_container_width=True, hide_index=False
    )

    # ── Playoff insights ──
    section("🤖 AI Playoff Insights")
    qualified = tdf[tdf["Playoff %"] >= 70]
    danger    = tdf[(tdf["Playoff %"] > 20) & (tdf["Playoff %"] < 50)]
    eliminated= tdf[tdf["Playoff %"] < 10]

    for _, r in qualified.iterrows():
        insight_card(f"✅ **{r['Team']}** — {r['Playoff %']}% chance to qualify (🟢 Strong position)", "success")
    for _, r in danger.iterrows():
        insight_card(f"⚠️ **{r['Team']}** — {r['Playoff %']}% qualification chance (must-win situation)", "warning")
    for _, r in eliminated.iterrows():
        insight_card(f"❌ **{r['Team']}** — Only {r['Playoff %']}% chance remaining (mathematically tough)", "warning")

    # ── Charts ──
    c1,c2 = st.columns(2)
    with c1:
        fig = px.bar(tdf, x="Team", y="Pts", color="Win%",
                      color_continuous_scale="Blues", title=f"IPL {season} — Points")
        fig.update_layout(template="plotly_dark", height=330, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(tdf.sort_values("Playoff %",ascending=True),
                      x="Playoff %", y="Team", orientation="h",
                      color="Playoff %", color_continuous_scale="RdYlGn",
                      title="Playoff Qualification Probability")
        fig.update_layout(template="plotly_dark", height=330)
        st.plotly_chart(fig, use_container_width=True)

    # ── Form trend ──
    section("📈 Form Trend (Last 5 Matches)")
    insight_card("🟢 = Win  |  🔴 = Loss  — Reading left to right = oldest to latest", "info")
    for _, r in tdf.iterrows():
        form_display = r["Form"] if r["Form"] else "No recent data"
        w_count = form_display.count("🟢")
        trend = "🔥 Hot" if w_count >= 4 else "✅ Good" if w_count >= 3 else "⚠️ Struggling" if w_count <= 1 else "📊 Mixed"
        insight_card(f"**{r['Team']}** — {form_display}  →  {trend}", "success" if w_count>=4 else "warning" if w_count<=1 else "info")


# ═══════════════════════════════════════════════════════
# PAGE 12 UPGRADED: PLAYER FORM TRACKER
# ═══════════════════════════════════════════════════════
def page_form_tracker_v2(matches, deliveries):
    st.title("📈 Player Form Tracker")
    headline_card(
        "🧠 AI Form Intelligence",
        "HOT / DECLINING classification • Risk score • Trend prediction • vs Career baseline"
    )

    players = sorted(set(deliveries["batter"].dropna().tolist()+deliveries["bowler"].dropna().tolist()))
    c1,c2,c3 = st.columns([3,2,1])
    player = c1.selectbox("Player",[""] + players, key="ft_p")
    role   = c2.radio("Role",["Batting","Bowling"],horizontal=True)
    last_n = int(c3.number_input("Last N",5,50,15))
    if not player:
        st.info("Select a player.")
        return

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")
    dm = dm.sort_values("date", na_position="last")

    if role == "Batting":
        bat = dm[dm["batter"]==player].groupby("match_id").agg(
            runs=("batsman_runs","sum"), balls=("is_legal","sum"),
            dismissed=("is_wicket","max"), date=("date","max")
        ).reset_index().sort_values("date", na_position="last").tail(last_n)
        if bat.empty:
            st.warning("No batting data.")
            return

        bat["sr"] = bat["runs"]/bat["balls"].replace(0,1)*100
        career_avg  = float(bat["runs"].mean())
        recent_5    = bat.tail(5)["runs"].tolist()
        recent_avg  = float(bat.tail(5)["runs"].mean()) if len(bat)>=5 else career_avg

        # ── AI Intelligence Layer ──
        form_label, ratio = classify_form(recent_avg, career_avg)
        r_score   = risk_score(bat["runs"].tolist())
        trend, slope = trend_prediction(bat["runs"].tolist())
        vs_career = form_vs_career(recent_avg, career_avg)
        sust      = sustainability_score(recent_5, career_avg)

        # ── Summary headline ──
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Form Status", form_label)
        c2.metric("vs Career Avg", f"{vs_career:+.1f}%",
                   delta_color="normal" if vs_career >= 0 else "inverse")
        c3.metric("Risk Score", f"{r_score}/10",
                   help="10=very inconsistent, 0=extremely consistent")
        c4.metric("Trend", "📈 Improving" if trend=="improving" else "📉 Declining" if trend=="declining" else "➡️ Stable")

        # ── Form classification banner ──
        if "HOT" in form_label:
            st.success(f"🔥 **{player} is in HOT FORM** — {recent_avg:.0f} runs/innings recently vs career {career_avg:.0f}. Deploy as captain candidate.")
        elif "DECLINING" in form_label:
            st.warning(f"📉 **{player} is DECLINING** — {vs_career:.0f}% below career average. Consider dropping or moving down the order.")
        else:
            st.info(f"📊 **{player}** — Performing at {abs(vs_career):.0f}% {'above' if vs_career>=0 else 'below'} career baseline")

        # ── Chart ──
        bat["roll"] = bat["runs"].rolling(5,min_periods=1).mean()
        colors = ["#3fb950" if r>=50 else "#58a6ff" if r>=30 else "#f0883e" if r>=10 else "#e84040"
                  for r in bat["runs"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bat["date"], y=bat["runs"], marker_color=colors,
                              name="Runs", opacity=0.85))
        fig.add_trace(go.Scatter(x=bat["date"], y=bat["roll"], name="5-Match Avg",
                                  line=dict(color="white",width=2,dash="dash")))
        fig.add_hline(y=career_avg, line_dash="dot", line_color="#f0883e",
                       opacity=0.7, annotation_text=f"Career avg ({career_avg:.0f})")
        fig.update_layout(template="plotly_dark",
                           title=f"{player} — Batting Form ({form_label})", height=360)
        st.plotly_chart(fig, use_container_width=True)

        # ── Intelligence cards ──
        section("🧠 AI Form Intelligence")
        insight_card(f"📊 **Career Average:** {career_avg:.1f} runs/innings | **Recent (last 5):** {recent_avg:.1f} runs/innings", "info")
        insight_card(f"📈 **Performance Trend:** {trend.title()} (slope: {slope:+.2f} runs/match)", "success" if trend=="improving" else "warning" if trend=="declining" else "info")
        insight_card(f"🎯 **Sustainability:** {sust:.0f}/100 — {'Highly sustainable form' if sust>70 else 'Moderately sustainable' if sust>45 else 'Form may be a fluke'}", "success" if sust>70 else "warning")
        insight_card(f"⚠️ **Risk Score:** {r_score}/10 — {'Very consistent' if r_score<3 else 'Moderately consistent' if r_score<6 else 'High variance player'}", "success" if r_score<3 else "warning" if r_score>6 else "info")

        # ── Scenario context ──
        section("🎯 Decision Recommendation")
        if "HOT" in form_label and r_score < 5:
            decision_card(f"✅ **SELECT as CAPTAIN/VC** — {player} is in hot form AND consistent. High floor, high ceiling.")
        elif "HOT" in form_label and r_score >= 6:
            decision_card(f"⚡ **High-risk, high-reward DIFFERENTIAL** pick — in hot form but inconsistent. Good for small leagues.")
        elif "DECLINING" in form_label:
            decision_card(f"⚠️ **AVOID as captain** — declining form means high risk. Pick as non-multiplier only if unavoidable.")
        else:
            decision_card(f"📊 **Safe BAT/BOWL pick** — consistent performer, safe for large leagues.")

    else:
        bowl = dm[dm["bowler"]==player].groupby("match_id").agg(
            wkts=("is_wicket","sum"), runs=("total_runs","sum"),
            balls=("is_legal","sum"), date=("date","max")
        ).reset_index().sort_values("date", na_position="last").tail(last_n)
        if bowl.empty:
            st.warning("No bowling data.")
            return
        bowl["econ"] = bowl["runs"]/(bowl["balls"]/6).replace(0,1)
        career_wkts  = float(bowl["wkts"].mean())
        recent_wkts  = float(bowl.tail(5)["wkts"].mean()) if len(bowl)>=5 else career_wkts

        form_label, _ = classify_form(recent_wkts, career_wkts)
        r_score  = risk_score(bowl["wkts"].tolist())
        trend, slope = trend_prediction(bowl["wkts"].tolist())

        c1,c2,c3 = st.columns(3)
        c1.metric("Form Status", form_label)
        c2.metric("Trend", "📈 Improving" if trend=="improving" else "📉 Declining" if trend=="declining" else "➡️ Stable")
        c3.metric("Risk Score", f"{r_score}/10")

        if "HOT" in form_label:
            st.success(f"🔥 **{player} is on a wicket-taking roll** — {recent_wkts:.1f} wkts/match recently")
        elif "DECLINING" in form_label:
            st.warning(f"📉 **{player}'s form declining** — from {career_wkts:.1f} to {recent_wkts:.1f} wkts/match")

        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=bowl["date"],y=bowl["wkts"],name="Wickets",marker_color="#3fb950",opacity=0.85),secondary_y=False)
        fig.add_trace(go.Scatter(x=bowl["date"],y=bowl["econ"],name="Economy",line=dict(color="#f0883e",width=2)),secondary_y=True)
        fig.add_hline(y=career_wkts, secondary_y=False, line_dash="dot",
                       line_color="white", opacity=0.5,
                       annotation_text=f"Career avg ({career_wkts:.1f})")
        fig.update_layout(template="plotly_dark",title=f"{player} — Bowling Form ({form_label})",height=360)
        st.plotly_chart(fig, use_container_width=True)

        section("🧠 AI Form Intelligence")
        insight_card(f"📊 Career avg: {career_wkts:.1f} wkts/match | Recent (last 5): {recent_wkts:.1f} wkts/match", "info")
        insight_card(f"📈 Trend: {trend.title()} | Risk Score: {r_score}/10", "success" if trend=="improving" else "warning")

        section("🎯 Decision Recommendation")
        if "HOT" in form_label:
            decision_card(f"✅ **BOWL them in powerplay AND death** — {player} is at peak. Give full quota.")
        elif "DECLINING" in form_label:
            decision_card(f"⚠️ **Rotate {player}** — form dipping. Combine with another bowler to cover risk.")


# ═══════════════════════════════════════════════════════
# PAGE 13 UPGRADED: BREAKOUT PLAYERS
# ═══════════════════════════════════════════════════════
def page_breakout_v2(matches, deliveries):
    st.title("⚡ Breakout Players & Rising Stars")
    headline_card(
        "🚀 AI Breakout Intelligence",
        "Why are they breaking out? • Sustainability score • Style change detection"
    )

    seasons = sorted(matches["season"].dropna().unique().tolist(), reverse=True)
    season  = st.selectbox("Current Season", [int(s) for s in seasons], key="bp_s")
    prev    = season - 1
    dm      = deliveries.copy()
    dm["season"] = pd.to_numeric(dm.get("season",0), errors="coerce")

    def season_bat(s):
        sub = dm[dm["season"]==s]
        agg = sub.groupby("batter").agg(
            runs=("batsman_runs","sum"), balls=("is_legal","sum"),
            innings=("match_id","nunique"), wkts=("is_wicket","sum")
        ).reset_index()
        agg["sr"]  = agg["runs"]/agg["balls"].replace(0,1)*100
        agg["avg"] = agg["runs"]/agg["wkts"].replace(0,1)
        return agg[agg["innings"]>=3]

    curr = season_bat(season)
    prev_df = season_bat(prev) if prev in dm["season"].dropna().unique() else pd.DataFrame()

    if prev_df.empty:
        section(f"Top Batsmen — {season}")
        st.dataframe(curr.sort_values("runs",ascending=False).head(15),hide_index=True)
        return

    mg = curr.merge(prev_df[["batter","runs","sr","avg","innings"]],
                     on="batter", suffixes=("_curr","_prev"), how="inner")
    mg["runs_growth"]  = mg["runs_curr"]  - mg["runs_prev"]
    mg["sr_growth"]    = mg["sr_curr"]    - mg["sr_prev"]
    mg["avg_growth"]   = mg["avg_curr"]   - mg["avg_prev"]
    mg["inn_growth"]   = mg["innings_curr"] - mg["innings_prev"]
    mg = mg[mg["runs_growth"]>0].sort_values("runs_growth",ascending=False).head(12)

    if mg.empty:
        st.info("No breakout players found for this season comparison.")
        return

    # Add intelligence columns
    mg["sustainability"] = mg.apply(lambda r:
        sustainability_score([r["runs_curr"]] * 5, r["runs_prev"]/max(r["innings_prev"],1)),
        axis=1)

    mg["why"] = mg.apply(lambda r:
        " | ".join(why_breakout(r["sr_curr"],r["sr_prev"],r["avg_curr"],r["avg_prev"],
                                 r["innings_curr"],r["innings_prev"])),
        axis=1)

    section(f"🚀 Biggest Improvements: {prev} → {season}")
    fig = px.bar(mg, x="runs_growth", y="batter", orientation="h",
                  color="sustainability", color_continuous_scale="Greens",
                  labels={"runs_growth":"Extra Runs","batter":"Player","sustainability":"Sustainability Score"},
                  title="Run Improvement with Sustainability Score")
    fig.update_layout(template="plotly_dark", height=420, yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    section("🧠 AI Breakout Intelligence — Why & How Sustainable?")
    for _, r in mg.head(8).iterrows():
        sust = r["sustainability"]
        sust_label = "🟢 Highly Sustainable" if sust>70 else "🟡 Moderately Sustainable" if sust>45 else "🔴 May be a Fluke"
        insight_card(
            f"**{r['batter']}** +{r['runs_growth']:.0f} runs | {sust_label} ({sust:.0f}/100) | "
            f"**Why:** {r['why']}",
            "success" if sust>70 else "warning" if sust<45 else "info"
        )

    section("📊 Sustainability vs Run Growth Bubble Chart")
    fig2 = px.scatter(mg, x="runs_growth", y="sustainability",
                       size="runs_curr", color="sr_growth",
                       text="batter", color_continuous_scale="RdYlGn",
                       labels={"runs_growth":"Run Growth","sustainability":"Sustainability Score",
                                "sr_growth":"SR Change","runs_curr":"Total Runs"},
                       title="Breakout Quality: Run Growth vs Sustainability")
    fig2.update_traces(textposition="top center")
    fig2.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig2, use_container_width=True)
    insight_card("💡 **Top-right quadrant** = genuine breakout players — high growth AND sustainable. Top-left = improvement but may regress.", "info")


# ═══════════════════════════════════════════════════════
# PAGE 14 UPGRADED: FANTASY OPTIMIZER
# ═══════════════════════════════════════════════════════
def page_fantasy_v2(matches, deliveries, optimize_team_ilp, estimate_player_fp):
    st.title("💰 Dream11 Fantasy Team Optimizer")
    headline_card(
        "🧠 AI-Powered Fantasy Intelligence",
        "ILP optimization • WHY THIS PLAYER reasoning • Risk vs Reward • Alternative picks • Ownership strategy"
    )

    teams = sorted(set(matches["team1"].dropna().tolist()+matches["team2"].dropna().tolist()))
    c1,c2 = st.columns(2)
    t1_sel = c1.selectbox("Team 1",[""] + teams, key="fan_t1")
    t2_sel = c2.selectbox("Team 2",[""] + teams, key="fan_t2")
    if not t1_sel or not t2_sel:
        st.info("Select both teams.")
        return

    strategy = st.radio("Optimization Strategy",["maximize","safe","differentiated"],horizontal=True,
                         format_func=lambda x:{
                             "maximize":"🚀 Maximize Points (aggressive)",
                             "safe":"🛡️ Safe/Consistent (lower risk)",
                             "differentiated":"🎯 Differentiated (low-ownership contrarian)"
                         }[x])

    st.info("💡 **Strategy Guide:** Maximize = best for head-to-head · Safe = best for large leagues · Differentiated = best for grand leagues with 50K+ teams")

    if st.button("⚡ Generate AI Fantasy Team", type="primary"):
        with st.spinner("Running ILP optimization..."):
            dm = deliveries.copy()
            dm["season"] = pd.to_numeric(dm.get("season",0), errors="coerce")
            latest = int(dm["season"].max())
            rec    = dm[dm["season"] >= latest-2]

            t1_bats  = rec[rec["batting_team"]==t1_sel]["batter"].value_counts().head(15).index.tolist()
            t2_bats  = rec[rec["batting_team"]==t2_sel]["batter"].value_counts().head(15).index.tolist()
            t1_bowls = rec[rec["bowling_team"]==t1_sel]["bowler"].value_counts().head(8).index.tolist() if "bowling_team" in rec.columns else []
            t2_bowls = rec[rec["bowling_team"]==t2_sel]["bowler"].value_counts().head(8).index.tolist() if "bowling_team" in rec.columns else []

            all_p = list(set(t1_bats+t2_bats+t1_bowls+t2_bowls))
            fp_df = estimate_player_fp(deliveries, matches, all_p, recent_n=5)
            fp_df = fp_df[fp_df["fp_total"]>0].copy()
            if fp_df.empty:
                fp_df = pd.DataFrame({"player":all_p,"fp_batting":30.0,"fp_bowling":10.0,"fp_total":40.0})

            def get_team(p):
                return t1_sel if p in t1_bats or p in t1_bowls else t2_sel
            fp_df["team"] = fp_df["player"].apply(get_team)

            bowl_set = set(t1_bowls+t2_bowls)
            bat_set  = set(t1_bats+t2_bats)
            ar_set   = bowl_set & bat_set

            def assign_role(p):
                if p in ar_set: return "AR"
                if p in bowl_set: return "BOWL"
                return "BAT"
            fp_df["role"] = fp_df["player"].apply(assign_role)

            t1_wk = [p for p in t1_bats if fp_df[fp_df["player"]==p]["role"].values[0]=="BAT"]
            t2_wk = [p for p in t2_bats if fp_df[fp_df["player"]==p]["role"].values[0]=="BAT"]
            for p in ([t1_wk[0]] if t1_wk else []) + ([t2_wk[0]] if t2_wk else []):
                fp_df.loc[fp_df["player"]==p,"role"] = "WK"

            if "AR" not in fp_df["role"].values:
                best = fp_df.sort_values("fp_total",ascending=False).iloc[0]["player"]
                fp_df.loc[fp_df["player"]==best,"role"] = "AR"

            mu  = fp_df["fp_total"].mean()
            std = max(fp_df["fp_total"].std(), 1.0)
            fp_df["credits"] = (9.0+(fp_df["fp_total"]-mu)/std*0.8).clip(7.5,11.5)

            result = optimize_team_ilp(fp_df, strategy=strategy)

        if "error" in result:
            st.error(f"Optimization failed: {result['error']}")
            return

        tdf = pd.DataFrame(result["team"])
        captain = result["captain"]
        vc      = result["vice_captain"]

        # ── Summary metrics ──
        div_score = diversity_score(tdf)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Est. FP", f"{result['total_fp']:.0f}")
        c2.metric("Credits Used", f"{result['total_credits']:.1f}/100")
        c3.metric("Team Diversity", f"{div_score:.0f}/10")
        c4.metric("Strategy", strategy.title())

        c1,c2 = st.columns(2)
        c1.success(f"👑 Captain: **{captain}** (+2× FP = {tdf[tdf['player']==captain]['fp_total'].values[0]*2:.0f} pts)")
        c2.info(f"🥈 Vice-Captain: **{vc}** (+1.5× FP = {tdf[tdf['player']==vc]['fp_total'].values[0]*1.5:.0f} pts)")

        # ── Team table ──
        section("🏏 Your Optimized Fantasy XI")
        tdf["Tag"] = tdf["player"].apply(lambda p: "👑 C" if p==captain else "🥈 VC" if p==vc else "")
        role_icons = {"WK":"🧤 WK","BAT":"🏏 BAT","AR":"⚡ AR","BOWL":"🎳 BOWL"}
        tdf["Role"] = tdf["role"].map(role_icons)
        st.dataframe(tdf[["player","team","Role","credits","fp_total","Tag"]].rename(columns={
            "player":"Player","team":"Team","credits":"Credits","fp_total":"Est. FP","Tag":""}),
            hide_index=True, use_container_width=True)

        # ── FP chart ──
        fig = px.bar(tdf.sort_values("fp_total"), x="fp_total", y="player",
                      orientation="h", color="role",
                      color_discrete_map={"WK":"#58a6ff","BAT":"#3fb950","AR":"#e3b341","BOWL":"#f0883e"})
        fig.update_layout(template="plotly_dark", height=380, xaxis_title="Est. Fantasy Points")
        st.plotly_chart(fig, use_container_width=True)

        # ── WHY THIS PLAYER reasoning ──
        section("🧠 WHY THIS PLAYER? — AI Reasoning")
        for _, row in tdf.sort_values("fp_total",ascending=False).iterrows():
            reasons = fantasy_player_reasoning(row, captain, vc)
            for r in reasons:
                kind = "success" if row["player"]==captain else "info"
                insight_card(f"**{row['player']}** ({row['role']}) — {r}", kind)

        # ── Risk vs Reward ──
        section("⚖️ Risk vs Reward Analysis")
        sorted_tdf = tdf.sort_values("fp_total",ascending=False)
        top3  = sorted_tdf.head(3)["player"].tolist()
        bot3  = sorted_tdf.tail(3)["player"].tolist()
        insight_card(f"🔒 **Safe Anchors** (low-risk, consistent): {', '.join(top3)}", "success")
        insight_card(f"🎯 **Differential picks** (high-risk, low-ownership): {', '.join(bot3)}", "warning")

        if strategy == "maximize":
            decision_card("🚀 **Aggressive strategy selected** — high-ceiling team. Best for H2H or small leagues where you need to top the leaderboard.")
        elif strategy == "safe":
            decision_card("🛡️ **Safe strategy** — consistent players, lower variance. Best for large leagues where finishing in top 20% is the goal.")
        else:
            decision_card("🎯 **Differentiated strategy** — low-ownership players selected. Best for grand leagues where you need unique picks to stand out from 100K+ teams.")

        # ── Alternative picks ──
        section("🔄 Alternative Picks (Swap Suggestions)")
        remaining = fp_df[~fp_df["player"].isin(tdf["player"])].sort_values("fp_total",ascending=False)
        if not remaining.empty:
            for _, alt in remaining.head(3).iterrows():
                # Find the lowest FP player with same role to swap
                same_role = tdf[tdf["role"]==alt["role"]].sort_values("fp_total")
                if not same_role.empty:
                    swap_out = same_role.iloc[0]["player"]
                    swap_fp  = same_role.iloc[0]["fp_total"]
                    insight_card(
                        f"🔄 **Swap:** Replace **{swap_out}** ({swap_fp:.0f} fp) → **{alt['player']}** ({alt['fp_total']:.0f} fp) | Role: {alt['role']}",
                        "info"
                    )

        # ── Ownership strategy ──
        section("📊 Ownership Strategy")
        insight_card(f"👑 **Captain {captain}** — High-ownership pick. Most teams will choose this. Safe but low differential.", "info")
        top_diff = remaining.head(1)
        if not top_diff.empty:
            dp = top_diff.iloc[0]
            insight_card(f"🎯 **Differential Captain Option: {dp['player']}** ({dp['fp_total']:.0f} fp) — Low-ownership, high upside for grand leagues.", "warning")
        insight_card(f"📊 **Team Diversity Score: {div_score:.0f}/10** — {'Well-balanced team' if div_score>6 else 'Consider diversifying roles/teams'}", "success" if div_score>6 else "warning")


# ═══════════════════════════════════════════════════════
# PAGE 15 UPGRADED: PLAYER RANKINGS
# ═══════════════════════════════════════════════════════
def page_rankings_v2(matches, deliveries):
    st.title("🏅 Player Power Rankings")
    headline_card(
        "🧠 Contextual + Impact Rankings",
        "Impact score • Venue-specific • Opponent-specific • Aggressive vs Anchor clustering"
    )

    seasons = sorted(matches["season"].dropna().unique().tolist(),reverse=True)
    c1,c2,c3 = st.columns(3)
    season   = c1.selectbox("Season",["Overall"]+[str(int(s)) for s in seasons])
    venue    = c2.selectbox("Venue (optional)",["All"]+sorted(matches["venue"].dropna().unique().tolist()))
    opp      = c3.selectbox("vs Team (optional)",["All"]+sorted(set(matches["team1"].dropna().tolist()+matches["team2"].dropna().tolist())))

    dm = deliveries.copy()
    dm["season"] = pd.to_numeric(dm.get("season",0), errors="coerce")
    dm["date"]   = pd.to_datetime(dm.get("date",""), errors="coerce")

    if season != "Overall":
        dm = dm[dm["season"]==int(season)]

    if venue != "All" and "venue" in matches.columns:
        venue_matches = matches[matches["venue"]==venue]["match_id"].tolist()
        dm = dm[dm["match_id"].isin(venue_matches)]

    if opp != "All":
        dm = dm[dm["bowling_team"]==opp] if "bowling_team" in dm.columns else dm

    tab1,tab2 = st.tabs(["🏏 Batsmen","🎳 Bowlers"])

    with tab1:
        bat = dm.groupby("batter").agg(
            runs=("batsman_runs","sum"), balls=("is_legal","sum"),
            innings=("match_id","nunique"), wkts=("is_wicket","sum"),
            fours=("batsman_runs",lambda x:(x==4).sum()),
            sixes=("batsman_runs",lambda x:(x==6).sum()),
        ).reset_index()
        bat["sr"]  = (bat["runs"]/bat["balls"].replace(0,1)*100).round(2)
        bat["avg"] = (bat["runs"]/bat["wkts"].replace(0,1)).round(2)
        bat["boundary_pct"] = ((bat["fours"]*4+bat["sixes"]*6)/bat["runs"].replace(0,1)).round(3)
        min_inn = 3 if season!="Overall" else 10
        bat = bat[bat["innings"]>=min_inn]

        # ── IMPACT SCORE ──
        bat["impact"] = bat.apply(lambda r:
            impact_score(r["runs"]/r["innings"], r["sr"], r["avg"], matches=r["innings"]), axis=1)

        # ── Play style clustering ──
        bat["style"], bat["style_desc"] = zip(*bat.apply(
            lambda r: classify_play_style(r["sr"], r["avg"], r["boundary_pct"]), axis=1))

        bat = bat.sort_values("impact",ascending=False).head(25).reset_index(drop=True)
        bat.index += 1

        # Style filter
        style_filter = st.multiselect("Filter by Play Style",
                                        sorted(bat["style"].unique().tolist()),
                                        key="style_filter")
        if style_filter:
            bat = bat[bat["style"].isin(style_filter)]

        st.dataframe(
            bat[["batter","innings","runs","avg","sr","fours","sixes","impact","style"]].rename(columns={
                "batter":"Player","innings":"Inn","runs":"Runs","avg":"Avg",
                "sr":"SR","fours":"4s","sixes":"6s","impact":"Impact","style":"Style"
            }).style.background_gradient(subset=["Impact","Runs","SR"],cmap="Blues"),
            use_container_width=True)

        # ── Bubble chart: SR vs Avg with Impact size ──
        section("🫧 Strike Rate vs Average — Impact Bubble Chart")
        fig = px.scatter(bat.head(20), x="sr", y="avg",
                          size="impact", color="style", text="batter",
                          labels={"sr":"Strike Rate","avg":"Average","impact":"Impact Score"},
                          title="Contextual Player Rankings")
        fig.update_traces(textposition="top center")
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

        section("🧠 Play Style Clusters")
        for style in bat["style"].unique():
            players_in_style = bat[bat["style"]==style]["batter"].head(3).tolist()
            desc = bat[bat["style"]==style]["style_desc"].iloc[0] if not bat[bat["style"]==style].empty else ""
            insight_card(f"**{style}**: {', '.join(players_in_style)} — {desc}", "info")

    with tab2:
        bowl = dm.groupby("bowler").agg(
            wkts=("is_wicket","sum"), runs=("total_runs","sum"),
            balls=("is_legal","sum"), innings=("match_id","nunique")
        ).reset_index()
        bowl["econ"] = (bowl["runs"]/(bowl["balls"]/6).replace(0,1)).round(2)
        bowl["avg"]  = (bowl["runs"]/bowl["wkts"].replace(0,1)).round(2)
        bowl["sr"]   = (bowl["balls"]/bowl["wkts"].replace(0,1)).round(2)
        bowl = bowl[bowl["innings"]>=min_inn]
        bowl["impact"] = bowl.apply(lambda r:
            impact_score(0, 120, r["avg"], wickets=r["wkts"]/r["innings"],
                          economy=r["econ"]), axis=1)
        bowl = bowl.sort_values("impact",ascending=False).head(25).reset_index(drop=True)
        bowl.index += 1
        st.dataframe(
            bowl[["bowler","innings","wkts","econ","avg","sr","impact"]].rename(columns={
                "bowler":"Player","innings":"Inn","wkts":"Wkts",
                "econ":"Econ","avg":"Avg","sr":"SR","impact":"Impact"
            }).style.background_gradient(subset=["Impact","Wkts"],cmap="Greens")
              .background_gradient(subset=["Econ"],cmap="RdYlGn_r"),
            use_container_width=True)

        if venue != "All":
            insight_card(f"📍 Rankings filtered for **{venue}** — showing contextual performance at this ground", "info")
        if opp != "All":
            insight_card(f"🎯 Rankings filtered **vs {opp}** — showing opponent-specific performance", "info")


# ═══════════════════════════════════════════════════════
# PAGE 16 UPGRADED: VENUE ANALYSIS
# ═══════════════════════════════════════════════════════
def page_venue_analysis_v2(matches, deliveries):
    st.title("🏟️ Deep Venue Analysis")
    headline_card(
        "🧠 AI Pitch Intelligence",
        "Pitch type classification • Best strategy • Player suitability • Toss insights"
    )

    venues = sorted(matches["venue"].dropna().unique().tolist())
    venue  = st.selectbox("Select Venue", venues)
    vm     = matches[matches["venue"]==venue]
    vd     = deliveries[deliveries["match_id"].isin(vm["match_id"])]

    if vm.empty:
        st.warning("No data for this venue.")
        return

    # ── Stats ──
    avg_score  = vd.groupby("match_id")["total_runs"].sum().mean()
    avg_wkts   = vd.groupby("match_id")["is_wicket"].sum().mean()

    # Phase run rates
    pp_rr, dth_rr = 8.0, 9.0
    if "phase" in vd.columns:
        for phase, var in [("powerplay","pp_rr"),("death","dth_rr")]:
            sub = vd[vd["phase"]==phase]
            lg  = sub[sub["is_legal"]==1]
            if len(lg)>0:
                rr = sub["total_runs"].sum()/(len(lg)/6)
                if phase=="powerplay": pp_rr=rr
                else: dth_rr=rr

    pitch_type, strategy = classify_pitch(avg_score, avg_wkts, pp_rr, dth_rr)

    # ── Pitch type banner ──
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Pitch Type", pitch_type)
    c2.metric("Avg Score", f"{avg_score:.0f} runs")
    c3.metric("Avg Wickets", f"{avg_wkts:.1f}")
    c4.metric("Total Matches", len(vm))

    st.success(f"**🏟️ {venue}:** {pitch_type} | **Best Strategy:** {strategy}")

    # ── Toss + bat/chase ──
    section("🎯 Toss & Batting Strategy Intelligence")
    c1,c2 = st.columns(2)
    with c1:
        if "bat_first_won" in vm.columns:
            bf = vm["bat_first_won"].mean()*100
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=bf,
                title={"text":"Bat First Win %"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#58a6ff"}}
            ))
            fig.update_layout(template="plotly_dark",height=250)
            st.plotly_chart(fig, use_container_width=True)
            if bf > 55:
                insight_card(f"✅ **Bat first wins {bf:.0f}%** at {venue} — toss winner should bat", "success")
            else:
                insight_card(f"🏃 **Chase wins {100-bf:.0f}%** at {venue} — field first if you win the toss", "info")

    with c2:
        if "phase" in vd.columns:
            ph_agg = vd.groupby("phase").agg(runs=("total_runs","sum"),balls=("is_legal","sum")).reset_index()
            ph_agg["rr"] = ph_agg["runs"]/ph_agg["balls"].replace(0,1)*6
            fig = px.bar(ph_agg,x="phase",y="rr",color="rr",
                          color_continuous_scale="RdYlGn",title="Run Rate by Phase",
                          labels={"phase":"Phase","rr":"RPO"})
            fig.update_layout(template="plotly_dark",height=250,coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Player Suitability ──
    section("🏆 Player Suitability — Who Performs Best Here?")
    if not vd.empty:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Top Batsmen at this Venue**")
            bat_v = vd.groupby("batter").agg(
                runs=("batsman_runs","sum"), balls=("is_legal","sum"),
                innings=("match_id","nunique")
            ).reset_index()
            bat_v["sr"] = bat_v["runs"]/bat_v["balls"].replace(0,1)*100
            bat_v = bat_v[bat_v["innings"]>=3].sort_values("runs",ascending=False).head(8)
            for _,r in bat_v.iterrows():
                insight_card(f"🏏 **{r['batter']}** — {r['runs']} runs @ SR {r['sr']:.0f} in {r['innings']} innings", "success")

        with c2:
            st.markdown("**Top Bowlers at this Venue**")
            bowl_v = vd.groupby("bowler").agg(
                wkts=("is_wicket","sum"), runs=("total_runs","sum"),
                balls=("is_legal","sum"), innings=("match_id","nunique")
            ).reset_index()
            bowl_v["econ"] = bowl_v["runs"]/(bowl_v["balls"]/6).replace(0,1)
            bowl_v = bowl_v[bowl_v["innings"]>=3].sort_values("wkts",ascending=False).head(8)
            for _,r in bowl_v.iterrows():
                insight_card(f"🎳 **{r['bowler']}** — {r['wkts']} wkts @ {r['econ']:.1f} econ in {r['innings']} innings", "info")

    # ── Top teams at venue ──
    section("🏆 Most Successful Teams at This Venue")
    tw = vm["winner"].value_counts().reset_index()
    tw.columns = ["team","wins"]
    tm_ = pd.concat([vm["team1"],vm["team2"]]).value_counts().reset_index()
    tm_.columns = ["team","played"]
    ts  = tw.merge(tm_,on="team")
    ts["win_pct"] = ts["wins"]/ts["played"]*100
    ts["suitability"] = ts["win_pct"].apply(lambda p: "✅ High Suitability" if p>60 else "⚠️ Average" if p>40 else "❌ Poor Record")

    for _,r in ts.sort_values("wins",ascending=False).head(5).iterrows():
        insight_card(f"**{r['team']}** — {r['wins']} wins from {r['played']} games ({r['win_pct']:.0f}%) → {r['suitability']}", "success" if r["win_pct"]>60 else "info")


# ═══════════════════════════════════════════════════════
# PAGE 17 UPGRADED: PLAYER SIMILARITY
# ═══════════════════════════════════════════════════════
def page_player_similarity_v2(matches, deliveries):
    st.title("🧬 Player Similarity Engine")
    headline_card(
        "🤖 AI Role Matching & Replacement Finder",
        "Play style explanation • Role matching • Best replacement player • Similarity reasoning"
    )

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    players = sorted(set(deliveries["batter"].dropna().tolist()+deliveries["bowler"].dropna().tolist()))
    c1,c2 = st.columns([3,1])
    player = c1.selectbox("Find players similar to:",[""] + players)
    role   = c2.radio("Role",["Batting","Bowling"],horizontal=True)
    top_n  = st.slider("Top N",3,15,8)
    if not player:
        st.info("Select a player.")
        return

    dm = deliveries.copy()
    if role == "Batting":
        feat = dm.groupby("batter").agg(
            avg_runs=("batsman_runs","mean"), total_runs=("batsman_runs","sum"),
            balls=("is_legal","sum"), innings=("match_id","nunique"),
            sixes=("batsman_runs",lambda x:(x==6).sum()),
            fours=("batsman_runs",lambda x:(x==4).sum()),
            wkts=("is_wicket","sum"),
        ).reset_index().rename(columns={"batter":"player"})
        feat["sr"]  = feat["total_runs"]/feat["balls"].replace(0,1)*100
        feat["avg"] = feat["total_runs"]/feat["wkts"].replace(0,1)
        feat["boundary_pct"] = (feat["fours"]*4+feat["sixes"]*6)/feat["total_runs"].replace(0,1)
        feat_cols = ["avg_runs","sr","avg","sixes","fours","boundary_pct"]
        feat = feat[feat["innings"]>=5]
    else:
        feat = dm.groupby("bowler").agg(
            total_wkts=("is_wicket","sum"), total_runs=("total_runs","sum"),
            balls=("is_legal","sum"), innings=("match_id","nunique"),
        ).reset_index().rename(columns={"bowler":"player"})
        feat["econ"] = feat["total_runs"]/(feat["balls"]/6).replace(0,1)
        feat["sr"]   = feat["balls"]/feat["total_wkts"].replace(0,1)
        feat["avg"]  = feat["total_runs"]/feat["total_wkts"].replace(0,1)
        feat_cols = ["total_wkts","econ","sr","avg"]
        feat = feat[feat["innings"]>=5]

    if player not in feat["player"].values:
        st.warning(f"Not enough data for {player} (needs 5+ innings).")
        return

    feat = feat.dropna(subset=feat_cols).reset_index(drop=True)
    X = StandardScaler().fit_transform(feat[feat_cols].fillna(0).values)
    idx = feat[feat["player"]==player].index[0]
    sims = cosine_similarity([X[idx]], X)[0]
    feat["similarity"] = sims
    sim_df = feat[feat["player"]!=player].sort_values("similarity",ascending=False).head(top_n)

    # ── Target player profile ──
    target = feat[feat["player"]==player].iloc[0]
    if role == "Batting":
        t_style, t_desc = classify_play_style(
            target.get("sr",120), target.get("avg",25), target.get("boundary_pct",0.3))
        headline_card(f"🎯 {player} — {t_style}", t_desc, "#f0883e")
    else:
        t_style = "🎳 BOWLER"
        t_desc  = f"Economy: {target.get('econ',8):.1f} | SR: {target.get('sr',20):.1f}"
        headline_card(f"🎯 {player} — {t_style}", t_desc, "#f0883e")

    # ── Similarity chart ──
    fig = px.bar(sim_df, x="similarity", y="player", orientation="h",
                  color="similarity", color_continuous_scale="Blues",
                  labels={"similarity":"Similarity Score","player":"Player"})
    fig.update_layout(template="plotly_dark",height=380,yaxis={"categoryorder":"total ascending"},
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Role matching + reasoning ──
    section("🧠 Play Style Matching & Role Explanation")
    for _, r in sim_df.iterrows():
        p = r["player"]
        sim = r["similarity"]
        match_pct = round(sim * 100, 0)

        if role == "Batting":
            p_style, p_desc = classify_play_style(
                r.get("sr",120), r.get("avg",25), r.get("boundary_pct",0.3))
            role_match = "✅ Same Style" if p_style == t_style else "📊 Similar Profile"
            insight_card(
                f"**{p}** ({match_pct:.0f}% match) — {p_style} | {role_match} | {p_desc}",
                "success" if sim > 0.9 else "info"
            )
        else:
            insight_card(
                f"**{p}** ({match_pct:.0f}% match) — Econ: {r.get('econ',0):.1f} | SR: {r.get('sr',0):.0f}",
                "success" if sim > 0.9 else "info"
            )

    # ── Best replacement ──
    section("🔄 Best Replacement Player")
    if not sim_df.empty:
        best_rep = sim_df.iloc[0]
        p = best_rep["player"]
        sim = best_rep["similarity"]
        if role == "Batting":
            b_style, b_desc = classify_play_style(best_rep.get("sr",120), best_rep.get("avg",25), best_rep.get("boundary_pct",0.3))
            decision_card(f"🔄 **Best Replacement for {player}: {p}** ({sim*100:.0f}% similarity) — {b_style} | {b_desc}")
        else:
            decision_card(f"🔄 **Best Replacement for {player}: {p}** ({sim*100:.0f}% similarity) | Same bowling profile")

        insight_card(f"💡 **Why {p}?** They share a nearly identical statistical fingerprint to {player} — suitable drop-in across same role and match context.", "info")


# ═══════════════════════════════════════════════════════
# PAGE 18 UPGRADED: ABOUT & MODEL PAGE
# ═══════════════════════════════════════════════════════
def page_about_v2(load_metrics):
    st.title("📖 About CricketBrain AI")
    headline_card(
        "🏏 CricketBrain AI — Production-Grade Cricket Decision Intelligence",
        "Not just analytics. A system that tells you WHAT TO DO, WHY IT MATTERS, and HOW CONFIDENT IT IS."
    )

    # ── WHY THIS IS DIFFERENT ──
    section("🚀 Why CricketBrain AI is Different")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        **Other dashboards:**
        - Show data ❌
        - Static charts ❌
        - No explanations ❌
        - No decisions ❌
        """)
    with cols[1]:
        st.markdown("""
        **CricketBrain AI:**
        - Generates decisions ✅
        - Explains WHY ✅
        - Simulates scenarios ✅
        - Quantifies confidence ✅
        """)
    with cols[2]:
        st.markdown("""
        **Real-world impact:**
        - Fantasy team optimization ✅
        - Match strategy advisor ✅
        - Playoff probability ✅
        - Risk scoring ✅
        """)

    # ── REAL-WORLD IMPACT ──
    section("💥 Real-World Impact")
    metrics_data = load_metrics()
    c1,c2,c3,c4 = st.columns(4)
    best_auc = metrics_data.get("calibrated_auc", metrics_data.get("XGBoost",{}).get("roc_auc",0.0))
    baseline_auc = 0.50
    improvement = ((best_auc - baseline_auc)/baseline_auc*100) if best_auc > 0 else 0

    c1.metric("Model AUC", f"{best_auc:.4f}" if best_auc else "Train first")
    c2.metric("vs Baseline (50% guess)", f"+{improvement:.1f}%" if improvement else "N/A",
               delta_color="normal")
    c3.metric("Fantasy Optimizer", "ILP (PuLP)", help="Integer Linear Programming optimizer")
    c4.metric("Monte Carlo Runs", "10,000 / match", help="Simulations per prediction")

    insight_card(f"📊 **Baseline model (random guess):** 50% AUC | **CricketBrain AI:** {best_auc:.3f} AUC — a meaningful improvement in match prediction accuracy", "success")
    insight_card("💰 **Fantasy optimizer** uses Integer Linear Programming to solve a constrained optimization problem — guaranteed optimal team selection vs manual guessing", "success")
    insight_card("🎲 **Monte Carlo engine** runs 10,000 simulations to give confidence intervals — not just a single probability, but a full distribution", "info")
    insight_card("🔍 **SHAP explainability** means every prediction has a human-readable reason — audit any decision in seconds", "info")

    # ── KEY INNOVATIONS ──
    section("🏆 Key Technical Innovations")
    innovations = [
        ("🚫 Zero Data Leakage", "All features use shift(1).rolling() — model has never seen the future during training"),
        ("⏱️ TimeSeriesSplit CV", "5-fold temporal cross-validation preserving chronological order — no future contamination"),
        ("🎯 Stacking Ensemble", "XGBoost + LightGBM + CatBoost meta-learned by Logistic Regression — best of all worlds"),
        ("📊 Calibrated Probabilities", "Isotonic regression calibration — probabilities are meaningful, not just rankings"),
        ("🧠 Intelligence Engine", "Rule-based + statistical hybrid insight system covering 8 decision domains"),
        ("⚡ ILP Fantasy Optimizer", "PuLP linear programming solves Dream11 constraints exactly — not greedy approximation"),
    ]
    c1,c2 = st.columns(2)
    for i,(title,desc) in enumerate(innovations):
        col = c1 if i%2==0 else c2
        with col:
            insight_card(f"**{title}** — {desc}", "success")

    # ── SYSTEM FLOW ──
    section("🔄 System Flow — Data to Decision")
    st.markdown("""
    ```
    📁 Raw IPL.csv (2008–2025)
           ↓
    🧹 ETL Pipeline (data_cleaning.py)
       • Team normalization • Date parsing • Phase derivation
           ↓
    ⚙️ Feature Engineering (feature_engine.py)
       • 70+ features • Rolling stats • H2H • Zero leakage
           ↓
    🤖 ML Training (train.py)
       • XGBoost → LightGBM → CatBoost → Stacking → Calibration
       • Optuna (50-trial hyperparameter search)
       • SHAP explainability
           ↓
    🎲 Monte Carlo Simulation (monte_carlo.py)
       • 10,000 simulations per match
       • Win % + 95% CI + Score distributions
           ↓
    🧠 Intelligence Engine (intelligence.py)
       • Form classification • Risk scoring • Turning points
       • Pitch classification • Playoff probability
           ↓
    💰 Fantasy Optimizer (fantasy_optimizer.py)
       • ILP (PuLP) • 3 strategies • Alternative picks
           ↓
    📊 18-Page Streamlit Dashboard
       • Every screen answers: "So what? What do I do?"
    ```
    """)

    # ── MODEL METRICS ──
    section("🤖 Model Performance")
    if not metrics_data:
        st.warning("Train models first: `python ml/train.py`")
    else:
        model_names = ["RandomForest","XGBoost","LightGBM","CatBoost","Stacking"]
        rows = []
        for k in model_names:
            if k in metrics_data and isinstance(metrics_data[k], dict):
                v = metrics_data[k]
                beat_baseline = float(v.get("roc_auc",v.get("cv_mean",0))) - 0.5
                rows.append({
                    "Model": k,
                    "ROC-AUC": f"{v.get('roc_auc',v.get('cv_mean',0)):.4f}",
                    "Accuracy": f"{v.get('accuracy',0):.4f}",
                    "Log Loss": f"{v.get('log_loss',0):.4f}",
                    "Brier Score": f"{v.get('brier_score',0):.4f}",
                    "CV AUC": f"{v.get('cv_mean',0):.4f}±{v.get('cv_std',0):.4f}",
                    "vs Baseline": f"+{beat_baseline:.4f}"
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        best = metrics_data.get("best_model","N/A")
        cal  = metrics_data.get("calibrated_auc",0)
        st.success(f"🏆 Best: **{best}** | Calibrated AUC: **{cal:.4f}** | Improvement over 50% baseline: **+{(cal-0.5)*100:.1f}%**")

    section("🚀 Quick Start")
    st.code("""
pip install -r requirements.txt
# Place IPL.csv in data/raw/IPL.csv
python etl/data_cleaning.py
python etl/feature_engine.py
python ml/train.py          # ~5-15 min
streamlit run app/app.py
    """, language="bash")