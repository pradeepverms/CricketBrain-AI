"""
CricketBrain AI — Main Streamlit App
18-page production dashboard | Full navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, json, warnings, joblib
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Local imports ──
from simulation.monte_carlo import run_simulation, simulate_player_performance
from optimizer.fantasy_optimizer import generate_fantasy_teams, optimize_team_ilp
from ml.weakness_detector import batsman_weakness, bowler_weakness, matchup_matrix
from etl.insight_generator import (
    team_form_insights, toss_advisor, bowling_strategy,
    player_form_insights, match_preview_insights, viral_insight
)

# ── Upgraded intelligence pages ──
from app.upgraded_pages import (
    page_win_probability_v2, page_run_heatmap_v2, page_season_table_v2,
    page_form_tracker_v2, page_breakout_v2, page_fantasy_v2,
    page_rankings_v2, page_venue_analysis_v2, page_player_similarity_v2,
    page_about_v2
)
from optimizer.fantasy_optimizer import optimize_team_ilp, estimate_player_fp
from app.phase2_pages import (
    page_decision_engine, page_pressure_momentum, page_player_deep,
    page_fantasy_elite, page_venue_advanced, page_backtesting
)
# Phase 2-12 upgraded pages



# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricketBrain AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0d1117; }
    .main { background: #0d1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2d3d, #162032);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .warning-card {
        background: #161b22;
        border-left: 4px solid #f0883e;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    .success-card {
        background: #161b22;
        border-left: 4px solid #3fb950;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.95rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #58a6ff;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #30363d;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        font-size: 1.8rem;
        font-weight: 900;
        color: #58a6ff;
        letter-spacing: 1px;
    }
    .sidebar-tagline {
        text-align: center;
        font-size: 0.75rem;
        color: #8b949e;
        margin-bottom: 1rem;
    }
    .stRadio > label { font-weight: 600; }
    .stSelectbox > label { font-weight: 600; }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE, "data")
CLEANED   = os.path.join(DATA_DIR, "cleaned")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SHAP_DIR  = os.path.join(DATA_DIR, "shap")
FEAT_DIR  = os.path.join(DATA_DIR, "features")

# ─────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_matches():
    p = os.path.join(CLEANED, "matches.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Guarantee date column exists and is parsed
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    # Guarantee 'season' column always exists
    if "season" not in df.columns:
        if "year" in df.columns:
            df["season"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        elif "date" in df.columns:
            df["season"] = pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64")
        else:
            df["season"] = 2024
    else:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    # Force to plain int (fill missing with 0, then cast)
    df["season"] = df["season"].fillna(0).astype(int)
    df.loc[df["season"] == 0, "season"] = None  # restore NaN for display
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    # Guarantee 'bat_first_won' exists
    if "bat_first_won" not in df.columns:
        df["bat_first_won"] = 0
    # Guarantee 'team1_won' exists
    if "team1_won" not in df.columns:
        df["team1_won"] = 0
    # Guarantee 'batting_first' exists
    if "batting_first" not in df.columns:
        df["batting_first"] = df.get("team1", "")
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=600)
def load_deliveries():
    p = os.path.join(CLEANED, "deliveries.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p)
    # Guarantee legal_ball column
    if "legal_ball" not in d.columns:
        if "is_legal" in d.columns:
            d["legal_ball"] = d["is_legal"]
        else:
            d["legal_ball"] = (d.get("is_wide", 0) + d.get("is_noball", 0) == 0).astype(int)
    # Guarantee is_legal column
    if "is_legal" not in d.columns:
        d["is_legal"] = d["legal_ball"]
    # Guarantee is_wicket is numeric
    if "is_wicket" in d.columns:
        d["is_wicket"] = pd.to_numeric(d["is_wicket"], errors="coerce").fillna(0).astype(int)
    # Guarantee total_runs is numeric
    for col in ["batsman_runs", "total_runs", "extra_runs"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
    # Guarantee phase column exists
    if "phase" not in d.columns and "over_num" in d.columns:
        d["phase"] = pd.cut(
            pd.to_numeric(d["over_num"], errors="coerce").fillna(0),
            bins=[-1, 5, 14, 20], labels=["powerplay", "middle", "death"]
        )
    return d

@st.cache_data(ttl=3600)
def load_model():
    p = os.path.join(MODEL_DIR, "best_model.pkl")
    return joblib.load(p) if os.path.exists(p) else None

@st.cache_data(ttl=3600)
def load_feature_names():
    p = os.path.join(MODEL_DIR, "feature_names.pkl")
    return joblib.load(p) if os.path.exists(p) else []

@st.cache_data(ttl=3600)
def load_metrics():
    p = os.path.join(MODEL_DIR, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {}


@st.cache_data(ttl=600)
def load_match_features():
    p = os.path.join(FEAT_DIR, "match_features.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

def safe_merge(deliveries, matches, extra_cols=None):
    """
    Safely merge deliveries with match metadata.
    - Forces match_id to string in both frames (avoids type mismatch)
    - Guarantees season is always a clean int column
    - Guarantees date column always exists (fills missing with NaT)
    """
    want = ["match_id", "date", "season"] + (extra_cols or [])
    cols = [c for c in want if c in matches.columns]
    m = matches[cols].copy()
    m["match_id"] = m["match_id"].astype(str)

    # Ensure date exists in matches slice
    if "date" not in m.columns:
        if "date" in matches.columns:
            m["date"] = pd.to_datetime(matches["date"], errors="coerce")
        else:
            m["date"] = pd.NaT

    if "season" in m.columns:
        m["season"] = pd.to_numeric(m["season"], errors="coerce").fillna(0).astype(int)

    d = deliveries.copy()
    d["match_id"] = d["match_id"].astype(str)
    merged = d.merge(m, on="match_id", how="left")

    # Always guarantee date column exists after merge
    if "date" not in merged.columns:
        merged["date"] = pd.NaT
    else:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    if "season" in merged.columns:
        merged["season"] = pd.to_numeric(merged["season"], errors="coerce").fillna(0).astype(int)

    return merged

@st.cache_data(ttl=600)
def get_teams(matches):
    if matches.empty:
        return []
    return sorted(set(matches["team1"].dropna().tolist() + matches["team2"].dropna().tolist()))

@st.cache_data(ttl=600)
def get_players(deliveries):
    if deliveries.empty:
        return []
    batters = deliveries["batter"].dropna().unique().tolist()
    bowlers = deliveries["bowler"].dropna().unique().tolist()
    return sorted(set(batters + bowlers))

@st.cache_data(ttl=600)
def get_venues(matches):
    if matches.empty:
        return []
    return sorted(matches["venue"].dropna().unique().tolist())

# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def check_data(matches, deliveries):
    if matches.empty or deliveries.empty:
        st.error("""
        ⚠️ **Data not found!**

        Please run the pipeline first:
        ```bash
        python etl/data_cleaning.py
        python etl/feature_engine.py
        ```
        Then place `IPL.csv` in `data/raw/IPL.csv`
        """)
        return False
    return True

def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight_card(text: str, kind: str = "info"):
    cls = {"info": "insight-card", "warning": "warning-card", "success": "success-card"}.get(kind, "insight-card")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PAGE 1: LEAGUE OVERVIEW
# ─────────────────────────────────────────────────────────
def page_overview():
    st.title("🏆 IPL League Overview")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    seasons = sorted(matches["season"].dropna().unique().tolist())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Matches",  f"{len(matches):,}")
    col2.metric("Seasons",        f"{len(seasons)}")
    col3.metric("Teams",          f"{len(get_teams(matches))}")
    col4.metric("Total Balls",    f"{len(deliveries):,}")
    col5.metric("Total Runs",     f"{int(deliveries['total_runs'].sum()):,}")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        section("📅 Matches per Season")
        ms = matches.groupby("season")["match_id"].count().reset_index(name="matches")
        fig = px.bar(ms, x="season", y="matches", color="matches",
                     color_continuous_scale="Blues",
                     labels={"season": "Season", "matches": "Matches"})
        fig.update_layout(template="plotly_dark", height=310, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section("🏆 Most IPL Wins (All-time)")
        wc = matches["winner"].dropna().value_counts().reset_index()
        wc.columns = ["team", "wins"]
        fig = px.bar(wc.head(10), x="wins", y="team", orientation="h",
                     color="wins", color_continuous_scale="Viridis",
                     labels={"wins": "Wins", "team": ""})
        fig.update_layout(template="plotly_dark", height=310, showlegend=False,
                           coloraxis_showscale=False,
                           yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        section("🎯 Toss Decision Distribution")
        if "toss_decision" in matches.columns:
            td = matches["toss_decision"].value_counts()
            fig = px.pie(values=td.values, names=td.index, hole=0.45,
                         color_discrete_sequence=["#58a6ff", "#f0883e"])
            fig.update_layout(template="plotly_dark", height=290)
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        section("📊 Bat First vs Chase Win Rate — by Season")
        if "bat_first_won" in matches.columns:
            bf = matches.groupby("season")["bat_first_won"].mean().reset_index()
            bf.columns = ["season", "bat_first_win_rate"]
            bf["chase_win_rate"] = 1 - bf["bat_first_win_rate"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bf["season"], y=bf["bat_first_win_rate"] * 100,
                                      name="Bat First Win%", mode="lines+markers",
                                      line=dict(color="#58a6ff", width=2)))
            fig.add_trace(go.Scatter(x=bf["season"], y=bf["chase_win_rate"] * 100,
                                      name="Chase Win%", mode="lines+markers",
                                      line=dict(color="#f0883e", width=2)))
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(template="plotly_dark", height=290, yaxis_title="Win %")
            st.plotly_chart(fig, use_container_width=True)

    section("📈 Average Match Runs by Season")
    try:
        # Force match_id to same type (str) in both frames before merging
        m_s = matches[["match_id","season"]].copy()
        m_s["match_id"] = m_s["match_id"].astype(str)
        m_s["season"]   = pd.to_numeric(m_s["season"], errors="coerce")
        m_s = m_s[m_s["season"].notna() & (m_s["season"] > 2000)].copy()
        m_s["season"]   = m_s["season"].astype(int)

        del_copy = deliveries[["match_id","total_runs"]].copy()
        del_copy["match_id"] = del_copy["match_id"].astype(str)

        dm2  = del_copy.merge(m_s, on="match_id", how="left")
        dm2  = dm2[dm2["season"].notna()].copy()
        dm2["season"] = dm2["season"].astype(int)

        if dm2.empty:
            st.info("Season data not available yet.")
        else:
            avg_r = dm2.groupby(["match_id","season"])["total_runs"].sum().reset_index()
            avg_s = avg_r.groupby("season")["total_runs"].mean().reset_index()
            fig = px.area(avg_s, x="season", y="total_runs",
                          labels={"total_runs":"Avg Match Runs","season":"Season"},
                          color_discrete_sequence=["#58a6ff"])
            fig.update_layout(template="plotly_dark", height=260)
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Run trend unavailable: {e}")

# ─────────────────────────────────────────────────────────
# PAGE 2: PLAYER PROFILE
# ─────────────────────────────────────────────────────────
def page_player_search():
    st.title("🔍 Player Search & Profile")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    players = get_players(deliveries)
    query = st.text_input("🔎 Search player name", placeholder="e.g. Virat Kohli, Jasprit Bumrah")
    if query:
        matched = [p for p in players if query.lower() in p.lower()][:30]
        if not matched:
            st.warning(f"No players found matching '{query}'")
            return
        player = st.selectbox("Select player", matched)
    else:
        player = st.selectbox("Or browse all players", [""] + players[:150])
        if not player:
            st.info("Search or select a player above.")
            return

    st.divider()
    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")
    dm = dm.sort_values("date", na_position="last")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Batting Stats", "🎳 Bowling Stats", "🧠 AI Insights", "⚠️ Weaknesses"])

    with tab1:
        bat = dm[dm["batter"] == player]
        if bat.empty:
            st.info("No batting records found.")
        else:
            bm = bat.groupby("match_id").agg(
                runs=("batsman_runs", "sum"), balls=("legal_ball", "sum"),
                fours=("batsman_runs", lambda x: (x == 4).sum()),
                sixes=("batsman_runs", lambda x: (x == 6).sum()),
                dismissed=("is_wicket", "max"),
                date=("date", "max")
            ).reset_index().sort_values("date")
            bm["sr"] = bm["runs"] / bm["balls"].replace(0, 1) * 100

            total_r  = int(bm["runs"].sum())
            career_sr = round(bm["runs"].sum() / bm["balls"].sum() * 100, 2)
            career_avg = round(bm["runs"].sum() / max(bm["dismissed"].sum(), 1), 2)
            best       = int(bm["runs"].max())

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Runs",    f"{total_r:,}")
            c2.metric("Innings",       f"{len(bm)}")
            c3.metric("Career SR",     f"{career_sr}")
            c4.metric("Career Avg",    f"{career_avg}")
            c1.metric("Best Score",    f"{best}")
            c2.metric("50+ Scores",    f"{(bm['runs']>=50).sum()}")
            c3.metric("30+ Scores",    f"{(bm['runs']>=30).sum()}")
            c4.metric("Duck Count",    f"{(bm['runs']==0).sum()}")

            fig = go.Figure()
            fig.add_trace(go.Bar(x=bm["date"].tail(25), y=bm["runs"].tail(25),
                                  name="Runs", marker_color="#58a6ff", opacity=0.8))
            fig.add_trace(go.Scatter(x=bm["date"].tail(25),
                                      y=bm["runs"].tail(25).rolling(5, min_periods=1).mean(),
                                      name="5-Match Avg", line=dict(color="#f0883e", width=2)))
            fig.update_layout(template="plotly_dark", title=f"{player} — Run Trend (Last 25 innings)",
                               height=350, xaxis_title="Match Date", yaxis_title="Runs")
            st.plotly_chart(fig, use_container_width=True)

            if "phase" in bat.columns:
                section("Phase Performance")
                pd_rows = []
                for ph in ["powerplay", "middle", "death"]:
                    sub = bat[bat["phase"] == ph]
                    lg  = sub[sub["legal_ball"] == 1]
                    if len(lg) < 3:
                        continue
                    pd_rows.append({
                        "Phase":       ph.title(),
                        "Runs":        int(sub["batsman_runs"].sum()),
                        "Balls":       len(lg),
                        "SR":          round(sub["batsman_runs"].sum() / len(lg) * 100, 1),
                        "Dismissals":  int(sub["is_wicket"].sum()),
                        "Avg":         round(sub["batsman_runs"].sum() / max(sub["is_wicket"].sum(), 1), 1),
                    })
                if pd_rows:
                    st.dataframe(pd.DataFrame(pd_rows), hide_index=True, use_container_width=True)

    with tab2:
        bwl = dm[dm["bowler"] == player]
        if bwl.empty:
            st.info("No bowling records found.")
        else:
            bm2 = bwl.groupby("match_id").agg(
                wkts=("is_wicket", "sum"), runs=("total_runs", "sum"),
                balls=("legal_ball", "sum"),
                date=("date", "max")
            ).reset_index().sort_values("date")
            bm2["econ"] = bm2["runs"] / (bm2["balls"] / 6).replace(0, 1)

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Wickets",  f"{int(bm2['wkts'].sum())}")
            c2.metric("Career Economy", f"{bm2['econ'].mean():.2f}")
            c3.metric("Best Spell",     f"{int(bm2['wkts'].max())} wkts")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=bm2["date"].tail(25), y=bm2["wkts"].tail(25),
                                  name="Wickets", marker_color="#3fb950", opacity=0.85), secondary_y=False)
            fig.add_trace(go.Scatter(x=bm2["date"].tail(25), y=bm2["econ"].tail(25),
                                      name="Economy", line=dict(color="#f0883e", width=2)),
                          secondary_y=True)
            fig.update_layout(template="plotly_dark", title=f"{player} — Bowling Trend (Last 25)",
                               height=350)
            fig.update_yaxes(title_text="Wickets", secondary_y=False)
            fig.update_yaxes(title_text="Economy", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            if "phase" in bwl.columns:
                section("Phase Bowling Performance")
                pb_rows = []
                for ph in ["powerplay", "middle", "death"]:
                    sub = bwl[bwl["phase"] == ph]
                    lg  = sub[sub["legal_ball"] == 1]
                    if len(lg) < 6:
                        continue
                    pb_rows.append({
                        "Phase":    ph.title(),
                        "Wickets":  int(sub["is_wicket"].sum()),
                        "Runs":     int(sub["total_runs"].sum()),
                        "Balls":    len(lg),
                        "Economy":  round(sub["total_runs"].sum() / (len(lg) / 6), 2),
                    })
                if pb_rows:
                    st.dataframe(pd.DataFrame(pb_rows), hide_index=True, use_container_width=True)

    with tab3:
        section("🧠 AI Form Insights")
        for ins in player_form_insights(player, deliveries, matches, "bat"):
            kind = "success" if any(k in ins for k in ["peak", "✅"]) else "warning" if any(k in ins for k in ["poor", "⚠️"]) else "info"
            insight_card(ins, kind)
        for ins in player_form_insights(player, deliveries, matches, "bowl"):
            kind = "success" if any(k in ins for k in ["✅", "roll"]) else "warning" if "⚠️" in ins else "info"
            insight_card(ins, kind)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            section("⚠️ Batting Weaknesses")
            bw = batsman_weakness(player, deliveries, matches)
            if "error" not in bw:
                for w in bw.get("weaknesses", []):
                    insight_card(w, "warning")
                for s in bw.get("strengths", []):
                    insight_card(s, "success")
                if bw.get("phase_stats"):
                    st.dataframe(pd.DataFrame(bw["phase_stats"]), hide_index=True, use_container_width=True)
        with col2:
            section("⚠️ Bowling Weaknesses")
            bwk = bowler_weakness(player, deliveries, matches)
            if "error" not in bwk:
                for w in bwk.get("weaknesses", []):
                    insight_card(w, "warning")
                for s in bwk.get("strengths", []):
                    insight_card(s, "success")
                if bwk.get("difficult_batsmen"):
                    st.markdown("**Batsmen who score well vs this bowler:**")
                    st.dataframe(pd.DataFrame(bwk["difficult_batsmen"]), hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 3: PLAYER COMPARISON
# ─────────────────────────────────────────────────────────
def page_player_comparison():
    st.title("⚔️ Player Head-to-Head Comparison")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    players = get_players(deliveries)
    c1, c2 = st.columns(2)
    p1 = c1.selectbox("Player 1", [""] + players, key="cmp_p1")
    p2 = c2.selectbox("Player 2", [""] + players, key="cmp_p2")
    if not p1 or not p2 or p1 == p2:
        st.info("Select two different players.")
        return

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")

    def bat_stats(name):
        b = dm[dm["batter"] == name].groupby("match_id").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"),
            wkts=("is_wicket","max"), date=("date","max")
        ).reset_index().sort_values("date")
        b["sr"] = b["runs"] / b["balls"].replace(0,1) * 100
        return b

    b1, b2 = bat_stats(p1), bat_stats(p2)

    section("🏏 Batting Comparison Summary")
    def bsummary(b, name):
        if b.empty:
            return {"Player": name, "Innings": 0, "Runs": 0, "Average": 0, "SR": 0, "Best": 0, "50+": 0}
        return {
            "Player":   name,
            "Innings":  len(b),
            "Runs":     int(b["runs"].sum()),
            "Average":  round(b["runs"].sum() / max(b["wkts"].sum(), 1), 1),
            "SR":       round(b["runs"].sum() / b["balls"].sum() * 100, 1),
            "Best":     int(b["runs"].max()),
            "50+":      int((b["runs"] >= 50).sum()),
            "30+":      int((b["runs"] >= 30).sum()),
        }
    st.dataframe(pd.DataFrame([bsummary(b1, p1), bsummary(b2, p2)]).set_index("Player"),
                 use_container_width=True)

    # Rolling average trend
    fig = go.Figure()
    for b, name, col in [(b1,p1,"#58a6ff"),(b2,p2,"#f0883e")]:
        if not b.empty:
            b_s = b.sort_values("date")
            fig.add_trace(go.Scatter(x=b_s["date"], y=b_s["runs"].rolling(5, min_periods=1).mean(),
                                      name=f"{name} (5-match avg)", line=dict(color=col, width=2)))
    fig.update_layout(template="plotly_dark", title="Rolling 5-Match Batting Average",
                       height=330, xaxis_title="Date", yaxis_title="Avg Runs")
    st.plotly_chart(fig, use_container_width=True)

    # Radar Chart
    section("🕸️ Performance Radar")
    def norm_val(v, lo, hi):
        return round((v - lo) / max(hi - lo, 0.001) * 10, 2)
    cats = ["Avg Runs", "Strike Rate", "Consistency", "Innings Played", "50+ Count"]
    def radar(b):
        if b.empty:
            return [0] * 5
        return [
            norm_val(b["runs"].mean(), 0, 70),
            norm_val(b["runs"].sum() / b["balls"].sum() * 100, 50, 200),
            norm_val(10 - b["runs"].std() / max(b["runs"].mean(), 1) * 10, 0, 10),
            norm_val(len(b), 0, 200),
            norm_val((b["runs"] >= 50).sum(), 0, 25),
        ]
    v1, v2 = radar(b1), radar(b2)
    fig = go.Figure()
    for vals, name, col in [(v1, p1, "#58a6ff"), (v2, p2, "#f0883e")]:
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                                       fill="toself", name=name, line=dict(color=col)))
    fig.update_layout(template="plotly_dark",
                       polar=dict(radialaxis=dict(visible=True, range=[0, 10])), height=400)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 4: BATSMAN vs BOWLER MATCHUP
# ─────────────────────────────────────────────────────────
def page_matchup():
    st.title("🥊 Batsman vs Bowler Matchup")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    batters = sorted(deliveries["batter"].dropna().unique().tolist())
    bowlers = sorted(deliveries["bowler"].dropna().unique().tolist())

    c1, c2 = st.columns(2)
    batsman = c1.selectbox("🏏 Select Batsman", [""] + batters)
    bowler  = c2.selectbox("🎳 Select Bowler",  [""] + bowlers)
    if not batsman or not bowler:
        st.info("Select a batsman and a bowler.")
        return

    result = matchup_matrix(batsman, bowler, deliveries)

    if result.get("balls", 0) < 5:
        st.warning(f"Only {result.get('balls',0)} deliveries found — not enough data for reliable analysis.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Balls",        result.get("balls", 0))
    c2.metric("Runs Scored",  result.get("runs", 0))
    c3.metric("Dismissals",   result.get("wickets", 0))
    c4.metric("Strike Rate",  result.get("strike_rate", 0))

    kind = "success" if result.get("strike_rate", 0) >= 140 else "warning" if result.get("wickets", 0) >= 2 else "info"
    insight_card(result.get("verdict", "No verdict"), kind)

    col1, col2 = st.columns(2)
    with col1:
        if result.get("phase_breakdown"):
            section("Phase-wise Matchup")
            rows = [{"Phase": ph.title(), **v} for ph, v in result["phase_breakdown"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with col2:
        if result.get("dismissal_types"):
            section("Dismissal Types")
            dt = result["dismissal_types"]
            fig = px.pie(values=list(dt.values()), names=list(dt.keys()), hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(template="plotly_dark", height=270)
            st.plotly_chart(fig, use_container_width=True)

    section("🌡️ Runs by Over")
    d_pair = deliveries[(deliveries["batter"] == batsman) & (deliveries["bowler"] == bowler)]
    if "over_num" in d_pair.columns and not d_pair.empty:
        ov = d_pair.groupby("over_num").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"), wkts=("is_wicket","sum")
        ).reset_index()
        ov["sr"] = ov["runs"] / ov["balls"].replace(0,1) * 100
        fig = px.bar(ov, x="over_num", y="runs", color="sr",
                     color_continuous_scale="RdYlGn",
                     labels={"over_num":"Over","runs":"Runs","sr":"SR"})
        fig.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 5: PARTNERSHIP TRACKER
# ─────────────────────────────────────────────────────────
def page_partnerships():
    st.title("🤝 Partnership Tracker")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    teams  = get_teams(matches)
    c1, c2 = st.columns(2)
    team   = c1.selectbox("Team", [""] + teams, key="part_t")
    season = c2.selectbox("Season", ["All"] + sorted(matches["season"].dropna().unique().tolist(), reverse=True), key="part_s")
    if not team:
        st.info("Select a team.")
        return

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")
    if season != "All":
        dm = dm[dm["season"] == int(season)]
    dm = dm[dm["batting_team"] == team]

    if dm.empty:
        st.warning("No data for this team/season.")
        return

    if "non_striker" not in dm.columns:
        st.info("Non-striker data not available in this dataset.")
        return

    dm["pair"] = dm.apply(lambda r: " & ".join(sorted([str(r["batter"]), str(r["non_striker"])])), axis=1)
    pairs = dm.groupby(["match_id","pair"]).agg(
        runs=("batsman_runs","sum"), balls=("legal_ball","sum")
    ).reset_index()
    agg = pairs.groupby("pair").agg(
        total_runs=("runs","sum"), total_balls=("balls","sum"), count=("match_id","count")
    ).reset_index()
    agg["rpo"] = agg["total_runs"] / agg["total_balls"].replace(0,1) * 6
    agg = agg.sort_values("total_runs", ascending=False)

    section("🔝 Top Partnerships by Runs")
    st.dataframe(agg.head(15).rename(columns={
        "pair":"Partnership","total_runs":"Runs","total_balls":"Balls","count":"Times Together","rpo":"RPO"
    }), hide_index=True, use_container_width=True)

    fig = px.bar(agg.head(12), x="total_runs", y="pair", orientation="h",
                 color="rpo", color_continuous_scale="Viridis",
                 labels={"total_runs":"Total Runs","pair":"Partnership"})
    fig.update_layout(template="plotly_dark", height=420, yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 6: TEAM COMPARISON
# ─────────────────────────────────────────────────────────
def page_team_comparison():
    st.title("⚔️ Team Head-to-Head Comparison")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    teams = get_teams(matches)
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Team 1", [""] + teams, key="tc1")
    t2 = c2.selectbox("Team 2", [""] + teams, key="tc2")
    if not t1 or not t2 or t1 == t2:
        st.info("Select two different teams.")
        return

    h2h = matches[
        ((matches["team1"]==t1)&(matches["team2"]==t2)) |
        ((matches["team1"]==t2)&(matches["team2"]==t1))
    ].copy()

    t1w = int((h2h["winner"]==t1).sum())
    t2w = int((h2h["winner"]==t2).sum())
    nr  = len(h2h) - t1w - t2w

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Matches", len(h2h))
    c2.metric(f"{t1} Wins", t1w)
    c3.metric(f"{t2} Wins", t2w)
    c4.metric("No Result", nr)

    col1, col2 = st.columns([1,2])
    with col1:
        fig = go.Figure(go.Pie(
            labels=[t1, t2, "NR"],
            values=[t1w, t2w, nr],
            hole=0.45,
            marker_colors=["#58a6ff","#f0883e","#484f58"]
        ))
        fig.update_layout(template="plotly_dark", height=260, title="Win Share")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if not h2h.empty:
            h2h["t1_won"] = (h2h["winner"] == t1).astype(int)
            h2h_s = h2h.sort_values("date")
            h2h_s["cum_t1"] = h2h_s["t1_won"].cumsum()
            h2h_s["cum_t2"] = (~h2h_s["t1_won"].astype(bool)).astype(int).cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=h2h_s["date"], y=h2h_s["cum_t1"],
                                      name=t1, line=dict(color="#58a6ff",width=2)))
            fig.add_trace(go.Scatter(x=h2h_s["date"], y=h2h_s["cum_t2"],
                                      name=t2, line=dict(color="#f0883e",width=2)))
            fig.update_layout(template="plotly_dark", height=260, title="Cumulative H2H Wins")
            st.plotly_chart(fig, use_container_width=True)

    section("🧠 AI Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t1}**")
        for ins in team_form_insights(t1, matches):
            insight_card(ins)
    with col2:
        st.markdown(f"**{t2}**")
        for ins in team_form_insights(t2, matches):
            insight_card(ins)

    section("📈 Rolling Win % (Last 15 matches each)")
    fig = go.Figure()
    for team, col in [(t1,"#58a6ff"),(t2,"#f0883e")]:
        tm = matches[(matches["team1"]==team)|(matches["team2"]==team)].sort_values("date").tail(15).copy()
        if not tm.empty:
            tm["won"] = (tm["winner"]==team).astype(int)
            tm["rwp"] = tm["won"].rolling(5,min_periods=1).mean()*100
            fig.add_trace(go.Scatter(x=tm["date"],y=tm["rwp"],name=f"{team} Win%",line=dict(color=col,width=2)))
    fig.add_hline(y=50,line_dash="dash",line_color="gray",opacity=0.5)
    fig.update_layout(template="plotly_dark",height=290,yaxis_title="Rolling Win %")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 7: AI MATCH PREDICTOR
# ─────────────────────────────────────────────────────────
def page_match_predictor():
    st.title("🤖 AI Match Predictor")
    st.caption("XGBoost + LightGBM + CatBoost Stacking Ensemble | SHAP Explainability | Monte Carlo Simulation")

    matches     = load_matches()
    deliveries  = load_deliveries()
    if not check_data(matches, deliveries):
        return

    model         = load_model()
    feature_names = load_feature_names()
    match_feats   = load_match_features()
    teams         = get_teams(matches)
    venues        = get_venues(matches)

    if model is None:
        st.error("❌ No trained model found. Run `python ml/train.py` first.")
        return

    c1,c2 = st.columns(2)
    team1 = c1.selectbox("🏏 Team 1", teams, key="pred_t1")
    team2 = c2.selectbox("🏏 Team 2", [t for t in teams if t != team1], key="pred_t2")
    venue = st.selectbox("🏟️ Venue", venues, key="pred_venue")
    c3,c4 = st.columns(2)
    toss_winner   = c3.selectbox("🎯 Toss Winner",   [team1, team2])
    toss_decision = c4.selectbox("↑ Toss Decision", ["bat", "field"])

    if st.button("🚀 Run AI Prediction", type="primary"):
        with st.spinner("🤖 Running ensemble prediction + Monte Carlo..."):

            # Build feature vector
            X_vec = np.zeros((1, len(feature_names))) if feature_names else None
            if not match_feats.empty and feature_names:
                mask = (
                    ((match_feats["team1"]==team1)&(match_feats["team2"]==team2)) |
                    ((match_feats["team1"]==team2)&(match_feats["team2"]==team1))
                )
                base = match_feats[mask]
                row  = base.iloc[-1] if not base.empty else match_feats.iloc[-1]
                X_vec = np.array([[float(row.get(f,0)) if pd.notna(row.get(f,0)) else 0.0 for f in feature_names]])
                fi = {n:i for i,n in enumerate(feature_names)}
                if "toss_decision_bat" in fi:
                    X_vec[0][fi["toss_decision_bat"]] = 1.0 if toss_decision=="bat" else 0.0
                if "toss_won_by_team1" in fi:
                    X_vec[0][fi["toss_won_by_team1"]] = 1.0 if toss_winner==team1 else 0.0

            proba  = float(model.predict_proba(X_vec)[0][1]) if X_vec is not None else 0.52
            winner = team1 if proba > 0.5 else team2
            conf   = round(abs(proba-0.5)*200, 1)

        st.divider()
        c1, c2 = st.columns(2)
        for col_obj, team, prob, color in [
            (c1, team1, proba*100, "#58a6ff"),
            (c2, team2, (1-proba)*100, "#f0883e")
        ]:
            with col_obj:
                st.markdown(f"### {team}")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob,
                    number={"suffix":"%","font":{"size":36}},
                    title={"text":"Win Probability","font":{"size":14}},
                    gauge={
                        "axis":{"range":[0,100]},
                        "bar":{"color":color,"thickness":0.25},
                        "bgcolor":"#21262d",
                        "borderwidth":0,
                        "steps":[
                            {"range":[0,40],"color":"#21262d"},
                            {"range":[40,60],"color":"#2d333b"},
                            {"range":[60,100],"color":"#1c2c3c" if color=="#58a6ff" else "#2c1c0c"},
                        ]
                    }
                ))
                fig.update_layout(template="plotly_dark", height=270, margin=dict(t=40,b=0,l=20,r=20))
                st.plotly_chart(fig, use_container_width=True)

        st.success(f"🏆 **Predicted Winner: {winner}** | Model Confidence: **{conf:.1f}%**")

        # SHAP
        section("🔍 SHAP Explainability — Why this prediction?")
        shap_path = os.path.join(SHAP_DIR,"shap_xgboost.csv")
        if not os.path.exists(shap_path):
            shap_path = os.path.join(SHAP_DIR,"shap_lightgbm.csv")
        if os.path.exists(shap_path):
            sd = pd.read_csv(shap_path).head(12)
            fig = px.bar(sd, x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         labels={"importance":"SHAP Importance","feature":"Feature"})
            fig.update_layout(template="plotly_dark", height=360,
                               yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            top3 = sd.head(3)["feature"].str.replace("_"," ").str.title().tolist()
            insight_card(f"💡 Top prediction drivers: **{' · '.join(top3)}**", "info")
        else:
            st.info("Run `python ml/train.py` with SHAP to enable explainability.")

        # Monte Carlo
        section("🎲 Monte Carlo Simulation — 10,000 Matches")
        sim = run_simulation(team1, team2, proba, n_sim=10000)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric(f"{team1} Wins", f"{sim['team1_win_pct']:.1f}%")
        c2.metric(f"{team2} Wins", f"{sim['team2_win_pct']:.1f}%")
        c3.metric("95% CI", f"{sim['ci_95_low']*100:.0f}% – {sim['ci_95_high']*100:.0f}%")
        c4.metric("Uncertainty", f"±{sim['std_sim_prob']*100:.1f}%")

        # Score distribution
        edges   = sim["score_hist_edges"]
        centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=centers, y=sim["team1_score_hist"], name=team1,
                              marker_color="#58a6ff", opacity=0.75))
        fig.add_trace(go.Bar(x=centers, y=sim["team2_score_hist"], name=team2,
                              marker_color="#f0883e", opacity=0.75))
        fig.update_layout(template="plotly_dark", barmode="overlay",
                           title="Simulated Score Distribution (10,000 runs)",
                           height=300, xaxis_title="Runs", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

        insight_card(f"🎯 Dominant {team1} victory: {sim['scenario_dominant_pct']:.1f}% | "
                     f"Contested: {sim['scenario_contested_pct']:.1f}% | "
                     f"Upset ({team2}): {sim['scenario_upset_pct']:.1f}%", "info")

        section("📢 Shareable Match Prediction")
        st.code(viral_insight(team1, team2, sim), language=None)

# ─────────────────────────────────────────────────────────
# PAGE 8: TOSS ADVISOR
# ─────────────────────────────────────────────────────────
def page_toss_advisor():
    st.title("🎯 AI Toss Advisor & Strategy Engine")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    teams  = get_teams(matches)
    venues = get_venues(matches)
    c1,c2,c3 = st.columns(3)
    t1    = c1.selectbox("Team 1",[""] + teams, key="ta1")
    t2    = c2.selectbox("Team 2",[""] + teams, key="ta2")
    venue = c3.selectbox("Venue", venues, key="tav")
    if not t1 or not t2 or t1==t2:
        st.info("Select both teams and a venue.")
        return

    res = toss_advisor(t1, t2, venue, matches)
    rec = "🏏 Bat First" if res["recommendation"]=="bat" else "🏃 Field First"
    col = "success" if res["recommendation"]=="bat" else "info"
    insight_card(f"**Recommendation: {rec}** — Confidence: {res['confidence']:.0f}%", col)
    for ins in res["insights"]:
        insight_card(ins)

    section("📊 Venue Toss Stats")
    vm = matches[matches["venue"]==venue]
    if len(vm) >= 5:
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
        with c2:
            if "toss_decision" in vm.columns:
                td = vm["toss_decision"].value_counts()
                fig = px.pie(values=td.values,names=td.index,hole=0.4,
                             color_discrete_sequence=["#58a6ff","#f0883e"],
                             title="Toss Decisions at Venue")
                fig.update_layout(template="plotly_dark",height=250)
                st.plotly_chart(fig, use_container_width=True)

    section("🎳 Bowling Strategy Insights")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**How to bowl vs {t2}**")
        for ins in bowling_strategy(t2, deliveries, matches)["strategy_insights"]:
            insight_card(ins)
    with c2:
        st.markdown(f"**How to bowl vs {t1}**")
        for ins in bowling_strategy(t1, deliveries, matches)["strategy_insights"]:
            insight_card(ins)

# ─────────────────────────────────────────────────────────
# PAGE 9: WIN PROBABILITY TRACKER
# ─────────────────────────────────────────────────────────
def page_win_probability():
    st.title("📈 Match Win Probability Tracker")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    teams = get_teams(matches)
    c1,c2 = st.columns(2)
    t1 = c1.selectbox("Team 1",[""] + teams, key="wp1")
    t2 = c2.selectbox("Team 2",[""] + teams, key="wp2")
    if not t1 or not t2:
        st.info("Select both teams.")
        return

    h2h = matches[
        ((matches["team1"]==t1)&(matches["team2"]==t2)) |
        ((matches["team1"]==t2)&(matches["team2"]==t1))
    ].sort_values("date")
    if h2h.empty:
        st.warning("No matches found between these teams.")
        return

    match_id = st.selectbox("Select Match", h2h["match_id"].tolist(),
        format_func=lambda x: f"{str(h2h[h2h['match_id']==x]['date'].values[0])[:10]} — Match {x}")

    dm = deliveries[deliveries["match_id"]==match_id].copy()
    mr = h2h[h2h["match_id"]==match_id].iloc[0]
    if dm.empty:
        st.warning("No ball-by-ball data.")
        return

    c1,c2,c3 = st.columns(3)
    c1.metric("Winner",    str(mr.get("winner","N/A")))
    c2.metric("Margin",    str(mr.get("result_margin","N/A")))
    c3.metric("Total Balls", len(dm))

    dm["ball_idx"] = range(len(dm))
    fig = go.Figure()
    for tm_name in dm["batting_team"].unique()[:2]:
        sub = dm[dm["batting_team"]==tm_name].copy()
        sub["cum_r"] = sub["total_runs"].cumsum()
        fig.add_trace(go.Scatter(x=sub["ball_idx"],y=sub["cum_r"],
                                  name=str(tm_name),mode="lines",line=dict(width=2)))
    fig.update_layout(template="plotly_dark",title="Cumulative Runs by Ball",
                       height=330,xaxis_title="Ball Number",yaxis_title="Runs")
    st.plotly_chart(fig, use_container_width=True)

    if "over_num" in dm.columns:
        section("Run Rate by Over")
        ov = dm[dm["legal_ball"]==1].groupby(["batting_team","over_num"]).agg(
            runs=("total_runs","sum"), balls=("legal_ball","sum")
        ).reset_index()
        ov["rr"] = ov["runs"]/(ov["balls"]/6)
        fig = px.bar(ov,x="over_num",y="rr",color="batting_team",barmode="group",
                     color_discrete_sequence=["#58a6ff","#f0883e"],
                     labels={"over_num":"Over","rr":"Run Rate"})
        fig.update_layout(template="plotly_dark",height=280)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 10: SEASON POINTS TABLE
# ─────────────────────────────────────────────────────────
def page_season_table():
    st.title("📋 Season Points Table")
    matches = load_matches()
    if matches.empty:
        st.error("Data not loaded.")
        return

    seasons = sorted(matches["season"].dropna().unique().tolist(), reverse=True)
    season  = st.selectbox("Select Season", seasons)
    sm      = matches[matches["season"]==season]
    all_teams = list(set(sm["team1"].dropna().tolist()+sm["team2"].dropna().tolist()))

    rows = []
    for t in all_teams:
        tm     = sm[(sm["team1"]==t)|(sm["team2"]==t)]
        played = len(tm)
        wins   = int((tm["winner"]==t).sum())
        nr     = int((tm.get("result","")=="no result").sum()) if "result" in tm else 0
        losses = played - wins - nr
        rows.append({"Team":t,"P":played,"W":wins,"L":losses,"NR":nr,"Pts":wins*2,"Win%":round(wins/max(played,1)*100,1)})

    tdf = pd.DataFrame(rows).sort_values("Pts",ascending=False).reset_index(drop=True)
    tdf.index += 1
    st.dataframe(tdf.style.background_gradient(subset=["Pts"],cmap="Blues")
                          .background_gradient(subset=["Win%"],cmap="Greens"),
                 use_container_width=True)

    fig = px.bar(tdf,x="Team",y="Pts",color="Win%",color_continuous_scale="Blues",
                 title=f"IPL {season} — Points",labels={"Pts":"Points"})
    fig.update_layout(template="plotly_dark",height=360,xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 11: RUN HEATMAP
# ─────────────────────────────────────────────────────────
def page_run_heatmap():
    st.title("🌡️ Run Scoring Heatmap")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    teams  = get_teams(matches)
    c1,c2  = st.columns(2)
    team   = c1.selectbox("Team",[""] + teams, key="rh_t")
    season = c2.selectbox("Season",["All"] + sorted(matches["season"].dropna().unique().tolist()), key="rh_s")
    if not team:
        st.info("Select a team.")
        return

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")
    if season != "All":
        dm = dm[dm["season"]==int(season)]
    dm = dm[dm["batting_team"]==team]
    if dm.empty:
        st.warning("No data.")
        return

    dm["over_bin"] = dm["over_num"].fillna(0).astype(int) if "over_num" in dm.columns else 0
    heat = dm.groupby("over_bin").agg(
        runs=("total_runs","sum"), balls=("legal_ball","sum"), wkts=("is_wicket","sum")
    ).reset_index()
    heat["rr"] = heat["runs"]/heat["balls"].replace(0,1)*6

    section(f"⚾ Run Rate by Over — {team}")
    fig = px.bar(heat,x="over_bin",y="rr",color="rr",color_continuous_scale="RdYlGn",
                 labels={"over_bin":"Over","rr":"Run Rate"})
    fig.add_vline(x=5.5,line_dash="dash",line_color="white",opacity=0.5,annotation_text="PP End")
    fig.add_vline(x=14.5,line_dash="dash",line_color="yellow",opacity=0.5,annotation_text="Middle End")
    fig.update_layout(template="plotly_dark",height=330,coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    section("💀 Wicket Density by Over")
    fig = px.scatter(heat,x="over_bin",y="wkts",size=heat["wkts"].clip(lower=0.1),color="wkts",
                     color_continuous_scale="Reds",size_max=40,
                     labels={"over_bin":"Over","wkts":"Wickets"})
    fig.update_layout(template="plotly_dark",height=270,coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 12: FORM TRACKER
# ─────────────────────────────────────────────────────────
def page_form_tracker():
    st.title("📈 Player Form Tracker")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    players = get_players(deliveries)
    c1,c2,c3 = st.columns([3,2,1])
    player = c1.selectbox("Player",[""] + players, key="ft_p")
    role   = c2.radio("Role",["Batting","Bowling"],horizontal=True, key="ft_r")
    last_n = c3.number_input("Last N",5,50,20, key="ft_n")
    if not player:
        st.info("Select a player.")
        return

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")
    dm = dm.sort_values("date", na_position="last")

    if role == "Batting":
        bat = dm[dm["batter"]==player].groupby("match_id").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"),
            dismissed=("is_wicket","max"), date=("date","max")
        ).reset_index().sort_values("date").tail(int(last_n))
        if bat.empty:
            st.warning("No batting data.")
            return
        bat["sr"]       = bat["runs"]/bat["balls"].replace(0,1)*100
        bat["roll_avg"] = bat["runs"].rolling(5,min_periods=1).mean()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Last 5 Avg",     f"{bat.tail(5)['runs'].mean():.1f}")
        c2.metric("Period Avg",     f"{bat['runs'].mean():.1f}")
        c3.metric("Best",           f"{bat['runs'].max()}")
        c4.metric("Consistency σ",  f"{bat['runs'].std():.1f}")

        fig = go.Figure()
        colors = ["#3fb950" if r>=50 else "#58a6ff" if r>=30 else "#f0883e" if r>=10 else "#e84040" for r in bat["runs"]]
        fig.add_trace(go.Bar(x=bat["date"],y=bat["runs"],marker_color=colors,name="Runs",opacity=0.85))
        fig.add_trace(go.Scatter(x=bat["date"],y=bat["roll_avg"],name="5-Match Avg",
                                  line=dict(color="white",width=2,dash="dash")))
        fig.update_layout(template="plotly_dark",title=f"{player} — Batting Form",height=350)
        st.plotly_chart(fig, use_container_width=True)
        for ins in player_form_insights(player, deliveries, matches, "bat"):
            kind = "success" if "peak" in ins.lower() else "warning" if "poor" in ins.lower() else "info"
            insight_card(ins, kind)

    else:
        bowl = dm[dm["bowler"]==player].groupby("match_id").agg(
            wkts=("is_wicket","sum"), runs=("total_runs","sum"),
            balls=("legal_ball","sum"), date=("date","max")
        ).reset_index().sort_values("date").tail(int(last_n))
        if bowl.empty:
            st.warning("No bowling data.")
            return
        bowl["econ"] = bowl["runs"]/(bowl["balls"]/6).replace(0,1)
        c1,c2,c3 = st.columns(3)
        c1.metric("Last 5 Wkts/Match", f"{bowl.tail(5)['wkts'].mean():.1f}")
        c2.metric("Last 5 Econ",       f"{bowl.tail(5)['econ'].mean():.2f}")
        c3.metric("Best Spell",         f"{bowl['wkts'].max()} wkts")

        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=bowl["date"],y=bowl["wkts"],name="Wickets",
                              marker_color="#3fb950",opacity=0.85),secondary_y=False)
        fig.add_trace(go.Scatter(x=bowl["date"],y=bowl["econ"],name="Economy",
                                  line=dict(color="#f0883e",width=2)),secondary_y=True)
        fig.update_layout(template="plotly_dark",title=f"{player} — Bowling Form",height=350)
        fig.update_yaxes(title_text="Wickets",secondary_y=False)
        fig.update_yaxes(title_text="Economy",secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
        for ins in player_form_insights(player, deliveries, matches, "bowl"):
            insight_card(ins)

# ─────────────────────────────────────────────────────────
# PAGE 13: BREAKOUT PLAYERS
# ─────────────────────────────────────────────────────────
def page_breakout():
    st.title("⚡ Breakout Players & Rising Stars")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    seasons = sorted(matches["season"].dropna().unique().tolist(), reverse=True)
    season  = st.selectbox("Current Season", seasons, key="bp_s")
    prev    = season - 1
    dm      = deliveries.copy()

    def season_bat(s):
        sub = dm[dm["season"]==s]
        agg = sub.groupby("batter").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"),
            innings=("match_id","nunique"), wkts=("is_wicket","sum")
        ).reset_index()
        agg["sr"]  = agg["runs"]/agg["balls"].replace(0,1)*100
        agg["avg"] = agg["runs"]/agg["wkts"].replace(0,1)
        return agg[agg["innings"]>=3]

    curr = season_bat(season)
    prev_df = season_bat(prev) if prev in dm["season"].dropna().unique() else pd.DataFrame()

    if not prev_df.empty:
        mg = curr.merge(prev_df[["batter","runs","sr"]], on="batter", suffixes=("_curr","_prev"), how="inner")
        mg["growth"] = mg["runs_curr"] - mg["runs_prev"]
        mg["sr_ch"]  = mg["sr_curr"]  - mg["sr_prev"]
        bo = mg[mg["growth"]>0].sort_values("growth",ascending=False).head(12)

        section(f"🚀 Biggest Improvements: {prev} → {season}")
        fig = px.bar(bo,x="growth",y="batter",orientation="h",color="sr_ch",
                     color_continuous_scale="Greens",
                     labels={"growth":"Extra Runs","batter":"Player","sr_ch":"SR Change"})
        fig.update_layout(template="plotly_dark",height=420,yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bo[["batter","runs_curr","runs_prev","growth","sr_curr","sr_prev"]].rename(columns={
            "batter":"Player","runs_curr":f"Runs {season}","runs_prev":f"Runs {prev}",
            "growth":"Growth","sr_curr":f"SR {season}","sr_prev":f"SR {prev}"
        }), hide_index=True, use_container_width=True)
    else:
        section(f"Top Batsmen — {season}")
        st.dataframe(curr.sort_values("runs",ascending=False).head(15),hide_index=True)

# ─────────────────────────────────────────────────────────
# PAGE 14: FANTASY OPTIMIZER
# ─────────────────────────────────────────────────────────
def page_fantasy():
    st.title("💰 Dream11 Fantasy Team Optimizer")
    st.caption("ILP optimization (PuLP) with 3 strategies: Maximize FP · Safe · Differentiated")

    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    teams  = get_teams(matches)
    c1,c2  = st.columns(2)
    t1_sel = c1.selectbox("Team 1",[""] + teams, key="fan_t1")
    t2_sel = c2.selectbox("Team 2",[""] + teams, key="fan_t2")
    if not t1_sel or not t2_sel:
        st.info("Select both teams first.")
        return

    strategy = st.radio("Strategy",["maximize","safe","differentiated"],horizontal=True,
                         format_func=lambda x:{
                             "maximize":"🚀 Maximize Points",
                             "safe":"🛡️ Safe/Consistent",
                             "differentiated":"🎯 Differentiated (low-ownership)"
                         }[x])

    if st.button("⚡ Generate Fantasy Team", type="primary"):
        with st.spinner("Optimizing with ILP..."):
            from optimizer.fantasy_optimizer import estimate_player_fp

            dm = deliveries.copy()
            dm["season"] = pd.to_numeric(dm["season"], errors="coerce")
            latest = int(dm["season"].max())
            rec = dm[dm["season"] >= latest - 2]

            # ── Get top batters for each team (up to 15 each) ──
            t1_bats = rec[rec["batting_team"]==t1_sel]["batter"].value_counts().head(15).index.tolist()
            t2_bats = rec[rec["batting_team"]==t2_sel]["batter"].value_counts().head(15).index.tolist()

            # ── Get top bowlers for each team ──
            t1_bowls = rec[rec["bowling_team"]==t1_sel]["bowler"].value_counts().head(8).index.tolist() if "bowling_team" in rec.columns else []
            t2_bowls = rec[rec["bowling_team"]==t2_sel]["bowler"].value_counts().head(8).index.tolist() if "bowling_team" in rec.columns else []

            # All unique players
            all_p = list(set(t1_bats + t2_bats + t1_bowls + t2_bowls))
            if len(all_p) < 11:
                st.error(f"Not enough players found ({len(all_p)}). Try different teams or seasons.")
                st.stop()

            # ── Estimate fantasy points ──
            fp_df = estimate_player_fp(deliveries, matches, all_p, recent_n=5)
            fp_df = fp_df[fp_df["fp_total"] > 0].copy()
            if fp_df.empty:
                fp_df = pd.DataFrame({"player": all_p, "fp_batting": 30.0, "fp_bowling": 10.0, "fp_total": 40.0})

            # ── Assign team ──
            def get_team(p):
                if p in t1_bats or p in t1_bowls:
                    return t1_sel
                return t2_sel
            fp_df["team"] = fp_df["player"].apply(get_team)

            # ── Assign roles intelligently ──
            bowl_set = set(t1_bowls + t2_bowls)
            bat_set  = set(t1_bats  + t2_bats)

            # Players who appear in both → All-rounder
            ar_set = bowl_set & bat_set

            def assign_role(row):
                p = row["player"]
                if p in ar_set:
                    return "AR"
                if p in bowl_set:
                    return "BOWL"
                return "BAT"

            fp_df["role"] = fp_df.apply(assign_role, axis=1)

            # Ensure minimum role coverage: need 1 WK, 3 BAT, 1 AR, 3 BOWL
            # Assign WK to top 2 batters (one per team) as proxy
            t1_wk_candidates = [p for p in t1_bats if p in fp_df["player"].values and fp_df[fp_df["player"]==p]["role"].values[0] == "BAT"]
            t2_wk_candidates = [p for p in t2_bats if p in fp_df["player"].values and fp_df[fp_df["player"]==p]["role"].values[0] == "BAT"]
            wk_assigned = []
            if t1_wk_candidates:
                wk_assigned.append(t1_wk_candidates[0])
            if t2_wk_candidates:
                wk_assigned.append(t2_wk_candidates[0])
            for p in wk_assigned:
                fp_df.loc[fp_df["player"]==p, "role"] = "WK"

            # Ensure at least 1 AR if none assigned
            if "AR" not in fp_df["role"].values:
                # Pick the player with both bat and bowl fp as AR
                fp_df["ar_score"] = fp_df["fp_batting"] + fp_df["fp_bowling"]
                ar_candidate = fp_df.sort_values("ar_score", ascending=False).iloc[0]["player"]
                fp_df.loc[fp_df["player"]==ar_candidate, "role"] = "AR"

            # ── Credits ──
            mu  = fp_df["fp_total"].mean()
            std = fp_df["fp_total"].std()
            std = std if std > 0 else 1.0
            fp_df["credits"] = (9.0 + (fp_df["fp_total"] - mu) / std * 0.8).clip(7.5, 11.5)

            # ── Debug info ──
            role_counts = fp_df["role"].value_counts().to_dict()
            st.caption(f"Pool: {len(fp_df)} players | Roles: {role_counts}")

            result = optimize_team_ilp(fp_df, strategy=strategy)

        if "error" in result:
            st.error(f"Optimization error: {result['error']}")
            return

        st.success(f"✅ Optimized! Total Expected FP: **{result['total_fp']}** | Credits: **{result['total_credits']:.1f}/100**")
        c1,c2 = st.columns(2)
        c1.metric("👑 Captain",     result["captain"])
        c2.metric("🥈 Vice-Captain", result["vice_captain"])

        tdf = pd.DataFrame(result["team"])
        tdf["Tag"] = tdf["player"].apply(lambda p:
            "👑 C" if p==result["captain"] else "🥈 VC" if p==result["vice_captain"] else "")
        role_colors = {"WK":"🧤","BAT":"🏏","AR":"⚡","BOWL":"🎳"}
        tdf["Role"] = tdf["role"].map(role_colors) + " " + tdf["role"]
        st.dataframe(tdf[["player","team","Role","credits","fp_total","Tag"]].rename(columns={
            "player":"Player","team":"Team","credits":"Credits","fp_total":"Est. FP","Tag":""
        }), hide_index=True, use_container_width=True)

        fig = px.bar(tdf.sort_values("fp_total"),x="fp_total",y="player",orientation="h",
                     color="role",
                     color_discrete_map={"WK":"#58a6ff","BAT":"#3fb950","AR":"#e3b341","BOWL":"#f0883e"},
                     labels={"fp_total":"Est. Fantasy Points","player":"Player"})
        fig.update_layout(template="plotly_dark",height=380)
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 15: PLAYER RANKINGS
# ─────────────────────────────────────────────────────────
def page_rankings():
    st.title("🏅 Player Power Rankings")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    seasons = sorted(matches["season"].dropna().unique().tolist(), reverse=True)
    c1,c2 = st.columns(2)
    season  = c1.selectbox("Season",["Overall"] + [str(s) for s in seasons])
    rank_by = c2.selectbox("Rank By",["Runs","Strike Rate","Wickets","Economy"])

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")
    if season != "Overall":
        dm = dm[dm["season"]==int(season)]

    tab1, tab2 = st.tabs(["🏏 Top Batsmen","🎳 Top Bowlers"])

    with tab1:
        bat = dm.groupby("batter").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"),
            innings=("match_id","nunique"), wkts=("is_wicket","sum"),
            fours=("batsman_runs",lambda x:(x==4).sum()),
            sixes=("batsman_runs",lambda x:(x==6).sum()),
        ).reset_index()
        bat["sr"]  = (bat["runs"]/bat["balls"].replace(0,1)*100).round(2)
        bat["avg"] = (bat["runs"]/bat["wkts"].replace(0,1)).round(2)
        bat = bat[bat["innings"]>=(3 if season!="Overall" else 10)]
        sc  = {"Runs":"runs","Strike Rate":"sr"}.get(rank_by,"runs")
        bat = bat.sort_values(sc,ascending=False).head(25).reset_index(drop=True)
        bat.index += 1
        st.dataframe(bat[["batter","innings","runs","avg","sr","fours","sixes"]].rename(columns={
            "batter":"Player","innings":"Inn","runs":"Runs","avg":"Avg","sr":"SR","fours":"4s","sixes":"6s"
        }).style.background_gradient(subset=["Runs","SR"],cmap="Blues"), use_container_width=True)

        fig = px.scatter(bat.head(20),x="sr",y="avg",size="runs",color="runs",text="batter",
                         color_continuous_scale="Blues",
                         labels={"sr":"Strike Rate","avg":"Average","runs":"Runs"})
        fig.update_traces(textposition="top center")
        fig.update_layout(template="plotly_dark",height=430,title="Strike Rate vs Average Bubble Chart",
                           coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bowl = dm.groupby("bowler").agg(
            wkts=("is_wicket","sum"), runs=("total_runs","sum"),
            balls=("legal_ball","sum"), innings=("match_id","nunique")
        ).reset_index()
        bowl["econ"] = (bowl["runs"]/(bowl["balls"]/6).replace(0,1)).round(2)
        bowl["avg"]  = (bowl["runs"]/bowl["wkts"].replace(0,1)).round(2)
        bowl["sr"]   = (bowl["balls"]/bowl["wkts"].replace(0,1)).round(2)
        bowl = bowl[bowl["innings"]>=(3 if season!="Overall" else 10)]
        sc   = {"Wickets":"wkts","Economy":"econ"}.get(rank_by,"wkts")
        asc  = sc in ["econ","avg"]
        bowl = bowl.sort_values(sc,ascending=asc).head(25).reset_index(drop=True)
        bowl.index += 1
        st.dataframe(bowl[["bowler","innings","wkts","econ","avg","sr"]].rename(columns={
            "bowler":"Player","innings":"Inn","wkts":"Wkts","econ":"Econ","avg":"Avg","sr":"SR"
        }).style.background_gradient(subset=["Wkts"],cmap="Greens")
          .background_gradient(subset=["Econ"],cmap="RdYlGn_r"), use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 16: VENUE ANALYSIS
# ─────────────────────────────────────────────────────────
def page_venue_analysis():
    st.title("🏟️ Deep Venue Analysis")
    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    venues = get_venues(matches)
    venue  = st.selectbox("Select Venue", venues)
    vm     = matches[matches["venue"]==venue]
    vd     = deliveries[deliveries["match_id"].isin(vm["match_id"])]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Matches Played", len(vm))
    c2.metric("Total Runs",     f"{int(vd['total_runs'].sum()):,}")
    c3.metric("Avg Match Runs", f"{vd.groupby('match_id')['total_runs'].sum().mean():.0f}")
    c4.metric("Avg Wickets",    f"{vd.groupby('match_id')['is_wicket'].sum().mean():.1f}")

    c1,c2 = st.columns(2)
    with c1:
        if "bat_first_won" in vm.columns:
            section("🎯 Bat First vs Chase")
            bf = vm["bat_first_won"].mean()*100
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=bf,
                title={"text":"Bat First Win %"},
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#58a6ff"}}
            ))
            fig.update_layout(template="plotly_dark",height=250)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "phase" in vd.columns:
            section("📊 Run Rate by Phase")
            ph_agg = vd.groupby("phase").agg(
                runs=("total_runs","sum"), balls=("legal_ball","sum")
            ).reset_index()
            ph_agg["rr"] = ph_agg["runs"]/ph_agg["balls"].replace(0,1)*6
            fig = px.bar(ph_agg,x="phase",y="rr",color="rr",
                         color_continuous_scale="Blues",
                         labels={"phase":"Phase","rr":"Run Rate"})
            fig.update_layout(template="plotly_dark",height=250,coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    section("🏆 Most Successful Teams at This Venue")
    tw = vm["winner"].value_counts().reset_index()
    tw.columns = ["team","wins"]
    tm_ = pd.concat([vm["team1"],vm["team2"]]).value_counts().reset_index()
    tm_.columns = ["team","played"]
    ts  = tw.merge(tm_,on="team")
    ts["win_pct"] = ts["wins"]/ts["played"]*100
    fig = px.bar(ts.sort_values("wins",ascending=False).head(10),
                 x="wins",y="team",orientation="h",color="win_pct",
                 color_continuous_scale="Greens",
                 labels={"wins":"Wins","team":"Team","win_pct":"Win%"})
    fig.update_layout(template="plotly_dark",height=330,yaxis={"categoryorder":"total ascending"},
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    section("🎯 Toss Insights")
    res = toss_advisor("Team A","Team B",venue,matches)
    for ins in res["insights"]:
        insight_card(ins)

# ─────────────────────────────────────────────────────────
# PAGE 17: PLAYER SIMILARITY
# ─────────────────────────────────────────────────────────
def page_player_similarity():
    st.title("🧬 Player Similarity Engine")
    st.caption("Cosine similarity on statistical profiles to find comparable players")

    matches    = load_matches()
    deliveries = load_deliveries()
    if not check_data(matches, deliveries):
        return

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    players = get_players(deliveries)
    c1,c2 = st.columns([3,1])
    player = c1.selectbox("Find players similar to:",[""] + players)
    role   = c2.radio("Role",["Batting","Bowling"],horizontal=True)
    top_n  = st.slider("Top N similar",3,15,8)
    if not player:
        st.info("Select a player.")
        return

    dm = deliveries.copy()
    dm["date"] = pd.to_datetime(dm["date"], errors="coerce")

    if role == "Batting":
        feat = dm.groupby("batter").agg(
            avg_runs=("batsman_runs","mean"), total_runs=("batsman_runs","sum"),
            balls=("legal_ball","sum"), innings=("match_id","nunique"),
            sixes=("batsman_runs",lambda x:(x==6).sum()),
            fours=("batsman_runs",lambda x:(x==4).sum()),
            wkts=("is_wicket","sum"),
        ).reset_index().rename(columns={"batter":"player"})
        feat["sr"]  = feat["total_runs"]/feat["balls"].replace(0,1)*100
        feat["avg"] = feat["total_runs"]/feat["wkts"].replace(0,1)
        feat_cols   = ["avg_runs","sr","avg","sixes","fours"]
        feat = feat[feat["innings"] >= 5]
    else:
        feat = dm.groupby("bowler").agg(
            total_wkts=("is_wicket","sum"), total_runs=("total_runs","sum"),
            balls=("legal_ball","sum"), innings=("match_id","nunique"),
        ).reset_index().rename(columns={"bowler":"player"})
        feat["econ"] = feat["total_runs"]/(feat["balls"]/6).replace(0,1)
        feat["sr"]   = feat["balls"]/feat["total_wkts"].replace(0,1)
        feat["avg"]  = feat["total_runs"]/feat["total_wkts"].replace(0,1)
        feat_cols    = ["total_wkts","econ","sr","avg"]
        feat = feat[feat["innings"] >= 5]

    if player not in feat["player"].values:
        st.warning(f"Not enough data for {player} (needs 5+ innings).")
        return

    feat = feat.dropna(subset=feat_cols).reset_index(drop=True)
    X = StandardScaler().fit_transform(feat[feat_cols].fillna(0).values)
    idx_arr = feat[feat["player"]==player].index[0]
    sims = cosine_similarity([X[idx_arr]], X)[0]
    feat["similarity"] = sims

    sim_df = feat[feat["player"]!=player].sort_values("similarity",ascending=False).head(top_n)

    st.markdown(f"### Players most similar to **{player}** ({role})")
    fig = px.bar(sim_df,x="similarity",y="player",orientation="h",
                 color="similarity",color_continuous_scale="Blues",
                 labels={"similarity":"Similarity Score","player":"Player"})
    fig.update_layout(template="plotly_dark",height=380,yaxis={"categoryorder":"total ascending"},
                       coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    display_cols = ["player","similarity"] + feat_cols
    st.dataframe(sim_df[display_cols].rename(columns={"player":"Player","similarity":"Score"}),
                 hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────
# PAGE 18: ABOUT & MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────
def page_about():
    st.title("📖 About CricketBrain AI")
    st.markdown("""
    **CricketBrain AI** is a fully production-grade IPL cricket intelligence platform.

    | Component | Technology |
    |---|---|
    | ML Engine | XGBoost · LightGBM · CatBoost · Stacking Ensemble |
    | Tuning | Optuna (50-trial Bayesian hyperparameter search) |
    | Explainability | SHAP TreeExplainer (feature-level attribution) |
    | Simulation | Monte Carlo (10,000+ iterations, Beta noise) |
    | Optimization | PuLP ILP (Dream11 fantasy optimizer) |
    | Backend | FastAPI (15 endpoints, async, Pydantic v2) |
    | Dashboard | Streamlit (18 pages, Plotly charts) |
    | Data | IPL 2008-2025 ball-by-ball (Kaggle) |
    """)

    st.divider()
    section("🤖 Model Performance Dashboard")
    metrics = load_metrics()
    if not metrics:
        st.warning("No trained models yet. Run `python ml/train.py`")
    else:
        model_names = ["RandomForest","XGBoost","LightGBM","CatBoost","Stacking"]
        rows = []
        for k in model_names:
            if k in metrics and isinstance(metrics[k], dict):
                v = metrics[k]
                rows.append({
                    "Model":       k,
                    "ROC-AUC":     f"{v.get('roc_auc', v.get('cv_mean',0)):.4f}",
                    "Accuracy":    f"{v.get('accuracy',0):.4f}",
                    "Log Loss":    f"{v.get('log_loss',0):.4f}",
                    "Brier Score": f"{v.get('brier_score',0):.4f}",
                    "CV AUC":      f"{v.get('cv_mean',0):.4f} ± {v.get('cv_std',0):.4f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        best    = metrics.get("best_model","N/A")
        cal_auc = metrics.get("calibrated_auc",0)
        st.success(f"🏆 Best Model: **{best}** | Calibrated ROC-AUC: **{cal_auc:.4f}**")

    section("📂 Project Structure")
    st.code("""
CricketBrain AI/
├── etl/
│   ├── data_cleaning.py       # Ball-by-ball IPL pipeline
│   ├── feature_engine.py      # 70+ engineered features (zero leakage)
│   └── insight_generator.py   # NL insights + toss advisor + strategy engine
├── ml/
│   ├── train.py               # XGB+LGB+CB+Stacking | Optuna | SHAP | Calibration
│   └── weakness_detector.py   # Phase/matchup/bowler weakness analysis
├── simulation/
│   └── monte_carlo.py         # 10,000 simulations | CI | Score distributions
├── optimizer/
│   └── fantasy_optimizer.py   # ILP Dream11 optimizer (3 strategies)
├── api/
│   └── main.py                # FastAPI — 15 production endpoints
├── app/
│   └── app.py                 # Streamlit — 18-page analytics dashboard
├── data/
│   ├── raw/IPL.csv            # Source dataset
│   ├── cleaned/               # Matches + Deliveries (CSV + Parquet)
│   ├── features/              # Engineered features (Parquet)
│   ├── models/                # Trained model PKLs + metrics.json
│   └── shap/                  # SHAP explainers + importance CSVs
├── init_db.py                 # SQLite schema setup
├── requirements.txt
├── Dockerfile
└── README.md
    """, language=None)

    section("🚀 Quick Start")
    st.code("""
# 1. Place IPL.csv → data/raw/IPL.csv
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run ETL pipeline
python etl/data_cleaning.py
python etl/feature_engine.py
python init_db.py

# 4. Train models (~5-15 min, GPU optional)
python ml/train.py

# 5. Launch Streamlit dashboard
streamlit run app/app.py

# 6. Launch API (optional)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    """, language="bash")

    st.markdown("📂 **Data:** [Kaggle — IPL Dataset 2008-2025](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025)")

# ─────────────────────────────────────────────────────────
# SIDEBAR + ROUTER
# ─────────────────────────────────────────────────────────

def _decision_engine():
    page_decision_engine(load_matches(), load_deliveries())

def _pressure_momentum():
    page_pressure_momentum(load_matches(), load_deliveries())

def _player_deep():
    page_player_deep(load_matches(), load_deliveries())

def _fantasy_elite():
    page_fantasy_elite(load_matches(), load_deliveries(), optimize_team_ilp, estimate_player_fp)

def _venue_advanced():
    page_venue_advanced(load_matches(), load_deliveries())

def _backtesting():
    page_backtesting(load_matches(), load_metrics)

# ── Wrappers for upgraded pages (pass data from cached loaders) ──
def _win_prob():
    page_win_probability_v2(load_matches(), load_deliveries())

def _run_heatmap():
    page_run_heatmap_v2(load_matches(), load_deliveries())

def _season_table():
    page_season_table_v2(load_matches())

def _form_tracker():
    page_form_tracker_v2(load_matches(), load_deliveries())

def _breakout():
    page_breakout_v2(load_matches(), load_deliveries())

def _fantasy():
    page_fantasy_v2(load_matches(), load_deliveries(), optimize_team_ilp, estimate_player_fp)

def _rankings():
    page_rankings_v2(load_matches(), load_deliveries())

def _venue():
    page_venue_analysis_v2(load_matches(), load_deliveries())

def _similarity():
    page_player_similarity_v2(load_matches(), load_deliveries())

def _about():
    page_about_v2(load_metrics)


# ── Phase 2-12 page wrappers ──
def _elite_win_prob():
    page_pressure_momentum(load_matches(), load_deliveries())

def _elite_form():
    page_form_tracker_v2(load_matches(), load_deliveries())

def _elite_fantasy():
    page_fantasy_elite(load_matches(), load_deliveries(), optimize_team_ilp, estimate_player_fp)

def _elite_venue():
    page_venue_advanced(load_matches(), load_deliveries())

def _system_identity():
    page_about_v2(load_metrics)

PAGES = {
    "🏆 League Overview":        page_overview,
    "🔍 Player Profile":         page_player_search,
    "⚔️ Player Comparison":      page_player_comparison,
    "🥊 Batsman vs Bowler":      page_matchup,
    "🤝 Partnership Tracker":    page_partnerships,
    "⚔️ Team Comparison":        page_team_comparison,
    "🤖 AI Match Predictor":     page_match_predictor,
    "🎯 Toss Advisor":           page_toss_advisor,
    "📈 Win Probability":        _win_prob,
    "📋 Season Points Table":    _season_table,
    "🌡️ Run Heatmap":            _run_heatmap,
    "📊 Form Tracker":           _form_tracker,
    "⚡ Breakout Players":       _breakout,
    "💰 Fantasy Optimizer":      _fantasy,
    "🏅 Player Rankings":        _rankings,
    "🏟️ Venue Analysis":         _venue,
    "🧬 Player Similarity":      _similarity,
    "📖 About & Model Info":     _about,
    # ── Elite AI Intelligence Pages ──
    "📈 Elite Win Probability":  _elite_win_prob,
    "📊 Elite Player Form":      _elite_form,
    "💎 Elite Fantasy":          _elite_fantasy,
    "🏟️ Elite Venue Intelligence": _elite_venue,
    "🧠 System Identity":        _about,
    "🧠 Decision Engine":        _decision_engine,
    "⚡ Pressure & Momentum":    _pressure_momentum,
    "🔬 Deep Player Analytics":  _player_deep,
    "💎 Elite Fantasy Teams":    _fantasy_elite,
    "🏟️ Venue DNA Advanced":     _venue_advanced,
    "📊 Backtesting Dashboard":  _backtesting,
}

def main():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">🏏 CricketBrain AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">Production-Grade IPL Intelligence Platform</div>',
                    unsafe_allow_html=True)
        st.divider()

        # Group pages
        st.markdown("**📊 Analytics**")
        page_groups = {
            "📊 Analytics": [
                "🏆 League Overview", "📋 Season Points Table",
                "🌡️ Run Heatmap", "🏟️ Venue Analysis"
            ],
            "👤 Players": [
                "🔍 Player Profile", "⚔️ Player Comparison",
                "🥊 Batsman vs Bowler", "🤝 Partnership Tracker",
                "📊 Form Tracker", "⚡ Breakout Players",
                "🏅 Player Rankings", "🧬 Player Similarity"
            ],
            "🏏 Teams": [
                "⚔️ Team Comparison"
            ],
            "🤖 AI / Intelligence": [
                "🤖 AI Match Predictor", "🎯 Toss Advisor",
                "📈 Win Probability", "💰 Fantasy Optimizer"
            ],
            "📖 Info": [
                "📖 About & Model Info"
            ],
        }

        # Filter out separator entries (None values)
        page_keys = [k for k, v in PAGES.items() if v is not None]
        selected = st.radio(
            "Navigate",
            page_keys,
            label_visibility="collapsed"
        )

        st.divider()
        matches = load_matches()
        if not matches.empty:
            st.caption(f"📦 {len(matches):,} matches | {matches['season'].nunique()} seasons")
            st.caption(f"🏏 {len(get_teams(matches))} teams | {matches['venue'].nunique()} venues")
        else:
            st.warning("⚠️ Data not loaded\nRun `python etl/data_cleaning.py`")

    fn = PAGES.get(selected)
    if fn is not None:
        fn()

if __name__ == "__main__":
    main()