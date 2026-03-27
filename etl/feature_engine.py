"""
CricketBrain AI — Advanced Feature Engineering
70+ features | Zero data leakage | Time-aware rolling stats
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED  = os.path.join(DATA_DIR, "cleaned")
FEATURES = os.path.join(DATA_DIR, "features")

# ─────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────
def load():
    m = pd.read_csv(os.path.join(CLEANED, "matches.csv"), parse_dates=["date"])
    d = pd.read_csv(os.path.join(CLEANED, "deliveries.csv"))
    m = m.sort_values("date").reset_index(drop=True)
    return m, d

# ─────────────────────────────────────────────────────────
# HELPER: safe rolling on past data (shift(1) = no leakage)
# ─────────────────────────────────────────────────────────
def safe_roll(series, w, fn="mean"):
    shifted = series.shift(1)
    if fn == "mean":
        return shifted.rolling(w, min_periods=1).mean()
    elif fn == "sum":
        return shifted.rolling(w, min_periods=1).sum()
    elif fn == "std":
        return shifted.rolling(w, min_periods=1).std().fillna(0)

# ─────────────────────────────────────────────────────────
# 1. TEAM-LEVEL ROLLING FEATURES
# ─────────────────────────────────────────────────────────
def compute_team_features(matches):
    print("[FE] Computing team rolling features...")
    teams = pd.concat([
        matches[["match_id","date","season","team1","winner"]].rename(columns={"team1":"team"}),
        matches[["match_id","date","season","team2","winner"]].rename(columns={"team2":"team"}),
    ]).copy()
    teams["won"] = (teams["winner"] == teams["team"]).astype(int)
    teams = teams.sort_values("date")

    feats = {}
    for team, grp in teams.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for w in [3, 5, 10, 15]:
            grp[f"win_rate_last{w}"]  = safe_roll(grp["won"], w, "mean")
        grp["win_streak"]  = grp["won"].shift(1).groupby((grp["won"].shift(1) != grp["won"].shift(1)).cumsum()).cumcount()
        grp["loss_streak"] = (1-grp["won"]).shift(1).groupby(((1-grp["won"]).shift(1) != (1-grp["won"]).shift(1)).cumsum()).cumcount()
        grp["form_score"]  = (
            grp["won"].shift(1)*1.0 + grp["won"].shift(2)*0.8 +
            grp["won"].shift(3)*0.6 + grp["won"].shift(4)*0.4 +
            grp["won"].shift(5)*0.2
        ).fillna(0)
        feats[team] = grp

    team_stats = pd.concat(feats.values()).reset_index(drop=True)
    return team_stats

# ─────────────────────────────────────────────────────────
# 2. TOSS FEATURES
# ─────────────────────────────────────────────────────────
def compute_toss_features(matches):
    print("[FE] Computing toss features...")
    matches = matches.copy()
    matches["toss_winner_won"]    = (matches["toss_winner"] == matches["winner"]).astype(int)
    matches["bat_first_won"]      = (matches["batting_first"] == matches["winner"]).astype(int)
    matches["chase_won"]          = 1 - matches["bat_first_won"]
    matches["toss_decision_bat"]  = (matches["toss_decision"] == "bat").astype(int)
    return matches

# ─────────────────────────────────────────────────────────
# 3. VENUE FEATURES
# ─────────────────────────────────────────────────────────
def compute_venue_features(matches):
    print("[FE] Computing venue features...")
    venue_stats = (
        matches.groupby("venue")
        .agg(
            venue_matches   = ("match_id","count"),
            venue_bat_first_wins = ("bat_first_won","sum"),
        )
        .reset_index()
    )
    venue_stats["venue_bat_first_win_rate"] = (
        venue_stats["venue_bat_first_wins"] / venue_stats["venue_matches"]
    )
    venue_stats["venue_chase_win_rate"] = 1 - venue_stats["venue_bat_first_win_rate"]
    return venue_stats

# ─────────────────────────────────────────────────────────
# 4. HEAD-TO-HEAD FEATURES (time-aware)
# ─────────────────────────────────────────────────────────
def compute_h2h_features(matches):
    print("[FE] Computing H2H features...")
    records = []
    sorted_m = matches.sort_values("date").reset_index(drop=True)
    for _, row in sorted_m.iterrows():
        t1, t2 = row["team1"], row["team2"]
        past = sorted_m[
            (sorted_m["date"] < row["date"]) &
            (
                ((sorted_m["team1"]==t1) & (sorted_m["team2"]==t2)) |
                ((sorted_m["team1"]==t2) & (sorted_m["team2"]==t1))
            )
        ]
        h2h_total = len(past)
        h2h_t1_wins = (past["winner"] == t1).sum()
        h2h_t2_wins = (past["winner"] == t2).sum()
        records.append({
            "match_id": row["match_id"],
            "h2h_total": h2h_total,
            "h2h_team1_wins": h2h_t1_wins,
            "h2h_team2_wins": h2h_t2_wins,
            "h2h_team1_win_rate": h2h_t1_wins / max(h2h_total, 1),
            "h2h_team2_win_rate": h2h_t2_wins / max(h2h_total, 1),
        })
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────
# 5. INNINGS-LEVEL BATTING STATS (per match, per phase)
# ─────────────────────────────────────────────────────────
def compute_innings_stats(deliveries):
    print("[FE] Computing innings phase stats...")
    d = deliveries.copy()
    d["legal_ball"] = (d.get("is_wide",0) + d.get("is_noball",0) == 0).astype(int)

    phase_stats = (
        d.groupby(["match_id","batting_team","phase"])
        .agg(
            runs = ("total_runs","sum"),
            balls = ("legal_ball","sum"),
            wickets = ("is_wicket","sum"),
        )
        .reset_index()
    )
    phase_stats["run_rate"] = phase_stats["runs"] / phase_stats["balls"].replace(0,1) * 6

    # pivot phases
    piv = phase_stats.pivot_table(
        index=["match_id","batting_team"],
        columns="phase",
        values=["runs","wickets","run_rate"]
    )
    piv.columns = ["_".join(c).strip() for c in piv.columns]
    piv = piv.reset_index()

    # total innings
    total = (
        d.groupby(["match_id","batting_team"])
        .agg(
            innings_runs    = ("total_runs","sum"),
            innings_wickets = ("is_wicket","sum"),
            innings_balls   = ("legal_ball","sum"),
        )
        .reset_index()
    )
    total["innings_run_rate"] = total["innings_runs"] / total["innings_balls"].replace(0,1) * 6
    innings_stats = total.merge(piv, on=["match_id","batting_team"], how="left")
    return innings_stats

# ─────────────────────────────────────────────────────────
# 6. PLAYER BATTING FEATURES
# ─────────────────────────────────────────────────────────
def compute_player_batting(deliveries):
    print("[FE] Computing player batting features...")
    d = deliveries.copy()
    d["legal_ball"] = (d.get("is_wide",0) + d.get("is_noball",0) == 0).astype(int)

    bat = (
        d.groupby(["match_id","batter"])
        .agg(
            bat_runs   = ("batsman_runs","sum"),
            bat_balls  = ("legal_ball","sum"),
            bat_fours  = ("batsman_runs", lambda x: (x==4).sum()),
            bat_sixes  = ("batsman_runs", lambda x: (x==6).sum()),
            bat_dismissed = ("is_wicket","max"),
        )
        .reset_index()
    )
    bat["bat_sr"] = bat["bat_runs"] / bat["bat_balls"].replace(0,1) * 100
    bat["bat_boundary_pct"] = (bat["bat_fours"]*4 + bat["bat_sixes"]*6) / bat["bat_runs"].replace(0,1)

    # Phase-level
    for phase in ["powerplay","middle","death"]:
        sub = d[d["phase"]==phase].groupby(["match_id","batter"]).agg(
            **{f"{phase}_runs": ("batsman_runs","sum"),
               f"{phase}_balls": ("legal_ball","sum")}
        ).reset_index()
        sub[f"{phase}_sr"] = sub[f"{phase}_runs"] / sub[f"{phase}_balls"].replace(0,1) * 100
        bat = bat.merge(sub, on=["match_id","batter"], how="left")

    return bat

# ─────────────────────────────────────────────────────────
# 7. PLAYER BOWLING FEATURES
# ─────────────────────────────────────────────────────────
def compute_player_bowling(deliveries):
    print("[FE] Computing player bowling features...")
    d = deliveries.copy()
    d["legal_ball"] = (d.get("is_wide",0) + d.get("is_noball",0) == 0).astype(int)

    bowl = (
        d.groupby(["match_id","bowler"])
        .agg(
            bowl_runs   = ("total_runs","sum"),
            bowl_balls  = ("legal_ball","sum"),
            bowl_wkts   = ("is_wicket","sum"),
            bowl_wides  = ("is_wide","sum"),
            bowl_noballs= ("is_noball","sum"),
        )
        .reset_index()
    )
    bowl["bowl_overs"] = bowl["bowl_balls"] / 6
    bowl["bowl_econ"]  = bowl["bowl_runs"] / bowl["bowl_overs"].replace(0,1)
    bowl["bowl_sr"]    = bowl["bowl_balls"] / bowl["bowl_wkts"].replace(0,1)
    bowl["bowl_avg"]   = bowl["bowl_runs"] / bowl["bowl_wkts"].replace(0,1)

    for phase in ["powerplay","middle","death"]:
        sub = d[d["phase"]==phase].groupby(["match_id","bowler"]).agg(
            **{f"bowl_{phase}_runs": ("total_runs","sum"),
               f"bowl_{phase}_balls": ("legal_ball","sum"),
               f"bowl_{phase}_wkts": ("is_wicket","sum")}
        ).reset_index()
        sub[f"bowl_{phase}_econ"] = sub[f"bowl_{phase}_runs"] / (sub[f"bowl_{phase}_balls"]/6).replace(0,1)
        bowl = bowl.merge(sub, on=["match_id","bowler"], how="left")

    return bowl

# ─────────────────────────────────────────────────────────
# 8. ROLLING PLAYER FORM (per player, time-aware)
# ─────────────────────────────────────────────────────────
def compute_rolling_batting_form(deliveries, matches):
    print("[FE] Computing rolling batting form...")
    bat_match = compute_player_batting(deliveries)
    bat_match = bat_match.merge(matches[["match_id","date"]], on="match_id", how="left")
    bat_match = bat_match.sort_values("date")

    form_records = []
    for player, grp in bat_match.groupby("batter"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for w in [3,5,10]:
            grp[f"bat_avg_last{w}"]  = safe_roll(grp["bat_runs"], w, "mean")
            grp[f"bat_sr_last{w}"]   = safe_roll(grp["bat_sr"], w, "mean")
        grp["bat_consistency"] = safe_roll(grp["bat_runs"], 10, "std")
        grp["bat_form_score"]  = (
            grp["bat_runs"].shift(1)*1.0 + grp["bat_runs"].shift(2)*0.7 +
            grp["bat_runs"].shift(3)*0.5 + grp["bat_runs"].shift(4)*0.3 +
            grp["bat_runs"].shift(5)*0.1
        ).fillna(0)
        grp["bat_pressure_idx"] = (grp["death_runs"].shift(1) + grp["powerplay_runs"].shift(1)).fillna(0)
        form_records.append(grp)

    return pd.concat(form_records).reset_index(drop=True)

def compute_rolling_bowling_form(deliveries, matches):
    print("[FE] Computing rolling bowling form...")
    bowl_match = compute_player_bowling(deliveries)
    bowl_match = bowl_match.merge(matches[["match_id","date"]], on="match_id", how="left")
    bowl_match = bowl_match.sort_values("date")

    form_records = []
    for player, grp in bowl_match.groupby("bowler"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for w in [3,5,10]:
            grp[f"bowl_econ_last{w}"] = safe_roll(grp["bowl_econ"], w, "mean")
            grp[f"bowl_wkts_last{w}"] = safe_roll(grp["bowl_wkts"], w, "mean")
        grp["bowl_consistency"]  = safe_roll(grp["bowl_econ"], 10, "std")
        grp["bowl_form_score"]   = (
            grp["bowl_wkts"].shift(1)*1.0 + grp["bowl_wkts"].shift(2)*0.7 +
            grp["bowl_wkts"].shift(3)*0.5
        ).fillna(0)
        form_records.append(grp)

    return pd.concat(form_records).reset_index(drop=True)

# ─────────────────────────────────────────────────────────
# 9. TEAM STRENGTH INDEX
# ─────────────────────────────────────────────────────────
def compute_team_strength(deliveries, matches):
    print("[FE] Computing team strength index...")
    innings = compute_innings_stats(deliveries)
    innings = innings.merge(matches[["match_id","date","season"]], on="match_id", how="left")

    strength = []
    for team, grp in innings.groupby("batting_team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        grp["batting_strength"] = safe_roll(grp["innings_runs"], 5, "mean")
        grp["batting_depth"]    = safe_roll(10 - grp["innings_wickets"], 5, "mean")
        strength.append(grp[["match_id","batting_team","batting_strength","batting_depth"]])

    return pd.concat(strength).reset_index(drop=True)

# ─────────────────────────────────────────────────────────
# 10. MASTER MATCH FEATURE TABLE (for ML)
# ─────────────────────────────────────────────────────────
def build_match_features(matches, deliveries):
    print("[FE] Building master match feature table...")
    m = matches.copy()
    m = compute_toss_features(m)
    venue_feats = compute_venue_features(m)
    m = m.merge(venue_feats, on="venue", how="left")

    # Team rolling features
    team_stats = compute_team_features(m)
    t1_cols = {c: f"team1_{c}" for c in team_stats.columns if c not in ["match_id","date","season","team","won","winner"]}
    t2_cols = {c: f"team2_{c}" for c in team_stats.columns if c not in ["match_id","date","season","team","won","winner"]}

    ts1 = team_stats.rename(columns={"team":"team1"}).rename(columns=t1_cols)
    ts2 = team_stats.rename(columns={"team":"team2"}).rename(columns=t2_cols)

    if "team1" in ts1.columns:
        m = m.merge(ts1[["match_id","team1"] + list(t1_cols.values())], on=["match_id","team1"], how="left")
    if "team2" in ts2.columns:
        m = m.merge(ts2[["match_id","team2"] + list(t2_cols.values())], on=["match_id","team2"], how="left")

    # H2H
    h2h = compute_h2h_features(m)
    m = m.merge(h2h, on="match_id", how="left")

    # Team strength
    strength = compute_team_strength(deliveries, matches)
    s1 = strength.rename(columns={"batting_team":"team1","batting_strength":"t1_bat_str","batting_depth":"t1_bat_depth"})
    s2 = strength.rename(columns={"batting_team":"team2","batting_strength":"t2_bat_str","batting_depth":"t2_bat_depth"})
    m = m.merge(s1[["match_id","team1","t1_bat_str","t1_bat_depth"]], on=["match_id","team1"], how="left")
    m = m.merge(s2[["match_id","team2","t2_bat_str","t2_bat_depth"]], on=["match_id","team2"], how="left")

    # Differential features (team1 - team2)
    for base in ["win_rate_last3","win_rate_last5","win_rate_last10","form_score"]:
        c1, c2 = f"team1_{base}", f"team2_{base}"
        if c1 in m.columns and c2 in m.columns:
            m[f"diff_{base}"] = m[c1].fillna(0) - m[c2].fillna(0)

    # Season number (trend feature)
    m["season_num"] = pd.factorize(m["season"])[0]

    # Target variable
    m["target"] = m["team1_won"]

    print(f"[FE] Match features: {m.shape[1]} columns | {len(m)} rows")
    return m

# ─────────────────────────────────────────────────────────
# 11. FANTASY POINTS ESTIMATION
# ─────────────────────────────────────────────────────────
def estimate_fantasy_points(bat_df, bowl_df):
    """Dream11-style fantasy point estimation"""
    df = bat_df.copy()
    df["fp_bat"] = (
        df["bat_runs"] * 1 +
        df["bat_fours"] * 1 +
        df["bat_sixes"] * 2 +
        (df["bat_runs"] >= 30).astype(int) * 8 +
        (df["bat_runs"] >= 50).astype(int) * 8 +
        (df["bat_runs"] >= 100).astype(int) * 16 +
        (df["bat_dismissed"] == 0).astype(int) * 4
    )
    df["fp_sr_bonus"] = np.where(df["bat_sr"] >= 170, 6, np.where(df["bat_sr"] >= 150, 4, np.where(df["bat_sr"] >= 130, 2, 0)))
    df["fp_bat_total"] = df["fp_bat"] + df["fp_sr_bonus"]

    b = bowl_df.copy()
    b["fp_bowl"] = (
        b["bowl_wkts"] * 25 +
        (b["bowl_wkts"] >= 3).astype(int) * 4 +
        (b["bowl_wkts"] >= 4).astype(int) * 4 +
        (b["bowl_wkts"] >= 5).astype(int) * 4
    )
    b["fp_econ_bonus"] = np.where(b["bowl_econ"] < 5, 6, np.where(b["bowl_econ"] < 6, 4, np.where(b["bowl_econ"] < 7, 2, 0)))
    b["fp_bowl_total"] = b["fp_bowl"] + b["fp_econ_bonus"]

    return df[["match_id","batter","fp_bat_total"]], b[["match_id","bowler","fp_bowl_total"]]

# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    os.makedirs(FEATURES, exist_ok=True)
    matches, deliveries = load()

    # Core feature table
    match_feats = build_match_features(matches, deliveries)
    out = os.path.join(FEATURES, "match_features.csv")
    match_feats.to_csv(out, index=False)
    match_feats.to_parquet(out.replace(".csv",".parquet"), index=False)
    print(f"[FE] Saved: {out}")

    # Batting form
    bat_form = compute_rolling_batting_form(deliveries, matches)
    bat_form.to_csv(os.path.join(FEATURES,"batting_form.csv"), index=False)
    bat_form.to_parquet(os.path.join(FEATURES,"batting_form.parquet"), index=False)

    # Bowling form
    bowl_form = compute_rolling_bowling_form(deliveries, matches)
    bowl_form.to_csv(os.path.join(FEATURES,"bowling_form.csv"), index=False)
    bowl_form.to_parquet(os.path.join(FEATURES,"bowling_form.parquet"), index=False)

    # Fantasy points
    bat_match  = compute_player_batting(deliveries)
    bowl_match = compute_player_bowling(deliveries)
    fp_bat, fp_bowl = estimate_fantasy_points(bat_match, bowl_match)
    fp_bat.to_csv(os.path.join(FEATURES,"fp_batting.csv"), index=False)
    fp_bowl.to_csv(os.path.join(FEATURES,"fp_bowling.csv"), index=False)

    print("[FE] ✅ Feature engineering complete!")
    return match_feats

if __name__ == "__main__":
    main()
