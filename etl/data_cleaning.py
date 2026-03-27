"""
CricketBrain AI — Data Cleaning Pipeline
Written specifically for the IPL.csv (278205 rows, 64 columns) dataset.
Exact column names confirmed from user's terminal output.
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Delhi Capitals": "Delhi Capitals",
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Pune Warriors": "Pune Warriors",
    "Kings XI Punjab": "Punjab Kings",
    "Punjab Kings": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Mumbai Indians": "Mumbai Indians",
    "Chennai Super Kings": "Chennai Super Kings",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Rajasthan Royals": "Rajasthan Royals",
    "Gujarat Titans": "Gujarat Titans",
    "Lucknow Super Giants": "Lucknow Super Giants",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",
}

VENUE_MAP = {
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "M. Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Wankhede Stadium": "Wankhede Stadium",
    "Eden Gardens": "Eden Gardens",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Feroz Shah Kotla Ground": "Arun Jaitley Stadium",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
    "Narendra Modi Stadium": "Narendra Modi Stadium",
    "Sardar Patel Stadium": "Narendra Modi Stadium",
    "DY Patil Stadium": "DY Patil Stadium",
    "Brabourne Stadium": "Brabourne Stadium",
    "Maharashtra Cricket Association Stadium": "Maharashtra Cricket Association Stadium",
}


def load_raw(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "raw", "IPL.csv")
    print(f"[ETL] Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"[ETL] Shape: {df.shape}")
    return df


def build_clean_df(raw):
    """
    Select and rename ONLY the columns we need using their EXACT names
    from this specific dataset. No renaming ambiguity.

    Confirmed columns in this dataset:
      match_id, date, innings, batting_team, bowling_team,
      over (float 0.0-19.5), ball (int 1-6), ball_no,
      batter, bowler, non_striker,
      runs_batter, runs_extras, runs_total, valid_ball,
      wicket_kind, player_out, fielders,
      match_won_by, toss_winner, toss_decision,
      venue, city, month, year, season (=year duplicate),
      player_of_match, win_outcome, result_type, method
    """

    df = pd.DataFrame()

    # ── Identity ──
    df["match_id"]      = raw["match_id"]
    df["innings"]       = raw["innings"] if "innings" in raw.columns else 1

    # ── Date / Season ──
    df["date"]          = pd.to_datetime(raw["date"], dayfirst=True, errors="coerce")
    # Use 'year' column directly — it's an int, no need to parse
    df["year"]          = pd.to_numeric(raw["year"], errors="coerce")
    df["month"]         = pd.to_numeric(raw["month"], errors="coerce") if "month" in raw.columns else df["date"].dt.month
    df["season"]        = df["year"].astype("Int64")

    # ── Teams ──
    df["batting_team"]  = raw["batting_team"].astype(str).str.strip()
    df["bowling_team"]  = raw["bowling_team"].astype(str).str.strip()

    # ── Over / Ball ──
    # 'over' column is float like 0.0, 0.1 ... 19.5 (over.ball_in_over)
    over_raw            = pd.to_numeric(raw["over"], errors="coerce").fillna(0)
    df["over_num"]      = over_raw.astype(int)
    df["ball_in_over"]  = pd.to_numeric(raw["ball"], errors="coerce").fillna(0).astype(int)
    df["ball_number"]   = df["over_num"] * 6 + df["ball_in_over"]

    # ── Phase ──
    df["phase"] = pd.cut(
        df["over_num"],
        bins=[-1, 5, 14, 20],
        labels=["powerplay", "middle", "death"]
    )

    # ── Players ──
    df["batter"]        = raw["batter"].astype(str).str.strip()
    df["bowler"]        = raw["bowler"].astype(str).str.strip()
    df["non_striker"]   = raw["non_striker"].astype(str).str.strip() if "non_striker" in raw.columns else ""

    # ── Runs ──
    df["batsman_runs"]  = pd.to_numeric(raw["runs_batter"],  errors="coerce").fillna(0).astype(int)
    df["extra_runs"]    = pd.to_numeric(raw["runs_extras"],  errors="coerce").fillna(0).astype(int)
    df["total_runs"]    = pd.to_numeric(raw["runs_total"],   errors="coerce").fillna(0).astype(int)

    # ── Legal ball ──
    # valid_ball = 1 means legal delivery
    df["is_legal"]      = pd.to_numeric(raw["valid_ball"],   errors="coerce").fillna(1).astype(int)
    df["is_wide"]       = 0
    df["is_noball"]     = 0
    if "extra_type" in raw.columns:
        df["is_wide"]   = (raw["extra_type"].astype(str).str.lower() == "wides").astype(int)
        df["is_noball"] = (raw["extra_type"].astype(str).str.lower() == "noballs").astype(int)

    # ── Wicket ──
    # wicket_kind is non-null when a wicket falls
    df["is_wicket"]     = raw["wicket_kind"].notna().astype(int) if "wicket_kind" in raw.columns else 0
    df["dismissal_kind"]= raw["wicket_kind"].astype(str).where(raw["wicket_kind"].notna(), "") if "wicket_kind" in raw.columns else ""
    df["player_dismissed"] = raw["player_out"].astype(str).where(raw["player_out"].notna(), "") if "player_out" in raw.columns else ""

    # ── Match metadata (repeated per ball, will be deduped) ──
    df["winner"]        = raw["match_won_by"].astype(str).str.strip() if "match_won_by" in raw.columns else ""
    df["toss_winner"]   = raw["toss_winner"].astype(str).str.strip()  if "toss_winner"  in raw.columns else ""
    df["toss_decision"] = raw["toss_decision"].astype(str).str.strip() if "toss_decision" in raw.columns else ""
    df["venue"]         = raw["venue"].astype(str).str.strip()         if "venue"         in raw.columns else ""
    df["city"]          = raw["city"].astype(str).str.strip()          if "city"          in raw.columns else ""
    df["player_of_match"] = raw["player_of_match"].astype(str).str.strip() if "player_of_match" in raw.columns else ""
    df["result"]        = raw["result_type"].astype(str).str.strip()   if "result_type"  in raw.columns else ""

    return df


def apply_team_map(df):
    for col in ["batting_team","bowling_team","toss_winner","winner"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: TEAM_MAP.get(x, x) if pd.notna(x) else x)
    return df


def apply_venue_map(df):
    if "venue" in df.columns:
        df["venue"] = df["venue"].map(lambda x: VENUE_MAP.get(x, x) if pd.notna(x) else x)
    return df


def infer_team1_team2(df):
    """Infer team1/team2 from innings 1 batting/bowling teams"""
    inn1 = df[df["innings"] == 1]
    t1 = inn1.groupby("match_id")["batting_team"].first().rename("team1")
    t2 = inn1.groupby("match_id")["bowling_team"].first().rename("team2")
    df = df.merge(t1.reset_index(), on="match_id", how="left")
    df = df.merge(t2.reset_index(), on="match_id", how="left")
    print(f"[ETL] Inferred team1/team2 for {t1.shape[0]} matches")
    return df


def build_matches_df(df):
    """One row per match"""
    m = df.drop_duplicates("match_id")[[
        c for c in [
            "match_id","season","date","year","city","venue",
            "team1","team2","toss_winner","toss_decision",
            "winner","result","player_of_match"
        ] if c in df.columns
    ]].copy().sort_values("date").reset_index(drop=True)

    # toss_won_by_team1
    if "toss_winner" in m.columns and "team1" in m.columns:
        m["toss_won_by_team1"] = (m["toss_winner"] == m["team1"]).astype(int)
    else:
        m["toss_won_by_team1"] = 0

    # batting_first / fielding_first
    if "toss_decision" in m.columns and "team1" in m.columns and "team2" in m.columns:
        td = m["toss_decision"].str.lower().str.strip()
        tw = m["toss_winner"]
        t1 = m["team1"]
        t2 = m["team2"]
        cond = ((td == "bat") & (tw == t1)) | ((td == "field") & (tw == t2))
        m["batting_first"]  = np.where(cond, t1, t2)
        m["fielding_first"] = np.where(cond, t2, t1)
    else:
        m["batting_first"]  = m.get("team1", "")
        m["fielding_first"] = m.get("team2", "")

    if "winner" in m.columns and "team1" in m.columns:
        m["team1_won"]     = (m["winner"] == m["team1"]).astype(int)
    else:
        m["team1_won"] = 0

    if "winner" in m.columns and "batting_first" in m.columns:
        m["bat_first_won"] = (m["winner"] == m["batting_first"]).astype(int)
    else:
        m["bat_first_won"] = 0

    seasons = sorted(m["season"].dropna().unique().tolist()) if "season" in m.columns else []
    print(f"[ETL] Matches: {len(m)} | Seasons: {seasons}")
    return m


def build_deliveries_df(df):
    keep = [c for c in [
        "match_id","season","date",
        "batting_team","bowling_team",
        "over_num","ball_in_over","ball_number","phase",
        "batter","non_striker","bowler",
        "batsman_runs","extra_runs","total_runs",
        "is_wide","is_noball","is_legal","is_wicket",
        "player_dismissed","dismissal_kind",
    ] if c in df.columns]
    d = df[keep].copy()
    print(f"[ETL] Deliveries: {len(d):,} balls")
    return d


def save_data(matches, deliveries):
    out = os.path.join(DATA_DIR, "cleaned")
    os.makedirs(out, exist_ok=True)
    m_p = os.path.join(out, "matches.csv")
    d_p = os.path.join(out, "deliveries.csv")
    matches.to_csv(m_p, index=False)
    deliveries.to_csv(d_p, index=False)
    matches.to_parquet(m_p.replace(".csv", ".parquet"), index=False)
    deliveries.to_parquet(d_p.replace(".csv", ".parquet"), index=False)
    print(f"[ETL] Saved: {m_p}")
    print(f"[ETL] Saved: {d_p}")


def main():
    raw        = load_raw()
    df         = build_clean_df(raw)       # select + rename exact columns
    df         = apply_team_map(df)
    df         = apply_venue_map(df)
    df         = infer_team1_team2(df)
    matches    = build_matches_df(df)
    deliveries = build_deliveries_df(df)
    save_data(matches, deliveries)
    print("[ETL] ✅ Data cleaning complete!")
    return matches, deliveries


if __name__ == "__main__":
    main()