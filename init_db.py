"""
CricketBrain AI — Database Initializer
Creates SQLite schema from cleaned CSVs
"""

import os, sqlite3
import pandas as pd

BASE    = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE, "data", "cricketbrain.db")
CLEANED = os.path.join(BASE, "data", "cleaned")

def create_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Schema
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id       INTEGER PRIMARY KEY,
        season         INTEGER,
        date           TEXT,
        city           TEXT,
        venue          TEXT,
        team1          TEXT,
        team2          TEXT,
        toss_winner    TEXT,
        toss_decision  TEXT,
        winner         TEXT,
        result         TEXT,
        result_margin  REAL,
        batting_first  TEXT,
        fielding_first TEXT,
        team1_won      INTEGER,
        bat_first_won  INTEGER
    );

    CREATE TABLE IF NOT EXISTS deliveries (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id       INTEGER,
        season         INTEGER,
        batting_team   TEXT,
        bowling_team   TEXT,
        over_num       INTEGER,
        ball_in_over   INTEGER,
        phase          TEXT,
        batter         TEXT,
        non_striker    TEXT,
        bowler         TEXT,
        batsman_runs   INTEGER,
        extra_runs     INTEGER,
        total_runs     INTEGER,
        is_wide        INTEGER,
        is_noball      INTEGER,
        is_legal       INTEGER,
        is_wicket      INTEGER,
        player_dismissed TEXT,
        dismissal_kind TEXT,
        FOREIGN KEY (match_id) REFERENCES matches(match_id)
    );

    CREATE INDEX IF NOT EXISTS idx_del_match  ON deliveries(match_id);
    CREATE INDEX IF NOT EXISTS idx_del_batter ON deliveries(batter);
    CREATE INDEX IF NOT EXISTS idx_del_bowler ON deliveries(bowler);
    CREATE INDEX IF NOT EXISTS idx_del_team   ON deliveries(batting_team);
    CREATE INDEX IF NOT EXISTS idx_del_season ON deliveries(season);
    CREATE INDEX IF NOT EXISTS idx_match_team ON matches(team1, team2);
    CREATE INDEX IF NOT EXISTS idx_match_venue ON matches(venue);
    """)
    conn.commit()

    # Load CSVs
    m_path = os.path.join(CLEANED, "matches.csv")
    d_path = os.path.join(CLEANED, "deliveries.csv")

    if os.path.exists(m_path):
        matches = pd.read_csv(m_path)
        m_cols = [c for c in [
            "match_id","season","date","city","venue","team1","team2",
            "toss_winner","toss_decision","winner","result","result_margin",
            "batting_first","fielding_first","team1_won","bat_first_won"
        ] if c in matches.columns]
        matches[m_cols].to_sql("matches", conn, if_exists="replace", index=False)
        print(f"[DB] Loaded {len(matches)} matches")

    if os.path.exists(d_path):
        deliveries = pd.read_csv(d_path)
        d_cols = [c for c in [
            "match_id","season","batting_team","bowling_team","over_num","ball_in_over",
            "phase","batter","non_striker","bowler","batsman_runs","extra_runs","total_runs",
            "is_wide","is_noball","is_legal","is_wicket","player_dismissed","dismissal_kind"
        ] if c in deliveries.columns]
        deliveries[d_cols].to_sql("deliveries", conn, if_exists="replace", index=False)
        print(f"[DB] Loaded {len(deliveries)} deliveries")

    conn.close()
    print(f"[DB] ✅ Database created: {DB_PATH}")

if __name__ == "__main__":
    create_db()
