"""
CricketBrain AI — Advanced Feature Engineering
Builds 50+ features with strict time-based validation to prevent data leakage.
Every feature uses ONLY data available BEFORE that match was played.
"""
import pandas as pd
import numpy as np
import os, logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CLEAN_DIR   = "data/cleaned"
FEATURE_DIR = "data/features"

# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────
def load():
    m = pd.read_csv(f"{CLEAN_DIR}/matches_clean.csv", low_memory=False)
    d = pd.read_csv(f"{CLEAN_DIR}/deliveries_clean.csv", low_memory=False)
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.sort_values("date").reset_index(drop=True)
    if "season" not in m.columns:
        m["season"] = m["date"].dt.year
    log.info(f"Loaded {len(m):,} matches · {len(d):,} deliveries")
    return m, d

# ─────────────────────────────────────────────────────────────────
# HELPER: rolling stats (strictly before each match)
# ─────────────────────────────────────────────────────────────────
def rolling_team_stats(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each team in each match, compute rolling stats using only
    matches played BEFORE this match (no data leakage).
    Returns a DataFrame indexed by (match_id, team).
    """
    t1c = "team1" if "team1" in matches.columns else "batting_team"
    t2c = "team2" if "team2" in matches.columns else "bowling_team"
    records = []

    for _, row in matches.iterrows():
        mid  = row["match_id"]
        date = row["date"]

        for team_col, opp_col in [(t1c, t2c), (t2c, t1c)]:
            team = row[team_col]
            opp  = row[opp_col]
            if pd.isna(team): continue

            # All past matches for this team BEFORE current date
            past = matches[
                ((matches[t1c]==team) | (matches[t2c]==team)) &
                (matches["date"] < date)
            ].tail(window)

            if len(past) == 0:
                records.append({"match_id":mid, "team":team, "opponent":opp,
                                 "recent_wins":0, "recent_win_rate":0.5,
                                 "recent_matches":0, "form_score":0.5,
                                 "home_win_rate":0.5, "vs_opp_win_rate":0.5,
                                 "toss_win_rate":0.5, "avg_score_recent":150.0,
                                 "win_streak":0, "loss_streak":0})
                continue

            wins   = (past["winner"] == team).sum()
            n      = len(past)
            win_rt = wins / n

            # Form score: weighted recent wins (recent = higher weight)
            weights     = np.linspace(0.5, 1.0, n)
            win_flags   = (past["winner"].values == team).astype(float)
            form_score  = np.average(win_flags, weights=weights) if n > 0 else 0.5

            # Toss win rate
            if "toss_winner" in past.columns:
                toss_wr = (past["toss_winner"] == team).sum() / n
            else:
                toss_wr = 0.5

            # Head-to-head vs this opponent
            h2h = past[((past[t1c]==team)&(past[t2c]==opp)) |
                        ((past[t2c]==team)&(past[t1c]==opp))]
            vs_win = (h2h["winner"]==team).sum() / max(len(h2h),1)

            # Win/loss streak
            streak_wins = streak_losses = 0
            for res in reversed(win_flags.tolist()):
                if res == 1:
                    if streak_losses == 0: streak_wins += 1
                    else: break
                else:
                    if streak_wins == 0: streak_losses += 1
                    else: break

            records.append({
                "match_id":       mid,
                "team":           team,
                "opponent":       opp,
                "recent_wins":    int(wins),
                "recent_win_rate":round(win_rt, 4),
                "recent_matches": int(n),
                "form_score":     round(form_score, 4),
                "toss_win_rate":  round(toss_wr, 4),
                "vs_opp_win_rate":round(vs_win, 4),
                "win_streak":     streak_wins,
                "loss_streak":    streak_losses,
            })

    df = pd.DataFrame(records)
    log.info(f"Rolling team stats computed: {len(df):,} rows")
    return df


def venue_features(matches: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute venue-level stats using only past