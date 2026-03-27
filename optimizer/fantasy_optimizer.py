"""
CricketBrain AI — Dream11 Fantasy Team Optimizer
ILP (PuLP) + greedy fallback | 3 strategies: maximize, safe, differentiated
"""

import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    print("[Fantasy] PuLP not found — using greedy fallback")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES = os.path.join(DATA_DIR, "features")

# ─────────────────────────────────────────────────────────
# DREAM11 CONSTRAINTS
# ─────────────────────────────────────────────────────────
D11 = {
    "total_players":    11,
    "total_credits":    100.0,
    "max_per_team":     7,
    "min_wk":           1, "max_wk": 4,
    "min_bat":          3, "max_bat": 6,
    "min_ar":           1, "max_ar": 4,
    "min_bowl":         3, "max_bowl": 6,
}

# ─────────────────────────────────────────────────────────
# FANTASY POINT ESTIMATION (from deliveries data)
# ─────────────────────────────────────────────────────────
def estimate_player_fp(deliveries: pd.DataFrame, matches: pd.DataFrame,
                        player_list: list, recent_n: int = 5) -> pd.DataFrame:
    """Estimate expected fantasy points per player based on recent form"""
    dm = deliveries.copy()
    dm = dm.sort_values("date", na_position="last")
    dm["legal_ball"] = (dm.get("is_wide",0) + dm.get("is_noball",0) == 0).astype(int)

    records = []
    for player in player_list:
        # Batting FP
        bat = dm[dm["batter"] == player].groupby("match_id").agg(
            runs=("batsman_runs","sum"), balls=("legal_ball","sum"),
            fours=("batsman_runs", lambda x: (x==4).sum()),
            sixes=("batsman_runs", lambda x: (x==6).sum()),
            dismissed=("is_wicket","max"), date=("date","max")
        ).reset_index().sort_values("date", na_position="last").tail(recent_n)

        if len(bat) > 0:
            bat["sr"] = bat["runs"] / bat["balls"].replace(0,1) * 100
            bat["fp_bat"] = (
                bat["runs"]*1 + bat["fours"]*1 + bat["sixes"]*2 +
                (bat["runs"]>=30).astype(int)*8 + (bat["runs"]>=50).astype(int)*8 +
                (bat["runs"]>=100).astype(int)*16 + (bat["dismissed"]==0).astype(int)*4 +
                np.where(bat["sr"]>=170, 6, np.where(bat["sr"]>=150, 4, np.where(bat["sr"]>=130, 2, 0)))
            )
            avg_fp_bat = bat["fp_bat"].mean()
        else:
            avg_fp_bat = 0

        # Bowling FP
        bowl = dm[dm["bowler"] == player].groupby("match_id").agg(
            wkts=("is_wicket","sum"), runs=("total_runs","sum"),
            balls=("legal_ball","sum"), date=("date","max")
        ).reset_index().sort_values("date", na_position="last").tail(recent_n)

        if len(bowl) > 0:
            bowl["econ"] = bowl["runs"] / (bowl["balls"]/6).replace(0,1)
            bowl["fp_bowl"] = (
                bowl["wkts"]*25 + (bowl["wkts"]>=3).astype(int)*4 +
                (bowl["wkts"]>=4).astype(int)*4 + (bowl["wkts"]>=5).astype(int)*4 +
                np.where(bowl["econ"]<5, 6, np.where(bowl["econ"]<6, 4, np.where(bowl["econ"]<7, 2, 0)))
            )
            avg_fp_bowl = bowl["fp_bowl"].mean()
        else:
            avg_fp_bowl = 0

        total_fp = avg_fp_bat + avg_fp_bowl
        records.append({
            "player": player,
            "fp_batting": round(avg_fp_bat, 1),
            "fp_bowling": round(avg_fp_bowl, 1),
            "fp_total": round(total_fp, 1),
        })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────
# ILP OPTIMIZER
# ─────────────────────────────────────────────────────────
def optimize_team_ilp(players_df: pd.DataFrame, strategy: str = "maximize") -> dict:
    """
    players_df columns required:
      player, team, role (WK/BAT/AR/BOWL), credits, fp_total
    """
    if not HAS_PULP:
        return optimize_team_greedy(players_df, strategy)

    required = {"player","team","role","credits","fp_total"}
    if not required.issubset(players_df.columns):
        missing = required - set(players_df.columns)
        return {"error": f"Missing columns: {missing}"}

    df = players_df.copy().reset_index(drop=True)
    n  = len(df)

    # Adjust FP by strategy
    if strategy == "safe":
        df["adj_fp"] = df["fp_total"] * (1 - df["fp_total"].rank(pct=True) * 0.2)
    elif strategy == "differentiated":
        median_fp = df["fp_total"].median()
        df["adj_fp"] = df["fp_total"] * np.where(df["fp_total"] < median_fp, 1.3, 0.9)
    else:
        df["adj_fp"] = df["fp_total"]

    prob = pulp.LpProblem("Dream11", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective
    prob += pulp.lpSum(df["adj_fp"].iloc[i] * x[i] for i in range(n))

    # Total players
    prob += pulp.lpSum(x) == D11["total_players"]

    # Budget
    prob += pulp.lpSum(df["credits"].iloc[i] * x[i] for i in range(n)) <= D11["total_credits"]

    # Role constraints
    for role, lo, hi in [("WK", D11["min_wk"], D11["max_wk"]),
                          ("BAT", D11["min_bat"], D11["max_bat"]),
                          ("AR", D11["min_ar"], D11["max_ar"]),
                          ("BOWL", D11["min_bowl"], D11["max_bowl"])]:
        idx = [i for i, r in enumerate(df["role"]) if r == role]
        if idx:
            prob += pulp.lpSum(x[i] for i in idx) >= lo
            prob += pulp.lpSum(x[i] for i in idx) <= hi

    # Max per team
    for team in df["team"].unique():
        idx = [i for i, t in enumerate(df["team"]) if t == team]
        if idx:
            prob += pulp.lpSum(x[i] for i in idx) <= D11["max_per_team"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return optimize_team_greedy(players_df, strategy)

    selected = df[[pulp.value(x[i]) == 1 for i in range(n)]].copy()
    total_fp = selected["fp_total"].sum()
    total_cr = selected["credits"].sum()

    # Captain = highest FP, Vice-captain = second highest
    selected_sorted = selected.sort_values("fp_total", ascending=False)
    captain      = selected_sorted.iloc[0]["player"]
    vice_captain = selected_sorted.iloc[1]["player"]

    return {
        "strategy": strategy,
        "solver":   "ILP (PuLP)",
        "captain":       captain,
        "vice_captain":  vice_captain,
        "total_fp":      round(total_fp, 1),
        "total_credits": round(total_cr, 1),
        "team": selected[["player","team","role","credits","fp_total"]].to_dict("records"),
    }

# ─────────────────────────────────────────────────────────
# GREEDY FALLBACK
# ─────────────────────────────────────────────────────────
def optimize_team_greedy(players_df: pd.DataFrame, strategy: str = "maximize") -> dict:
    df = players_df.copy().sort_values("fp_total", ascending=False)
    selected = []
    used_credits = 0
    role_counts = {"WK": 0, "BAT": 0, "AR": 0, "BOWL": 0}
    team_counts = {}

    # Pass 1: respect all constraints
    for _, row in df.iterrows():
        if len(selected) >= 11:
            break
        role = str(row.get("role", "BAT"))
        team = str(row.get("team", "UNK"))
        cred = float(row.get("credits", 8.5))
        if used_credits + cred > D11["total_credits"] + 5:  # slight budget flex
            continue
        max_role = D11.get(f"max_{role.lower()}", 6)
        if role_counts.get(role, 0) >= max_role:
            continue
        if team_counts.get(team, 0) >= D11["max_per_team"]:
            continue
        selected.append(row)
        used_credits += cred
        role_counts[role] = role_counts.get(role, 0) + 1
        team_counts[team] = team_counts.get(team, 0) + 1

    # Pass 2: if still < 11, relax team constraint and fill remaining
    if len(selected) < 11:
        selected_players = {r["player"] for r in selected}
        for _, row in df.iterrows():
            if len(selected) >= 11:
                break
            if row["player"] in selected_players:
                continue
            selected.append(row)
            selected_players.add(row["player"])

    if len(selected) < 11:
        return {"error": f"Only found {len(selected)} valid players. Need more players in pool.", "solver": "greedy"}

    sel_df = pd.DataFrame(selected)
    sel_df_sorted = sel_df.sort_values("fp_total", ascending=False)
    captain      = sel_df_sorted.iloc[0]["player"]
    vice_captain = sel_df_sorted.iloc[1]["player"]

    return {
        "strategy":      strategy,
        "solver":        "Greedy",
        "captain":       captain,
        "vice_captain":  vice_captain,
        "total_fp":      round(sel_df["fp_total"].sum(), 1),
        "total_credits": round(used_credits, 1),
        "team": sel_df[["player","team","role","credits","fp_total"]].to_dict("records"),
    }

# ─────────────────────────────────────────────────────────
# MAIN FANTASY API
# ─────────────────────────────────────────────────────────
def generate_fantasy_teams(players: list, strategy: str = "maximize") -> dict:
    """
    players: list of dicts with keys: player, team, role, credits, fp_total
    Returns three strategy variants: maximize, safe, differentiated
    """
    df = pd.DataFrame(players)
    if "fp_total" not in df.columns:
        df["fp_total"] = np.random.uniform(20, 80, len(df))  # fallback

    results = {}
    for strat in ["maximize", "safe", "differentiated"]:
        results[strat] = optimize_team_ilp(df, strategy=strat)

    return results

if __name__ == "__main__":
    # Demo
    sample_players = [
        {"player":"MS Dhoni","team":"CSK","role":"WK","credits":9.0,"fp_total":55.0},
        {"player":"Virat Kohli","team":"RCB","role":"BAT","credits":11.0,"fp_total":68.0},
        {"player":"Rohit Sharma","team":"MI","role":"BAT","credits":10.5,"fp_total":62.0},
        {"player":"Hardik Pandya","team":"MI","role":"AR","credits":10.0,"fp_total":58.0},
        {"player":"Ravindra Jadeja","team":"CSK","role":"AR","credits":9.0,"fp_total":52.0},
        {"player":"Jasprit Bumrah","team":"MI","role":"BOWL","credits":10.5,"fp_total":60.0},
        {"player":"Yuzvendra Chahal","team":"RR","role":"BOWL","credits":9.0,"fp_total":48.0},
        {"player":"KL Rahul","team":"LSG","role":"WK","credits":10.5,"fp_total":61.0},
        {"player":"Shubman Gill","team":"GT","role":"BAT","credits":10.0,"fp_total":57.0},
        {"player":"Rashid Khan","team":"GT","role":"BOWL","credits":9.5,"fp_total":56.0},
        {"player":"Trent Boult","team":"RR","role":"BOWL","credits":8.5,"fp_total":45.0},
        {"player":"Devon Conway","team":"CSK","role":"BAT","credits":8.5,"fp_total":43.0},
        {"player":"Ruturaj Gaikwad","team":"CSK","role":"BAT","credits":9.0,"fp_total":50.0},
        {"player":"Axar Patel","team":"DC","role":"AR","credits":8.5,"fp_total":46.0},
        {"player":"Arshdeep Singh","team":"PBKS","role":"BOWL","credits":8.5,"fp_total":44.0},
    ]
    results = generate_fantasy_teams(sample_players, "maximize")
    for strat, res in results.items():
        print(f"\n{strat.upper()} Team:")
        if "team" in res:
            for p in res["team"]:
                print(f"  {p['player']:25} {p['role']:4} {p['credits']} cr  {p['fp_total']:.0f} fp")
            print(f"  Captain: {res['captain']} | VC: {res['vice_captain']}")
            print(f"  Total FP: {res['total_fp']} | Credits: {res['total_credits']}")