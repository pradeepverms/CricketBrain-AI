"""
CricketBrain AI — FastAPI Backend
15 production-grade endpoints | Async | Caching | Pydantic v2
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os, json, joblib, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# ── Local imports ──
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation.monte_carlo import run_simulation
from optimizer.fantasy_optimizer import generate_fantasy_teams
from ml.weakness_detector import batsman_weakness, bowler_weakness, matchup_matrix
from etl.insight_generator import (
    team_form_insights, toss_advisor, bowling_strategy,
    player_form_insights, match_preview_insights, viral_insight
)

# ─────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
SHAP_DIR  = os.path.join(DATA_DIR, "shap")
CLEANED   = os.path.join(DATA_DIR, "cleaned")

app = FastAPI(
    title="CricketBrain AI API",
    description="Production-grade IPL Analytics & Prediction Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# STARTUP — load models & data
# ─────────────────────────────────────────────────────────
_cache = {}

def get_model():
    if "model" not in _cache:
        p = os.path.join(MODEL_DIR, "best_model.pkl")
        _cache["model"] = joblib.load(p) if os.path.exists(p) else None
    return _cache["model"]

def get_features():
    if "features" not in _cache:
        p = os.path.join(MODEL_DIR, "feature_names.pkl")
        _cache["features"] = joblib.load(p) if os.path.exists(p) else []
    return _cache["features"]

def get_metrics():
    if "metrics" not in _cache:
        p = os.path.join(MODEL_DIR, "metrics.json")
        _cache["metrics"] = json.load(open(p)) if os.path.exists(p) else {}
    return _cache["metrics"]

def get_matches():
    if "matches" not in _cache:
        p = os.path.join(CLEANED, "matches.csv")
        _cache["matches"] = pd.read_csv(p, parse_dates=["date"]) if os.path.exists(p) else pd.DataFrame()
    return _cache["matches"]

def get_deliveries():
    if "deliveries" not in _cache:
        p = os.path.join(CLEANED, "deliveries.csv")
        _cache["deliveries"] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        if "legal_ball" not in _cache["deliveries"].columns:
            d = _cache["deliveries"]
            d["legal_ball"] = (d.get("is_wide",0) + d.get("is_noball",0) == 0).astype(int)
    return _cache["deliveries"]

def get_match_features():
    if "match_feats" not in _cache:
        p = os.path.join(DATA_DIR, "features", "match_features.csv")
        _cache["match_feats"] = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return _cache["match_feats"]

# ─────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str = "bat"
    season: int = 2024

class SimulateRequest(BaseModel):
    team1: str
    team2: str
    base_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    n_simulations: int = Field(default=10000, ge=100, le=50000)

class FantasyRequest(BaseModel):
    players: List[dict]
    strategy: str = "maximize"

class MatchupRequest(BaseModel):
    batsman: str
    bowler: str

# ─────────────────────────────────────────────────────────
# INTERNAL: build feature vector
# ─────────────────────────────────────────────────────────
def _build_feature_vector(req: PredictRequest) -> np.ndarray:
    mf = get_match_features()
    feature_names = get_features()
    if mf.empty or not feature_names:
        return None

    # Find last known stats for both teams
    t1_rows = mf[mf["team1"] == req.team1].sort_values("date", key=pd.to_datetime, errors="coerce")
    t2_rows = mf[mf["team2"] == req.team2].sort_values("date", key=pd.to_datetime, errors="coerce")

    if t1_rows.empty or t2_rows.empty:
        # Try reversed
        t1_rows = mf[mf["team2"] == req.team1]
        t2_rows = mf[mf["team1"] == req.team2]

    base = mf[
        ((mf["team1"] == req.team1) & (mf["team2"] == req.team2)) |
        ((mf["team1"] == req.team2) & (mf["team2"] == req.team1))
    ].sort_values("date", key=pd.to_datetime, errors="coerce")

    if not base.empty:
        row = base.iloc[-1]
    elif not t1_rows.empty:
        row = t1_rows.iloc[-1]
    else:
        row = mf.iloc[-1]

    vec = []
    for f in feature_names:
        val = row.get(f, 0)
        vec.append(float(val) if pd.notna(val) else 0.0)

    # Override toss features
    feat_map = {n: i for i, n in enumerate(feature_names)}
    if "toss_decision_bat" in feat_map:
        vec[feat_map["toss_decision_bat"]] = 1.0 if req.toss_decision == "bat" else 0.0
    if "toss_won_by_team1" in feat_map:
        vec[feat_map["toss_won_by_team1"]] = 1.0 if req.toss_winner == req.team1 else 0.0

    return np.array([vec])

def _shap_explanation(model, X_vec, feature_names, team1, team2):
    try:
        import shap
        explainer_path = os.path.join(SHAP_DIR, "explainer_xgboost.pkl")
        if not os.path.exists(explainer_path):
            explainer_path = os.path.join(SHAP_DIR, "explainer_lightgbm.pkl")
        if not os.path.exists(explainer_path):
            return None, "No SHAP explainer available"

        explainer = joblib.load(explainer_path)
        base_model = model.estimator if hasattr(model, "estimator") else model
        sv = explainer.shap_values(X_vec)
        if isinstance(sv, list):
            sv = sv[1]
        sv = sv[0]

        top_n = 5
        feat_imp = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        parts = []
        for feat, val in feat_imp:
            direction = "↑" if val > 0 else "↓"
            label = feat.replace("_"," ").title()
            parts.append(f"{label} {direction} ({val:+.3f})")

        winner = team1 if sum(sv) > 0 else team2
        nl = f"{winner} is favoured due to: " + " | ".join(parts[:3])

        return [{"feature": f, "shap_value": round(v, 4)} for f, v in feat_imp], nl
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"service": "CricketBrain AI API", "version": "2.0", "status": "running"}

@app.get("/health", tags=["health"])
def health():
    model_loaded    = get_model() is not None
    data_loaded     = not get_matches().empty
    metrics         = get_metrics()
    return {
        "model_loaded": model_loaded,
        "data_loaded":  data_loaded,
        "best_auc":     metrics.get("calibrated_auc", metrics.get("roc_auc", "N/A")),
        "features":     len(get_features()),
    }

@app.post("/predict", tags=["prediction"])
def predict_match(req: PredictRequest):
    model = get_model()
    feature_names = get_features()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run ml/train.py first.")

    X = _build_feature_vector(req)
    if X is None:
        raise HTTPException(status_code=400, detail="Could not build feature vector. Run feature_engine.py first.")

    proba = float(model.predict_proba(X)[0][1])
    pred  = int(proba > 0.5)
    winner = req.team1 if pred == 1 else req.team2

    # SHAP
    shap_vals, nl_explanation = _shap_explanation(model, X, feature_names, req.team1, req.team2)

    # Simulation
    sim = run_simulation(req.team1, req.team2, proba, n_sim=5000)

    return {
        "team1": req.team1,
        "team2": req.team2,
        "venue": req.venue,
        "team1_win_probability": round(proba, 4),
        "team2_win_probability": round(1-proba, 4),
        "predicted_winner": winner,
        "confidence": round(abs(proba - 0.5) * 200, 1),
        "explanation": nl_explanation,
        "shap_features": shap_vals,
        "simulation": {
            "team1_win_pct": sim["team1_win_pct"],
            "team2_win_pct": sim["team2_win_pct"],
            "ci_95": [sim["ci_95_low"], sim["ci_95_high"]],
        }
    }

@app.post("/simulate", tags=["simulation"])
def simulate(req: SimulateRequest):
    result = run_simulation(req.team1, req.team2, req.base_prob, req.n_simulations)
    return result

@app.post("/fantasy/optimize", tags=["fantasy"])
def fantasy_optimize(req: FantasyRequest):
    if not req.players or len(req.players) < 11:
        raise HTTPException(status_code=400, detail="Need at least 11 players")
    results = generate_fantasy_teams(req.players, req.strategy)
    return results

@app.get("/players/search", tags=["players"])
def player_search(q: str = Query(..., min_length=2)):
    d = get_deliveries()
    if d.empty:
        return {"results": []}
    batters  = d[d["batter"].str.contains(q, case=False, na=False)]["batter"].unique().tolist()
    bowlers  = d[d["bowler"].str.contains(q, case=False, na=False)]["bowler"].unique().tolist()
    combined = list(set(batters + bowlers))[:20]
    return {"query": q, "results": sorted(combined)}

@app.get("/players/{player_name}/stats", tags=["players"])
def player_stats(player_name: str):
    d = get_deliveries()
    m = get_matches()
    if d.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")

    d["legal_ball"] = (d.get("is_wide",0) + d.get("is_noball",0) == 0).astype(int)
    bat = d[d["batter"] == player_name]
    bowl = d[d["bowler"] == player_name]

    batting = {}
    if len(bat) > 0:
        legal = bat[bat["legal_ball"]==1]
        total_runs = int(bat["batsman_runs"].sum())
        total_balls = int(len(legal))
        wkts_lost = int(bat["is_wicket"].sum())
        innings_played = bat["match_id"].nunique()
        batting = {
            "innings": innings_played,
            "runs": total_runs,
            "balls": total_balls,
            "sr": round(total_runs/max(total_balls,1)*100, 2),
            "avg": round(total_runs/max(wkts_lost,1), 2),
            "fours": int((bat["batsman_runs"]==4).sum()),
            "sixes": int((bat["batsman_runs"]==6).sum()),
        }

    bowling = {}
    if len(bowl) > 0:
        legal = bowl[bowl["legal_ball"]==1]
        total_runs = int(bowl["total_runs"].sum())
        total_balls = int(len(legal))
        wkts = int(bowl["is_wicket"].sum())
        overs = total_balls/6
        bowling = {
            "innings": bowl["match_id"].nunique(),
            "wickets": wkts,
            "runs": total_runs,
            "overs": round(overs, 1),
            "economy": round(total_runs/max(overs,0.1), 2),
            "avg": round(total_runs/max(wkts,1), 2),
            "sr":  round(total_balls/max(wkts,1), 2),
        }

    return {
        "player": player_name,
        "batting": batting,
        "bowling": bowling,
    }

@app.get("/players/{player_name}/weakness", tags=["players"])
def player_weakness_endpoint(player_name: str, role: str = Query("bat", enum=["bat","bowl"])):
    d = get_deliveries()
    m = get_matches()
    if d.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    if role == "bat":
        return batsman_weakness(player_name, d, m)
    else:
        return bowler_weakness(player_name, d, m)

@app.get("/players/{player_name}/insights", tags=["players"])
def player_insights_endpoint(player_name: str, role: str = Query("bat", enum=["bat","bowl"])):
    d = get_deliveries()
    m = get_matches()
    if d.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    insights = player_form_insights(player_name, d, m, role)
    return {"player": player_name, "role": role, "insights": insights}

@app.get("/players/top/batsmen", tags=["players"])
def top_batsmen(season: Optional[int] = None, top_n: int = 10):
    d = get_deliveries()
    m = get_matches()
    if d.empty:
        return {"batsmen": []}
    dm = d.merge(m[["match_id","season"]], on="match_id", how="left")
    if season:
        dm = dm[dm["season"] == season]
    dm["legal_ball"] = (dm.get("is_wide",0) + dm.get("is_noball",0) == 0).astype(int)
    agg = dm.groupby("batter").agg(
        runs=("batsman_runs","sum"),
        balls=("legal_ball","sum"),
        wkts=("is_wicket","sum"),
        innings=("match_id","nunique"),
    ).reset_index()
    agg["sr"] = (agg["runs"]/agg["balls"].replace(0,1)*100).round(2)
    agg["avg"] = (agg["runs"]/agg["wkts"].replace(0,1)).round(2)
    agg = agg[agg["innings"] >= 5].sort_values("runs", ascending=False).head(top_n)
    return {"batsmen": agg.to_dict("records")}

@app.get("/players/top/bowlers", tags=["players"])
def top_bowlers(season: Optional[int] = None, top_n: int = 10):
    d = get_deliveries()
    m = get_matches()
    if d.empty:
        return {"bowlers": []}
    dm = d.merge(m[["match_id","season"]], on="match_id", how="left")
    if season:
        dm = dm[dm["season"] == season]
    dm["legal_ball"] = (dm.get("is_wide",0) + dm.get("is_noball",0) == 0).astype(int)
    agg = dm.groupby("bowler").agg(
        wkts=("is_wicket","sum"),
        runs=("total_runs","sum"),
        balls=("legal_ball","sum"),
        innings=("match_id","nunique"),
    ).reset_index()
    agg["econ"] = (agg["runs"]/(agg["balls"]/6).replace(0,1)).round(2)
    agg["avg"]  = (agg["runs"]/agg["wkts"].replace(0,1)).round(2)
    agg = agg[agg["innings"] >= 5].sort_values("wkts", ascending=False).head(top_n)
    return {"bowlers": agg.to_dict("records")}

@app.post("/matchup", tags=["players"])
def player_matchup(req: MatchupRequest):
    d = get_deliveries()
    if d.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return matchup_matrix(req.batsman, req.bowler, d)

@app.get("/teams", tags=["teams"])
def list_teams():
    m = get_matches()
    if m.empty:
        return {"teams": []}
    teams = sorted(set(m["team1"].dropna().tolist() + m["team2"].dropna().tolist()))
    return {"teams": teams}

@app.get("/teams/{team_name}", tags=["teams"])
def team_stats(team_name: str):
    m = get_matches()
    if m.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    tm = m[(m["team1"]==team_name) | (m["team2"]==team_name)]
    if tm.empty:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")
    wins = (tm["winner"]==team_name).sum()
    total = len(tm)
    return {
        "team": team_name,
        "matches": int(total),
        "wins": int(wins),
        "win_pct": round(wins/max(total,1)*100, 2),
        "seasons": sorted(tm["season"].dropna().unique().astype(int).tolist()),
    }

@app.get("/teams/insights/{team_name}", tags=["teams"])
def team_insights_endpoint(team_name: str):
    m = get_matches()
    d = get_deliveries()
    if m.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    form      = team_form_insights(team_name, m)
    strategy  = bowling_strategy(team_name, d, m)
    return {
        "team": team_name,
        "form_insights": form,
        "bowling_strategy_against": strategy["strategy_insights"],
    }

@app.get("/match/preview", tags=["prediction"])
def match_preview(team1: str, team2: str, venue: str = "Wankhede Stadium"):
    m = get_matches()
    d = get_deliveries()
    if m.empty:
        raise HTTPException(status_code=503, detail="Data not loaded")
    preview = match_preview_insights(team1, team2, venue, m, d)
    toss    = toss_advisor(team1, team2, venue, m)
    return {**preview, "toss_recommendation": toss}

@app.get("/venues", tags=["venues"])
def list_venues():
    m = get_matches()
    if m.empty:
        return {"venues": []}
    venues = sorted(m["venue"].dropna().unique().tolist())
    return {"venues": venues}

@app.get("/model/info", tags=["model"])
def model_info():
    metrics = get_metrics()
    features = get_features()
    return {
        "model_type": metrics.get("best_model", "Unknown"),
        "features": len(features),
        "feature_names": features,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
    }

@app.get("/shap/{model_name}", tags=["model"])
def shap_importance(model_name: str = "xgboost"):
    path = os.path.join(SHAP_DIR, f"shap_{model_name}.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"SHAP data not found for {model_name}. Train model first.")
    df = pd.read_csv(path)
    return {"model": model_name, "features": df.to_dict("records")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
