# 🏏 CricketBrain AI

> **Production-grade IPL Cricket Analytics & Decision Intelligence Platform**

A complete end-to-end ML engineering system built from scratch — featuring ensemble ML models with Optuna tuning, Monte Carlo match simulation, SHAP explainability, ILP fantasy optimization, and an 18-page Streamlit analytics dashboard backed by a FastAPI microservice.

---

## 🎯 System Capabilities

| Feature | Details |
|---|---|
| **Match Prediction** | XGBoost + LightGBM + CatBoost Stacking Ensemble, Optuna-tuned |
| **Explainability** | SHAP TreeExplainer — feature-level attribution for every prediction |
| **Simulation** | Monte Carlo (10,000+ iterations) — win %, score distributions, 95% CI |
| **Fantasy Optimizer** | ILP via PuLP — Dream11 constraints with 3 optimization strategies |
| **Weakness Engine** | Phase-level batting/bowling weaknesses, spin/pace matchups |
| **Insight Engine** | Rule-based + statistical hybrid — auto-generates NL cricket insights |
| **Decision Engine** | Toss advisor, bowling strategy, batting order recommendations |
| **API** | FastAPI — 15 production endpoints, async, Pydantic v2, cached |
| **Dashboard** | Streamlit — 18 pages, Plotly charts, dark theme |

---

## 🏗️ Architecture

```
CricketBrain AI/
├── etl/
│   ├── data_cleaning.py       # Ball-by-ball IPL pipeline (2008-2025)
│   ├── feature_engine.py      # 70+ time-aware features, zero leakage
│   └── insight_generator.py   # NL insights + toss advisor + strategy
├── ml/
│   ├── train.py               # XGB+LGB+CB+Stack | Optuna | SHAP | Calibration
│   └── weakness_detector.py   # Phase/bowler type/matchup weakness analysis
├── simulation/
│   └── monte_carlo.py         # 10,000 simulations | CI | Score distributions
├── optimizer/
│   └── fantasy_optimizer.py   # ILP Dream11 | 3 strategies | Greedy fallback
├── api/
│   └── main.py                # FastAPI — 15 production endpoints
├── app/
│   └── app.py                 # Streamlit — 18-page analytics dashboard
├── data/
│   ├── raw/IPL.csv            # ← Place your Kaggle download here
│   ├── cleaned/               # Matches + Deliveries (CSV + Parquet)
│   ├── features/              # Engineered feature tables (Parquet)
│   ├── models/                # Trained model PKLs + metrics.json
│   └── shap/                  # SHAP explainer PKLs + importance CSVs
├── init_db.py                 # SQLite schema builder
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Mac M-series: `brew install libomp` (for XGBoost)

### 1. Get the data
Download IPL Dataset (2008–2025) from Kaggle:
```
https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025
```
Place the downloaded file at: `data/raw/IPL.csv`

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ETL pipeline
```bash
python etl/data_cleaning.py
python etl/feature_engine.py
python init_db.py
```

### 4. Train ML models
```bash
python ml/train.py
# ~5-15 minutes depending on hardware
# Trains: RandomForest → XGBoost (Optuna 50 trials) → LightGBM (40 trials)
# → CatBoost → Stacking Ensemble
# Outputs: data/models/best_model.pkl + metrics.json + SHAP CSVs
```

### 5. Launch the dashboard
```bash
streamlit run app/app.py
# Opens at http://localhost:8501
```

### 6. Launch the API (optional)
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs at http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

```bash
# Build + start API and Dashboard
docker-compose up --build

# API  → http://localhost:8000/docs
# App  → http://localhost:8501
```

---

## 🤖 ML Pipeline

### Feature Engineering (Zero Data Leakage)
- **70+ features** engineered from ball-by-ball data
- Rolling stats: `win_rate_last3/5/10/15`, `form_score`, `win_streak`
- Time-aware: all features use `shift(1).rolling()` to prevent leakage
- Phase-level: powerplay / middle / death overs for batting & bowling
- H2H features computed with historical mask (only past matches used)
- Differential features: `diff_win_rate_last3` = team1 - team2 performance gap

### Model Architecture
```
RandomForest (baseline)
    ↓
XGBoost (Optuna, 50 trials, TimeSeriesSplit-5)
    ↓
LightGBM (Optuna, 40 trials, TimeSeriesSplit-5)
    ↓
CatBoost (500 iterations, depth=6)
    ↓
StackingClassifier (meta-learner: Logistic Regression)
    ↓
CalibratedClassifierCV (isotonic regression)
```

### Evaluation Strategy
- **TimeSeriesSplit CV** (no random shuffling — preserves temporal order)
- Test set: last 2 seasons held out
- Metrics: ROC-AUC, Log Loss, Accuracy, Brier Score, Average Precision

---

## 🎲 Monte Carlo Engine

```python
from simulation.monte_carlo import run_simulation

result = run_simulation("CSK", "MI", base_prob=0.62, n_sim=10000)
# Returns: win%, CI_95, scenario breakdown, score distributions
```

Output:
```
🎲 CricketBrain AI simulated this match 10,000 times.
📊 CSK wins in 62.4% of simulations (95% CI: 52%–72%)
🎯 CSK: 167 runs (range: 139–195)
🎯 MI:  161 runs (range: 135–188)
```

---

## 💡 SHAP Explainability

Every prediction comes with SHAP-attributed explanations:

```
"CSK is favoured due to:
  - team1_form_score ↑ (+0.142)
  - diff_win_rate_last5 ↑ (+0.089)
  - venue_bat_first_win_rate ↑ (+0.063)"
```

---

## 💰 Fantasy Optimizer

```python
from optimizer.fantasy_optimizer import generate_fantasy_teams

result = generate_fantasy_teams(players, strategy="maximize")
# Strategies: maximize | safe | differentiated
# Solver: PuLP ILP (CBC) → greedy fallback
# Constraints: 11 players, 100 credits, role balance, max 7/team
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Match prediction + SHAP + simulation |
| `/simulate` | POST | Monte Carlo simulation (N runs) |
| `/fantasy/optimize` | POST | Dream11 ILP optimizer |
| `/players/{name}/stats` | GET | Full career batting + bowling stats |
| `/players/{name}/weakness` | GET | Phase/matchup weakness analysis |
| `/players/{name}/insights` | GET | AI form insights |
| `/matchup` | POST | Batsman vs bowler head-to-head |
| `/teams/{name}` | GET | Team overview |
| `/teams/insights/{name}` | GET | Team AI insights + strategy |
| `/match/preview` | GET | Full pre-match briefing |
| `/shap/{model}` | GET | SHAP feature importance |
| `/model/info` | GET | Model metrics + feature list |
| `/health` | GET | System health check |

---

## 📊 Dashboard Pages

| # | Page | Key Features |
|---|---|---|
| 1 | 🏆 League Overview | Season trends, team wins, toss analysis |
| 2 | 🔍 Player Profile | Career stats, phase analysis, AI insights |
| 3 | ⚔️ Player Comparison | Side-by-side stats, radar chart |
| 4 | 🥊 Batsman vs Bowler | Matchup matrix, phase breakdown, dismissals |
| 5 | 🤝 Partnership Tracker | Best partnerships, RPO analysis |
| 6 | ⚔️ Team H2H | H2H record, cumulative wins, form |
| 7 | 🤖 AI Predictor | Ensemble prediction, SHAP, Monte Carlo |
| 8 | 🎯 Toss Advisor | Venue stats, bowling strategy |
| 9 | 📈 Win Probability | Ball-by-ball cumulative runs |
| 10 | 📋 Points Table | Season standings, color-coded |
| 11 | 🌡️ Run Heatmap | Run rate + wickets by over |
| 12 | 📊 Form Tracker | Color-coded form chart, rolling avg |
| 13 | ⚡ Breakout Players | Season-over-season growth |
| 14 | 💰 Fantasy Optimizer | ILP Dream11 team builder |
| 15 | 🏅 Rankings | Batting/bowling leaderboards, bubble chart |
| 16 | 🏟️ Venue Analysis | Pitch profiling, toss stats |
| 17 | 🧬 Player Similarity | Cosine similarity engine |
| 18 | 📖 About | Model metrics, system architecture |

---

## 🧰 Tech Stack

- **Data:** Pandas, NumPy, PyArrow (Parquet)
- **ML:** XGBoost, LightGBM, CatBoost, scikit-learn
- **Tuning:** Optuna (Bayesian hyperparameter optimization)
- **Explainability:** SHAP
- **Optimization:** PuLP (ILP)
- **API:** FastAPI, Uvicorn, Pydantic v2
- **Dashboard:** Streamlit, Plotly
- **Database:** SQLite (extensible to PostgreSQL)
- **DevOps:** Docker, docker-compose

---

## 📄 License

MIT License — Free for personal and educational use.

---

*Built with ❤️ for cricket analytics enthusiasts and ML engineers*
