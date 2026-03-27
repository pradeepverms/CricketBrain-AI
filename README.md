# 🏏 CricketBrain AI

**Decision Intelligence System for IPL Cricket**

> Not just analytics. An AI system that tells you **what to do**, **why it matters**, and **how confident it is** — on every screen.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What is CricketBrain AI?

Most cricket dashboards show data. CricketBrain AI makes **decisions**.

| Other platforms | CricketBrain AI |
|---|---|
| Shows win probability | Explains WHY and what to do next |
| Displays stats | Generates actionable decisions |
| Static charts | Live pressure + momentum tracking |
| Generic rankings | Clutch / Stable / High-Risk classification |
| One fantasy team | 3 teams — Safe / Grand League / Differential |

---

## System Architecture

```
📁 IPL.csv (2008–2025, 278K+ deliveries)
        │
        ▼
┌─────────────────────────────────────┐
│  ETL Pipeline                       │
│  • Auto-detects any column format   │
│  • Team/venue normalization         │
│  • Phase: Powerplay / Middle / Death│
│  • Parquet + CSV output             │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Feature Engineering (70+ features) │
│  • Zero data leakage (shift+rolling)│
│  • Rolling win rates (last 3/5/10)  │
│  • H2H stats (time-aware)           │
│  • Pressure Index + Momentum Index  │
│  • EMA batting form (α=0.3)         │
│  • xRuns, xWickets                  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  ML Pipeline                        │
│  • XGBoost  (Optuna, 50 trials)     │
│  • LightGBM (Optuna, 40 trials)     │
│  • CatBoost (500 iterations)        │
│  • Stacking Ensemble (LR meta)      │
│  • Isotonic calibration             │
│  • SHAP TreeExplainer               │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Intelligence Engines               │
│  • Decision Engine (5K simulations) │
│  • Monte Carlo  (10K simulations)   │
│  • Fantasy ILP Optimizer (PuLP)     │
│  • Weakness Detector                │
│  • Insight Generator (NL)           │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  24-Page Streamlit Dashboard        │
│  + FastAPI Backend (15 endpoints)   │
└─────────────────────────────────────┘
```

---

## Features

### 🤖 Machine Learning
- **Stacking Ensemble** — XGBoost + LightGBM + CatBoost meta-learned by Logistic Regression
- **Optuna Tuning** — 50–90 trial Bayesian hyperparameter search per model
- **SHAP Explainability** — every prediction shows top feature contributions
- **Calibrated Probabilities** — isotonic regression, probabilities are reliable
- **Zero Data Leakage** — all rolling features use `shift(1)` — model never sees the future
- **TimeSeriesSplit CV** — 5-fold chronological validation

### 🧠 Decision Intelligence Engine
Every AI decision includes all five components:
1. **Data Evidence** — exact RR deficit, pressure index shown numerically
2. **Simulation Support** — 5,000 runs comparing Aggressive vs Safe strategies with win %
3. **Confidence Level** — HIGH / MEDIUM / LOW with reason
4. **Multiple Options** — Aggressive and Conservative with win % each
5. **Consequence** — "If not followed → win probability drops by X%"
6. **Sensitivity Analysis** — what if required RR rises? What if a wicket falls?

### 📈 Win Probability + Momentum
- Ball-by-ball win probability with per-over delta chart
- **Turning Point Detection** — highlights overs where WP shifted ≥8%
- **Pressure Index** — 0–100 based on required rate, wickets, overs left
- **Momentum Index** — last 12 balls, exponential decay weighted (−100 to +100)
- Required RR vs Current RR gap chart with colour zones

### 🎲 Monte Carlo Simulation
- 10,000 match simulations per prediction
- Win %, 95% confidence interval, score distributions (P10–P90)
- Dominant / contested / upset scenario breakdown
- Auto-generated natural language match summary

### 💰 Elite Fantasy Optimizer
- **3 teams generated simultaneously** — Safe, Grand League, Differential
- **Ceiling / Floor / Variance** per player shown in pool table
- **Ownership Prediction** — estimated % of teams picking each player
- **WHY THIS PLAYER** reasoning for every pick
- **Pitch Scenario Mode** — batting/bowling pitch adjusts fantasy weights
- ILP (PuLP CBC) with greedy fallback
- Alternative swap suggestions per team

### 👤 Deep Player Analytics
- **4-way Classification** — 🔥 Clutch / ⚡ High-Risk / 🛡️ Stable / 🎯 Flat-track Bully
- **EMA Form Line** — exponentially weighted recent form plotted on chart
- **Consistency Score** (0–100) + **Volatility Score** (0–10) as gauges
- **Clutch Score** — pressure performance vs normal performance ratio
- **xRuns / xWickets** — expected stats beyond simple averages
- Phase-level breakdown with above/below career baseline

### 🏟️ Advanced Venue Intelligence
- **Pitch DNA Classification** — Batting Paradise / Balanced / Bowler-Friendly / Seam Track
- **Spin vs Pace Analysis** — economy rate and wicket rate head-to-head
- **Phase Dominance** — which phase defines this ground
- **AI Best XI** — venue-specific player selection from historical data
- **Strategy Advisor** — data-backed bat first vs chase recommendation

### 📋 Season Intelligence
- **Playoff Probability** — Monte Carlo qualification simulation per team
- **Form Trend** — last 5 matches as 🟢🔴 visual
- **Expected Final Position** column
- Full colour-coded points table

### 🔬 Backtesting & Calibration
- Season-by-season accuracy chart
- Calibration curve (Reliability Diagram)
- All model metrics vs random baseline
- Full temporal validation methodology

---

## Dashboard Pages

| # | Page | Key Intelligence |
|---|---|---|
| 1 | 🏆 League Overview | Season trends, team dominance |
| 2 | 🔍 Player Profile | Stats, phase breakdown, AI weaknesses |
| 3 | ⚔️ Player Comparison | Side-by-side, radar chart |
| 4 | 🥊 Batsman vs Bowler | Matchup matrix, dismissal types |
| 5 | 🤝 Partnership Tracker | Best pairs, RPO analysis |
| 6 | ⚔️ Team Comparison | H2H record, cumulative wins |
| 7 | 🤖 AI Match Predictor | Ensemble + SHAP + Monte Carlo |
| 8 | 🎯 Toss Advisor | Venue data, bowling strategy |
| 9 | 📈 Win Probability | Ball-by-ball WP + turning points |
| 10 | 📋 Season Points Table | Standings + playoff probability |
| 11 | 🌡️ Run Heatmap | Phase analysis, anomaly detection |
| 12 | 📊 Form Tracker | HOT/DECLINING + risk score |
| 13 | ⚡ Breakout Players | Why breakout + sustainability |
| 14 | 💰 Fantasy Optimizer | ILP team + ownership strategy |
| 15 | 🏅 Player Rankings | Impact score + style clusters |
| 16 | 🏟️ Venue Analysis | Pitch type + player suitability |
| 17 | 🧬 Player Similarity | Role match + best replacement |
| 18 | 📖 About & Model Info | Architecture + metrics vs baseline |
| 19 | 🧠 Decision Engine | Full 5-component AI decision |
| 20 | ⚡ Pressure & Momentum | Pressure index + momentum graph |
| 21 | 🔬 Deep Player Analytics | Classification + xRuns + clutch |
| 22 | 💎 Elite Fantasy Teams | 3-team system + ceiling/floor |
| 23 | 🏟️ Venue DNA Advanced | Spin/pace + best XI + phase |
| 24 | 📊 Backtesting Dashboard | Accuracy + calibration curve |

---

## Project Structure

```
cricketbrain/
├── etl/
│   ├── data_cleaning.py        # IPL pipeline, auto-detects column formats
│   ├── feature_engine.py       # 70+ features, zero leakage
│   └── insight_generator.py    # NL insights, toss advisor
│
├── ml/
│   ├── train.py                # Full ML pipeline with Optuna + SHAP
│   └── weakness_detector.py    # Phase/matchup weakness analysis
│
├── simulation/
│   └── monte_carlo.py          # 10,000-run match simulator
│
├── optimizer/
│   └── fantasy_optimizer.py    # ILP Dream11 optimizer, 3 strategies
│
├── api/
│   └── main.py                 # FastAPI, 15 endpoints, Pydantic v2
│
├── app/
│   ├── app.py                  # Main router, 24-page sidebar nav
│   ├── upgraded_pages.py       # Pages 9–18 with AI intelligence
│   ├── phase2_pages.py         # Pages 19–24, Decision Engine
│   ├── decision_engine.py      # Pressure/Momentum/xRuns/Decisions
│   └── intelligence.py         # Form classification, risk scoring
│
├── data/
│   ├── raw/IPL.csv             # ← Place Kaggle download here
│   ├── cleaned/                # matches.csv + deliveries.csv
│   ├── features/               # Engineered feature tables
│   ├── models/                 # Trained PKLs + metrics.json
│   └── shap/                   # SHAP explainers + importance CSVs
│
├── init_db.py                  # SQLite schema setup
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### 1. Get the data

Download from Kaggle and place at `data/raw/IPL.csv`:
```
https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# Mac M-series only
brew install libomp
```

### 3. Run ETL pipeline

```bash
python etl/data_cleaning.py
python etl/feature_engine.py
python init_db.py
```

### 4. Train models

```bash
python ml/train.py
# ~5–15 minutes | Trains XGBoost → LightGBM → CatBoost → Stacking → Calibration
```

### 5. Launch dashboard

```bash
streamlit run app/app.py
# http://localhost:8501
```

### 6. Launch API (optional)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Docs at http://localhost:8000/docs
```

---

## Docker

```bash
docker-compose up --build
# Dashboard → http://localhost:8501
# API       → http://localhost:8000/docs
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Match prediction + SHAP + simulation |
| `/simulate` | POST | Monte Carlo simulation |
| `/fantasy/optimize` | POST | Dream11 ILP optimizer |
| `/players/{name}/stats` | GET | Career stats |
| `/players/{name}/weakness` | GET | Weakness analysis |
| `/players/{name}/insights` | GET | AI form insights |
| `/matchup` | POST | Batsman vs bowler H2H |
| `/teams/{name}/insights` | GET | Team insights + strategy |
| `/match/preview` | GET | Pre-match AI briefing |
| `/shap/{model}` | GET | SHAP feature importance |
| `/model/info` | GET | Metrics + feature list |
| `/health` | GET | System health |

---

## Key Innovations

| Innovation | Details |
|---|---|
| Pressure Index | Proprietary: f(required_rate, wickets_left, overs_left) |
| Momentum Index | Last 12 balls, exponential decay weighted |
| EMA Form | α=0.3 exponentially weighted batting average |
| Zero Leakage | All features use `shift(1).rolling()` |
| Clutch Score | High-pressure vs normal performance ratio |
| Pitch DNA | Auto pitch classification from venue stats |
| ILP Optimizer | PuLP CBC — guaranteed optimal team, not greedy |
| xRuns/xWickets | Expected value metrics beyond raw averages |
| Calibration | Isotonic regression — probabilities are meaningful |
| Backtesting | Season-by-season accuracy with calibration curve |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data | Pandas, NumPy, PyArrow |
| ML | XGBoost, LightGBM, CatBoost, scikit-learn |
| Tuning | Optuna (Bayesian) |
| Explainability | SHAP |
| Optimization | PuLP (ILP) |
| Simulation | NumPy Monte Carlo |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Database | SQLite |
| Deployment | Docker + docker-compose |

---

*MIT License — free for personal, educational, and portfolio use.*
📊 Dashboard Preview

🔥 IPL League Overview

<p align="center">
  <img src="assets/screenshots/overview.png" width="85%">
</p>🤖 AI Match Predictor

<p align="center">
  <img src="assets/screenshots/predictor.png" width="85%">
</p>🧠 Explainable AI (SHAP)

<p align="center">
  <img src="assets/screenshots/shap.png" width="85%">
</p>💰 Fantasy Team Optimizer

<p align="center">
  <img src="assets/screenshots/optimizer.png" width="85%">
</p>

👨‍💻 Author

Pradip Kumar Verma
AI & Data Science StudentPassionate about building real-world ML systems 🚀