"""
CricketBrain AI — Advanced ML Training Pipeline
XGBoost + LightGBM + CatBoost + Stacking Ensemble
Optuna hyperparameter tuning | SHAP explainability | Calibration
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score,
    precision_recall_curve, average_precision_score,
    brier_score_loss
)
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[ML] XGBoost not found, skipping")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[ML] LightGBM not found, skipping")

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
    print("[ML] CatBoost not found, skipping")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[ML] Optuna not found, using default hyperparameters")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[ML] SHAP not found, skipping explainability")

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR  = os.path.join(DATA_DIR, "models")
SHAP_DIR   = os.path.join(DATA_DIR, "shap")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# FEATURE COLUMNS
# ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Team form
    "team1_win_rate_last3","team1_win_rate_last5","team1_win_rate_last10","team1_win_rate_last15",
    "team2_win_rate_last3","team2_win_rate_last5","team2_win_rate_last10","team2_win_rate_last15",
    "team1_form_score","team2_form_score",
    "team1_win_streak","team2_win_streak",
    # Differential
    "diff_win_rate_last3","diff_win_rate_last5","diff_win_rate_last10","diff_form_score",
    # Toss
    "toss_decision_bat","toss_won_by_team1",
    # Venue
    "venue_bat_first_win_rate","venue_chase_win_rate","venue_matches",
    # H2H
    "h2h_total","h2h_team1_win_rate","h2h_team2_win_rate",
    # Strength
    "t1_bat_str","t1_bat_depth","t2_bat_str","t2_bat_depth",
    # Season
    "season_num",
]

def load_features():
    path = os.path.join(DATA_DIR, "features", "match_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run feature_engine.py first! Missing: {path}")
    df = pd.read_csv(path)
    print(f"[ML] Loaded features: {df.shape}")
    return df

def prepare_data(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[ML] Missing features (will skip): {missing}")

    X = df[available].fillna(0).astype(float)
    y = df["target"].fillna(0).astype(int)

    # Time-based split: last 2 seasons = test
    seasons = sorted(df["season"].dropna().unique())
    test_seasons = seasons[-2:]
    train_mask = df["season"] < min(test_seasons)
    test_mask  = df["season"] >= min(test_seasons)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"[ML] Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(available)}")
    print(f"[ML] Test seasons: {test_seasons}")
    return X_train, X_test, y_train, y_test, available

# ─────────────────────────────────────────────────────────
# OPTUNA TUNING
# ─────────────────────────────────────────────────────────
def tune_xgboost(X_train, y_train, n_trials=50):
    if not HAS_OPTUNA or not HAS_XGB:
        return {"n_estimators":300,"max_depth":5,"learning_rate":0.05,"subsample":0.8,"colsample_bytree":0.8,"min_child_weight":3,"gamma":0.1,"reg_alpha":0.1,"reg_lambda":1.0}

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 100, 600),
            "max_depth":       trial.suggest_int("max_depth", 3, 8),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
            "gamma":           trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha":       trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 3),
            "use_label_encoder": False, "eval_metric": "logloss",
            "random_state": 42, "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[ML] XGB best AUC: {study.best_value:.4f}")
    return study.best_params

def tune_lgbm(X_train, y_train, n_trials=40):
    if not HAS_OPTUNA or not HAS_LGB:
        return {"n_estimators":300,"max_depth":6,"learning_rate":0.05,"num_leaves":50,"subsample":0.8,"colsample_bytree":0.8,"min_child_samples":20}

    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 100),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 50),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[ML] LGB best AUC: {study.best_value:.4f}")
    return study.best_params

# ─────────────────────────────────────────────────────────
# EVALUATE MODEL
# ─────────────────────────────────────────────────────────
def evaluate(name, model, X_train, X_test, y_train, y_test):
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc", n_jobs=-1)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    pred  = model.predict(X_test)

    metrics = {
        "roc_auc":    roc_auc_score(y_test, proba),
        "log_loss":   log_loss(y_test, proba),
        "accuracy":   accuracy_score(y_test, pred),
        "avg_precision": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "cv_mean":    float(cv_scores.mean()),
        "cv_std":     float(cv_scores.std()),
    }
    print(f"[ML] {name:25s} | AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f} | CV: {metrics['cv_mean']:.4f}±{metrics['cv_std']:.4f}")
    return metrics

# ─────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────
def compute_shap(model, X_train, feature_names, model_name):
    if not HAS_SHAP:
        return
    try:
        print(f"[ML] Computing SHAP for {model_name}...")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train[:500])
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        shap_df = shap_df.sort_values("importance", ascending=False)
        shap_df.to_csv(os.path.join(SHAP_DIR, f"shap_{model_name}.csv"), index=False)
        joblib.dump(explainer, os.path.join(SHAP_DIR, f"explainer_{model_name}.pkl"))
        print(f"[ML] SHAP saved: {model_name}")
        return shap_df
    except Exception as e:
        print(f"[ML] SHAP failed for {model_name}: {e}")
        return None

# ─────────────────────────────────────────────────────────
# MAIN TRAINING
# ─────────────────────────────────────────────────────────
def train():
    df = load_features()
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df)

    all_metrics = {}
    trained_models = {}

    # ── RandomForest baseline ──
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    all_metrics["RandomForest"] = evaluate("RandomForest", rf, X_train, X_test, y_train, y_test)
    trained_models["RandomForest"] = rf

    # ── XGBoost ──
    if HAS_XGB:
        print("[ML] Tuning XGBoost (Optuna 50 trials)...")
        xgb_params = tune_xgboost(X_train, y_train, n_trials=50)
        xgb_params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1})
        xgb_params.pop("use_label_encoder", None)
        xgb_model = xgb.XGBClassifier(**xgb_params)
        all_metrics["XGBoost"] = evaluate("XGBoost", xgb_model, X_train, X_test, y_train, y_test)
        trained_models["XGBoost"] = xgb_model
        compute_shap(xgb_model, X_train, feature_names, "xgboost")

    # ── LightGBM ──
    if HAS_LGB:
        print("[ML] Tuning LightGBM (Optuna 40 trials)...")
        lgb_params = tune_lgbm(X_train, y_train, n_trials=40)
        lgb_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        all_metrics["LightGBM"] = evaluate("LightGBM", lgb_model, X_train, X_test, y_train, y_test)
        trained_models["LightGBM"] = lgb_model
        compute_shap(lgb_model, X_train, feature_names, "lightgbm")

    # ── CatBoost ──
    if HAS_CB:
        print("[ML] Training CatBoost...")
        cb_model = cb.CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            eval_metric="AUC", random_seed=42, verbose=0,
            l2_leaf_reg=3, border_count=128
        )
        all_metrics["CatBoost"] = evaluate("CatBoost", cb_model, X_train, X_test, y_train, y_test)
        trained_models["CatBoost"] = cb_model

    # ── Stacking Ensemble ──
    base_estimators = []
    if HAS_XGB and "XGBoost" in trained_models:
        base_estimators.append(("xgb", trained_models["XGBoost"]))
    if HAS_LGB and "LightGBM" in trained_models:
        base_estimators.append(("lgb", trained_models["LightGBM"]))
    base_estimators.append(("rf", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)))

    if len(base_estimators) >= 2:
        print("[ML] Training Stacking Ensemble...")
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000),
            cv=5, n_jobs=-1, passthrough=False
        )
        all_metrics["Stacking"] = evaluate("Stacking", stack, X_train, X_test, y_train, y_test)
        trained_models["Stacking"] = stack

    # ── Best Model + Calibration ──
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    best_model = trained_models[best_name]
    print(f"\n[ML] 🏆 Best model: {best_name} (AUC={all_metrics[best_name]['roc_auc']:.4f})")

    # Calibrate probabilities
    best_model.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    calibrated.fit(X_train, y_train)
    calib_proba = calibrated.predict_proba(X_test)[:,1]
    calib_auc = roc_auc_score(y_test, calib_proba)
    print(f"[ML] Calibrated AUC: {calib_auc:.4f}")

    # ── Save everything ──
    joblib.dump(calibrated, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(trained_models, os.path.join(MODEL_DIR, "all_models.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    # Save metrics
    all_metrics["best_model"] = best_name
    all_metrics["calibrated_auc"] = calib_auc
    all_metrics["feature_count"] = len(feature_names)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[ML] ✅ Training complete! Models saved to: {MODEL_DIR}")
    return calibrated, all_metrics, feature_names

if __name__ == "__main__":
    train()
