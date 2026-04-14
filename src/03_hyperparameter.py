"""
03_hyperparameter.py — Phase 3 : Optimisation des hyperparamètres
==================================================================
Recherche des meilleurs hyperparamètres via Optuna (Bayesian TPE)
avec pruning successive halving (MedianPruner) et tracking MLflow.

Améliorations vs version notebook :
    ✓ Cible = residual (aligné avec 04_training.py)
    ✓ Dataset complet (pas de sous-échantillon)
    ✓ Optuna TPE pour tous les modèles (bayésien > random search)
    ✓ MedianPruner = successive halving intégré dans Optuna
    ✓ Early stopping XGBoost sur split temporel dédié
    ✓ MLflow tracking de chaque trial
    ✓ n_trials élevé (conventions standard)
    ✓ SimpleImputer dans tous les pipelines (cohérence avec 04_training)

Entrées  : data/processed/train_preprocessed.csv
Sorties  : data/outputs/best_params.json
           mlruns/  (expériences MLflow)

Conventions n_trials :
    Modèles simples (Ridge)        :  50 trials  — espace réduit
    Modèles complexes (RF, GB)     : 100 trials  — espace moyen
    Modèles très complexes (XGB)   : 150 trials  — espace large + early stopping
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.dummy          import DummyRegressor
from sklearn.ensemble       import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute         import SimpleImputer
from sklearn.linear_model   import Ridge
from sklearn.metrics        import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
import xgboost as xgb

from config import (
    TRAIN_FILE, BEST_PARAMS_FILE,
    FEATURES_FINAL, TARGET,
    RANDOM_SEED,
    section, ok, info,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_CV_SPLITS = 3

# Conventions n_trials selon complexité du modèle
N_TRIALS = {
    "Ridge":             50,
    "RandomForest":     100,
    "GradientBoosting": 100,
    "XGBoost":          150,
}

# MLflow — fallback sur stockage local si le serveur n'est pas disponible
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("velib_hyperparameter_tuning")
    MLFLOW_OK = True
    info(f"MLflow connecté : {MLFLOW_URI}")
except Exception:
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("velib_hyperparameter_tuning")
    MLFLOW_OK = True
    info("MLflow — stockage local : /app/mlruns")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def make_tscv() -> TimeSeriesSplit:
    """Validation croisée temporelle — respecte l'ordre chronologique."""
    return TimeSeriesSplit(n_splits=N_CV_SPLITS)


def eval_cv(pipeline: Pipeline,
            X: pd.DataFrame,
            y: pd.Series,
            cv: TimeSeriesSplit) -> float:
    """
    MAE moyenne sur les folds de CV temporelle.
    neg_mean_absolute_error : sklearn retourne des valeurs négatives → on inverse.
    """
    scores = cross_val_score(
        pipeline, X, y,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def make_base_pipeline(*steps) -> Pipeline:
    """
    Construit un pipeline avec SimpleImputer en première étape.
    Le SimpleImputer(median) remplace les NaN par la médiane de la feature.
    Nécessaire pour Ridge (et tous les modèles linéaires) qui rejettent les NaN.
    Les modèles à base d'arbres (XGB, RF, GB) tolèrent les NaN nativement
    mais on normalise l'approche pour tous.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        *steps,
    ])


def log_trial_to_mlflow(trial_number: int,
                         model_name: str,
                         params: dict,
                         mae: float) -> None:
    """Enregistre un trial Optuna dans MLflow (run imbriqué)."""
    with mlflow.start_run(run_name=f"{model_name}_trial_{trial_number:03d}",
                          nested=True):
        mlflow.log_param("model", model_name)
        mlflow.log_params(params)
        mlflow.log_metric("mae_cv", mae)


def es_split(X: pd.DataFrame,
             y: pd.Series,
             ratio: float = 0.85) -> tuple:
    """
    Split temporel 85/15 pour l'early stopping XGBoost.
    Séparé des folds de CV pour ne pas contaminer l'évaluation.
    """
    n = int(len(X) * ratio)
    return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def tune_baseline(X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict:
    """Baseline : prédire toujours la moyenne. Référence pour tous les modèles."""
    pipeline = Pipeline([("model", DummyRegressor(strategy="mean"))])
    mae      = eval_cv(pipeline, X, y, cv)
    ok(f"Baseline MAE CV = {mae:.2f}%  (référence à battre)")
    return {"mae": round(mae, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# RIDGE — 50 trials
# ─────────────────────────────────────────────────────────────────────────────

def tune_ridge(X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict:
    """
    Ridge Regression — espace réduit à 1 hyperparamètre (alpha).
    50 trials suffisent pour explorer log-uniformément [1e-4, 1e3].

    MedianPruner : arrête les trials dont les scores intermédiaires
    sont sous la médiane des trials précédents → successive halving implicite.
    """
    def objective(trial: optuna.Trial) -> float:
        alpha    = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        pipeline = make_base_pipeline(
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=alpha)),
        )
        mae = eval_cv(pipeline, X, y, cv)
        log_trial_to_mlflow(trial.number, "Ridge", {"alpha": alpha}, mae)
        return mae

    with mlflow.start_run(run_name="Ridge_optuna"):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                               n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=N_TRIALS["Ridge"], show_progress_bar=True)
        best = study.best_params
        mae  = study.best_value
        mlflow.log_params(best)
        mlflow.log_metric("best_mae_cv", mae)

    ok(f"Ridge — alpha={best['alpha']:.4f}  MAE={mae:.2f}%  ({N_TRIALS['Ridge']} trials)")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# RANDOM FOREST — 100 trials
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_forest(X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict:
    """
    Random Forest — 5 hyperparamètres, 100 trials Optuna TPE.
    TPE (Tree-structured Parzen Estimator) apprend des trials précédents
    et concentre la recherche sur les zones prometteuses.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     100, 500),
            "max_depth":        trial.suggest_int("max_depth",         4,  15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf",  5,  150),
            "max_features":     trial.suggest_float("max_features",    0.3,  1.0),
            "min_samples_split":trial.suggest_int("min_samples_split", 10, 200),
        }
        pipeline = make_base_pipeline(
            ("model", RandomForestRegressor(
                **params, n_jobs=-1, random_state=RANDOM_SEED
            )),
        )
        mae = eval_cv(pipeline, X, y, cv)
        log_trial_to_mlflow(trial.number, "RandomForest", params, mae)
        return mae

    with mlflow.start_run(run_name="RandomForest_optuna"):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED,
                                               n_startup_trials=20),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20,
                                               n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=N_TRIALS["RandomForest"],
                       show_progress_bar=True)
        best = study.best_params
        mae  = study.best_value
        mlflow.log_params(best)
        mlflow.log_metric("best_mae_cv", mae)

    ok(f"RandomForest — MAE={mae:.2f}%  ({N_TRIALS['RandomForest']} trials)")
    info(f"  {best}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT BOOSTING — 100 trials
# ─────────────────────────────────────────────────────────────────────────────

def tune_gradient_boosting(X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict:
    """
    Gradient Boosting sklearn — 6 hyperparamètres, 100 trials.
    Espace plus large que Ridge → plus de trials nécessaires.
    """
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",      100, 500),
            "max_depth":        trial.suggest_int("max_depth",            3,   7),
            "learning_rate":    trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample",        0.5,  1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf",    10, 150),
            "max_features":     trial.suggest_float("max_features",     0.3,  1.0),
        }
        pipeline = make_base_pipeline(
            ("model", GradientBoostingRegressor(
                **params, random_state=RANDOM_SEED
            )),
        )
        mae = eval_cv(pipeline, X, y, cv)
        log_trial_to_mlflow(trial.number, "GradientBoosting", params, mae)
        return mae

    with mlflow.start_run(run_name="GradientBoosting_optuna"):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED,
                                               n_startup_trials=20),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20,
                                               n_warmup_steps=1),
        )
        study.optimize(objective, n_trials=N_TRIALS["GradientBoosting"],
                       show_progress_bar=True)
        best = study.best_params
        mae  = study.best_value
        mlflow.log_params(best)
        mlflow.log_metric("best_mae_cv", mae)

    ok(f"GradientBoosting — MAE={mae:.2f}%  ({N_TRIALS['GradientBoosting']} trials)")
    info(f"  {best}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST — 150 trials + early stopping
# ─────────────────────────────────────────────────────────────────────────────

def tune_xgboost(X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit) -> dict:
    """
    XGBoost — 150 trials Optuna TPE + early stopping.

    Early stopping :
        Plutôt que de fixer n_estimators arbitrairement, on laisse XGBoost
        s'arrêter automatiquement quand le score ne s'améliore plus sur
        un split de validation temporel dédié (85% train / 15% val).
        early_stopping_rounds=50 : arrêt si pas d'amélioration sur 50 rounds.
        Cela évite l'overfitting et accélère drastiquement chaque trial.

    n_estimators dans Optuna :
        On optimise le n_estimators maximum autorisé (borne haute).
        Le nombre réel d'arbres est déterminé par l'early stopping.
    """
    # Split temporel dédié à l'early stopping (hors CV)
    X_es_tr, X_es_val, y_es_tr, y_es_val = es_split(X, y, ratio=0.85)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      200,  800),
            "max_depth":         trial.suggest_int("max_depth",            3,   9),
            "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample",        0.5,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5,  1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel",0.5,  1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight",     1,  30),
            "reg_alpha":         trial.suggest_float("reg_alpha",        1e-5, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",       1e-5, 10.0, log=True),
            "gamma":             trial.suggest_float("gamma",            1e-5,  5.0, log=True),
        }

        # Imputer avant XGBoost (cohérence pipeline)
        imputer  = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_es_tr)
        X_vl_imp = imputer.transform(X_es_val)

        # Early stopping — détermine le nombre réel d'arbres nécessaires
        # On utilise n_estimators comme borne haute, l'early stopping arrête avant
        model_es = xgb.XGBRegressor(
            **params,
            tree_method="hist",
            random_state=RANDOM_SEED,
            verbosity=0,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="mae",
        )
        model_es.fit(
            X_tr_imp, y_es_tr,
            eval_set=[(X_vl_imp, y_es_val)],
            verbose=False,
        )

        # Nombre réel d'arbres après early stopping
        best_n_estimators = model_es.best_iteration + 1

        # Évaluation CV avec le nombre d'arbres optimal trouvé par early stopping
        # On remplace n_estimators dans params par la valeur optimale
        params_cv = {**params, "n_estimators": best_n_estimators}

        pipeline = make_base_pipeline(
            ("model", xgb.XGBRegressor(
                **params_cv,
                tree_method="hist",
                random_state=RANDOM_SEED,
                verbosity=0, n_jobs=-1,
            )),
        )
        mae = eval_cv(pipeline, X, y, cv)
        log_trial_to_mlflow(trial.number, "XGBoost",
                            {**params_cv, "best_n_estimators": best_n_estimators}, mae)
        return mae

    with mlflow.start_run(run_name="XGBoost_optuna"):
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=RANDOM_SEED,
                n_startup_trials=30,      # 30 trials aléatoires avant TPE
                multivariate=True,        # capture les corrélations entre params
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=30,
                n_warmup_steps=1,
                interval_steps=1,
            ),
        )
        study.optimize(objective, n_trials=N_TRIALS["XGBoost"],
                       show_progress_bar=True)
        best = study.best_params
        mae  = study.best_value
        mlflow.log_params(best)
        mlflow.log_metric("best_mae_cv", mae)

        # Optuna importance des hyperparamètres
        try:
            importance = optuna.importance.get_param_importances(study)
            info("  Importance des hyperparamètres XGBoost :")
            for param, imp in list(importance.items())[:5]:
                info(f"    {param:<25} : {imp:.3f}")
        except Exception:
            pass

    ok(f"XGBoost — MAE={mae:.2f}%  ({N_TRIALS['XGBoost']} trials + early stopping)")
    info(f"  {best}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _save(best_params: dict) -> None:
    """
    Sauvegarde best_params.json après chaque modèle optimisé.
    Permet de reprendre le script depuis le dernier modèle terminé
    en cas d'interruption, sans tout relancer depuis le début.
    """
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
    ok(f"best_params.json sauvegardé ({list(best_params.keys())})")


def main() -> None:
    section("PHASE 3 — HYPERPARAMETER TUNING (Optuna + MLflow)")

    # ── Chargement du dataset complet ─────────────────────────────────────
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Train non trouvé : {TRAIN_FILE}\n"
            "Lancer 02_preprocessing.py d'abord."
        )

    train_df = pd.read_csv(TRAIN_FILE)
    ok(f"Train chargé : {len(train_df):,} lignes (dataset complet)")

    # ── Chargement des résultats existants ────────────────────────────────
    # Si best_params.json existe déjà, on charge les résultats précédents
    # et on ne relance que les modèles manquants (reprise après interruption)
    if BEST_PARAMS_FILE.exists():
        with open(BEST_PARAMS_FILE) as f:
            best_params = json.load(f)
        info(f"best_params.json existant chargé — modèles déjà optimisés :")
        for name in best_params:
            ok(f"  {name} — déjà optimisé, ignoré")
    else:
        best_params = {}

    # ── Vérification des features disponibles ─────────────────────────────
    available = [f for f in FEATURES_FINAL if f in train_df.columns]
    missing   = [f for f in FEATURES_FINAL if f not in train_df.columns]
    if missing:
        info(f"Features absentes ignorées : {missing}")

    X = train_df[available]
    if TARGET not in train_df.columns:
        raise ValueError(
            f"Colonne '{TARGET}' absente — vérifier que 02_preprocessing.py "
            "a bien été lancé (il calcule la colonne residual)."
        )
    y = train_df[TARGET]

    info(f"Features : {len(available)}")
    info(f"Cible    : '{TARGET}'  (résidu — aligné avec l'entraînement)")
    info(f"Lignes   : {len(X):,}  (dataset complet — pas de sous-échantillon)")

    cv = make_tscv()
    info(f"CV       : TimeSeriesSplit({N_CV_SPLITS} folds)")

    # ── Baseline ──────────────────────────────────────────────────────────
    if "Baseline" not in best_params:
        section("Baseline")
        best_params["Baseline"] = tune_baseline(X, y, cv)
        _save(best_params)
    else:
        info("Baseline — déjà dans best_params.json, ignoré")

    # ── Ridge ─────────────────────────────────────────────────────────────
    if "Ridge" not in best_params:
        section(f"Ridge — {N_TRIALS['Ridge']} trials Optuna TPE")
        best_params["Ridge"] = tune_ridge(X, y, cv)
        _save(best_params)
    else:
        info("Ridge — déjà dans best_params.json, ignoré")

    # ── Random Forest ─────────────────────────────────────────────────────
    if "RandomForest" not in best_params:
        section(f"Random Forest — {N_TRIALS['RandomForest']} trials Optuna TPE")
        best_params["RandomForest"] = tune_random_forest(X, y, cv)
        _save(best_params)
    else:
        info("RandomForest — déjà dans best_params.json, ignoré")

    # ── Gradient Boosting ─────────────────────────────────────────────────
    if "GradientBoosting" not in best_params:
        section(f"Gradient Boosting — {N_TRIALS['GradientBoosting']} trials Optuna TPE")
        best_params["GradientBoosting"] = tune_gradient_boosting(X, y, cv)
        _save(best_params)
    else:
        info("GradientBoosting — déjà dans best_params.json, ignoré")

    # ── XGBoost ───────────────────────────────────────────────────────────
    if "XGBoost" not in best_params:
        section(f"XGBoost — {N_TRIALS['XGBoost']} trials Optuna TPE + Early Stopping")
        best_params["XGBoost"] = tune_xgboost(X, y, cv)
        _save(best_params)
    else:
        info("XGBoost — déjà dans best_params.json, ignoré")

    # ── Récapitulatif ─────────────────────────────────────────────────────
    section("RÉCAPITULATIF")
    print(f"\n  {'Modèle':<25} {'Trials':>8}")
    print(f"  {'─' * 35}")
    print(f"  {'Baseline':<25} {'—':>8}")
    for name in ["Ridge", "RandomForest", "GradientBoosting", "XGBoost"]:
        print(f"  {name:<25} {N_TRIALS[name]:>8}")

    section("PHASE 3 TERMINÉE")
    info(f"MLflow UI : http://localhost:5000  (si le service mlflow tourne)")
    info(f"Résultats : {BEST_PARAMS_FILE}")


if __name__ == "__main__":
    main()
