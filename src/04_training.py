"""
04_training.py — Phase 4 : Entraînement & Évaluation des modèles
=================================================================
Entraîne tous les modèles retenus, évalue leurs performances et
exporte les résultats et le meilleur modèle.

Entrées  : data/processed/train_preprocessed.csv
           data/processed/test_preprocessed.csv
           data/outputs/best_params.json
Sorties  : data/outputs/resultats_modeles_v2.csv
           data/outputs/feature_importance.csv
           data/outputs/best_model.joblib
           mlruns/  (expériences MLflow — chaque modèle = 1 run)

Ce que prédit chaque modèle :
    TARGET = résidu = taux_réel - station_trend_avg
    Reconstruction : taux_prédit = station_trend_avg + résidu_prédit

Note : la CV temporelle a été retirée — avec ~1 mois de données,
    les folds (~6 jours d'entraînement) sont trop courts pour capturer
    les habitudes hebdomadaires. Le test set temporel 80/20 est la
    métrique de référence.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.dummy          import DummyRegressor
from sklearn.impute         import SimpleImputer
from sklearn.linear_model   import Ridge
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
import xgboost as xgb

from config import (
    TRAIN_FILE, TEST_FILE,
    BEST_PARAMS_FILE, RESULTS_FILE,
    IMPORTANCE_FILE, MODEL_FILE,
    FEATURES_FINAL, TARGET, TARGET_RAW,
    RANDOM_SEED,
    section, ok, info,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

# Fallback local si le serveur MLflow n'est pas disponible
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("velib_training")
    MLFLOW_OK = True
    info(f"MLflow connecté : {MLFLOW_URI}")
except Exception:
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("velib_training")
    MLFLOW_OK = True
    info("MLflow — stockage local : /app/mlruns")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule MAE, RMSE et R² en une seule passe."""
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
    }


def load_best_params(path: Path) -> dict:
    """Charge les hyperparamètres depuis le JSON produit par 03_hyperparameter.py."""
    if not path.exists():
        info("best_params.json non trouvé — utilisation des paramètres par défaut")
        return {}
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

def make_base_pipeline(*steps) -> Pipeline:
    """
    Construit un pipeline avec SimpleImputer(median) en première étape.

    Pourquoi SimpleImputer ?
        Certaines features contiennent des NaN résiduels :
        - morning_evening_ratio : stations sans données matin ou soir
        - lag_60min / lag_240min : relevés en début d'historique
        Ridge et les modèles linéaires rejettent les NaN → crash.
        SimpleImputer remplace les NaN par la médiane de chaque feature.
        Les modèles à arbres (XGB, RF) gèrent les NaN nativement
        mais on uniformise l'approche pour la cohérence.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        *steps,
    ])


def build_models(params: dict) -> dict:
    """
    Construit les pipelines sklearn pour chaque modèle.

    Paramètres alignés sur le notebook 3_modeling.ipynb (résultats Optuna).
    Si best_params.json existe (03_hyperparameter.py), il prend la priorité.
    """
    ridge_params = params.get("Ridge", {"alpha": 990.4792})

    rf_params = params.get("RandomForest", {
        "n_estimators":    468,
        "max_depth":        15,
        "min_samples_leaf": 146,
        "max_features":     0.9825487423645705,
    })

    xgb_raw = params.get("XGBoost", {
        "n_estimators":      232,
        "max_depth":           6,
        "learning_rate":       0.12272662063860754,
        "subsample":           0.9372938568071518,
        "colsample_bytree":    0.5236769398724859,
        "colsample_bylevel":   0.562214355487758,
        "min_child_weight":    21,
        "reg_alpha":           1.4002604468647064,
        "reg_lambda":          1.4716333468536236,
        "gamma":               0.7759872843048987,
    })
    xgb_params = {k: v for k, v in xgb_raw.items()
                  if k not in ["n_estimators_actual"]}

    return {
        "Baseline (moyenne)": Pipeline([
            ("model", DummyRegressor(strategy="mean"))
        ]),
        "Ridge": make_base_pipeline(
            ("scaler", StandardScaler()),
            ("model",  Ridge(**ridge_params)),
        ),
        "Random Forest": make_base_pipeline(
            ("model", RandomForestRegressor(
                **rf_params, n_jobs=-1, random_state=RANDOM_SEED
            )),
        ),
        "XGBoost": make_base_pipeline(
            ("model", xgb.XGBRegressor(
                **xgb_params,
                tree_method="hist",
                random_state=RANDOM_SEED,
                verbosity=0, n_jobs=-1,
            )),
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRAÎNEMENT, ÉVALUATION ET LOGGING MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(models: dict,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_test:  pd.DataFrame, y_test:  pd.Series,
                       test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Entraîne chaque modèle, évalue sur le test set et logue dans MLflow.

    Chaque modèle = 1 run MLflow avec :
        - params   : hyperparamètres du modèle
        - metrics  : MAE, RMSE, R², R²_reconstruit, temps d'entraînement
        - tags     : model_name, is_best (mis à jour après le classement)
        - artifacts: best_model.joblib + feature_importance.csv (meilleur modèle)
    """
    print(f"\n{'=' * 70}")
    print(f"  {'Modèle':<35} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Temps':>8}")
    print(f"{'=' * 70}")

    results = []
    run_ids = {}  # nom → run_id pour tagger le meilleur modèle après classement

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            # ── Entraînement ──────────────────────────────────────────────
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            duree  = time.time() - t0
            y_pred = pipeline.predict(X_test)

            # ── Métriques sur le résidu ───────────────────────────────────
            m = metrics(y_test.values, y_pred)

            # ── Métriques sur le taux reconstruit ─────────────────────────
            taux_pred = np.clip(
                test_df["station_trend_avg"].values + y_pred, 0, 100
            )
            taux_reel = test_df[TARGET_RAW].values
            r2_reconst = round(float(r2_score(taux_reel, taux_pred)), 4)

            # ── Logging MLflow ────────────────────────────────────────────
            # Tag identifiant le modèle
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("is_best", "false")  # mis à jour plus bas

            # Hyperparamètres : extraits depuis le step "model" du pipeline
            model_step = pipeline.named_steps["model"]
            model_params = model_step.get_params()
            # On préfixe pour éviter les collisions avec d'autres runs
            mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
            mlflow.log_param("training_rows",  len(X_train))
            mlflow.log_param("test_rows",      len(X_test))
            mlflow.log_param("n_features",     X_train.shape[1])

            # Métriques
            mlflow.log_metric("MAE",           m["MAE"])
            mlflow.log_metric("RMSE",          m["RMSE"])
            mlflow.log_metric("R2_residu",     m["R2"])
            mlflow.log_metric("R2_reconstruit", r2_reconst)
            mlflow.log_metric("train_time_s",  round(duree, 2))

            run_ids[name] = mlflow.active_run().info.run_id

        # ── Résumé console ────────────────────────────────────────────────
        results.append({
            "Modele": name,
            "MAE":    m["MAE"],
            "RMSE":   m["RMSE"],
            "R2":     m["R2"],
            "Temps":  round(duree, 1),
        })
        print(f"  {name:<35} {m['MAE']:>6.2f}% {m['RMSE']:>6.2f}% {m['R2']:>7.4f} {duree:>6.1f}s")

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)

    # ── Tagger le meilleur modèle dans MLflow ─────────────────────────────
    best_name   = results_df.iloc[0]["Modele"]
    best_run_id = run_ids[best_name]
    mlflow.MlflowClient().set_tag(best_run_id, "is_best", "true")
    ok(f"Run MLflow '{best_name}' tagué is_best=true (run_id={best_run_id[:8]}…)")

    return results_df, models, run_ids


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANCE DES FEATURES (meilleur modèle)
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importance(best_pipeline: Pipeline,
                                feature_names: list) -> pd.DataFrame:
    """
    Extrait l'importance des features du meilleur modèle.
    Compatible XGBoost, Random Forest.
    Pour Ridge, utilise les coefficients absolus normalisés.
    """
    model = best_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        importances = importances / importances.sum()
    else:
        return pd.DataFrame()

    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": (importances * 100).round(2),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENREGISTREMENT DANS LE MODEL REGISTRY MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

def register_best_model(best_pipeline: Pipeline,
                        best_name: str,
                        best_run_id: str,
                        imp_df: pd.DataFrame) -> None:
    """
    Logue le modèle final et l'importance des features dans le run MLflow
    du meilleur modèle, puis enregistre dans le Model Registry.

    Le Model Registry permet de versionner et promouvoir les modèles :
        None → Staging → Production
    """
    with mlflow.start_run(run_id=best_run_id):

        # ── Artifact : importance des features ───────────────────────────
        if not imp_df.empty:
            tmp = Path("/tmp/feature_importance.csv")
            imp_df.to_csv(tmp, index=False)
            mlflow.log_artifact(str(tmp), artifact_path="features")

        # ── Artifact : modèle sérialisé (joblib) ─────────────────────────
        # log_artifact logue le fichier brut (utile pour récupérer le .joblib exact)
        mlflow.log_artifact(str(MODEL_FILE), artifact_path="model_joblib")

        # ── Model Registry ────────────────────────────────────────────────
        # mlflow.sklearn.log_model logue le pipeline sklearn dans le format MLflow
        # et permet l'enregistrement dans le registry avec un nom standardisé.
        #
        # registered_model_name : nom du modèle dans le registry
        #   → chaque appel crée une nouvelle version (v1, v2, v3...)
        #   → on peut ensuite promouvoir une version en "Staging" ou "Production"
        #      depuis l'UI MLflow ou via mlflow.MlflowClient()
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="pipeline",
            registered_model_name="velib_fill_rate_predictor",
            input_example=None,
        )

    ok(f"Modèle enregistré dans le registry MLflow : 'velib_fill_rate_predictor'")
    info("  Accéder au registry : http://localhost:5000/#/models/velib_fill_rate_predictor")
    info("  Promouvoir en Staging/Production depuis l'UI MLflow ou via MlflowClient")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    section("PHASE 4 — ENTRAÎNEMENT DES MODÈLES")

    # ── Chargement ────────────────────────────────────────────────────────
    for path in [TRAIN_FILE, TEST_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier non trouvé : {path}\nLancer 02_preprocessing.py d'abord."
            )

    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)
    ok(f"Train : {len(train_df):,} lignes × {train_df.shape[1]} colonnes")
    ok(f"Test  : {len(test_df):,}  lignes × {test_df.shape[1]} colonnes")

    # ── Vérification des features ─────────────────────────────────────────
    available = [f for f in FEATURES_FINAL if f in train_df.columns]
    missing   = [f for f in FEATURES_FINAL if f not in train_df.columns]
    if missing:
        info(f"Features absentes ignorées : {missing}")

    X_train = train_df[available]
    y_train = train_df[TARGET]
    X_test  = test_df[available]
    y_test  = test_df[TARGET]

    info(f"Features utilisées : {len(available)}/{len(FEATURES_FINAL)}")

    # ── Chargement des hyperparamètres ────────────────────────────────────
    params = load_best_params(BEST_PARAMS_FILE)

    # ── Construction des modèles ──────────────────────────────────────────
    models = build_models(params)
    ok(f"{len(models)} modèles définis")

    # ── Entraînement, évaluation et logging MLflow ────────────────────────
    section("Entraînement, évaluation et logging MLflow")
    results_df, models, run_ids = train_and_evaluate(
        models, X_train, y_train, X_test, y_test, test_df
    )

    results_df.to_csv(RESULTS_FILE, index=False)
    ok(f"Résultats exportés → {RESULTS_FILE.name}")

    # ── Analyse du meilleur modèle ────────────────────────────────────────
    section("Analyse du meilleur modèle")
    best_name     = results_df.iloc[0]["Modele"]
    best_pipeline = models[best_name]
    y_pred        = best_pipeline.predict(X_test)

    ok(f"Meilleur modèle : {best_name}")
    info(f"  MAE  : {results_df.iloc[0]['MAE']:.2f}%")
    info(f"  RMSE : {results_df.iloc[0]['RMSE']:.2f}%")
    info(f"  R²   : {results_df.iloc[0]['R2']:.4f}")

    # Reconstruction du taux réel
    taux_pred  = np.clip(test_df["station_trend_avg"].values + y_pred, 0, 100)
    taux_reel  = test_df[TARGET_RAW].values
    r2_reconst = r2_score(taux_reel, taux_pred)
    info(f"  R² taux reconstruit : {r2_reconst:.4f}  (intègre station_trend_avg)")

    # ── Importance des features ───────────────────────────────────────────
    section("Importance des features")
    imp_df = extract_feature_importance(best_pipeline, available)
    if not imp_df.empty:
        imp_df.to_csv(IMPORTANCE_FILE, index=False)
        ok(f"Importances exportées → {IMPORTANCE_FILE.name}")
        print()
        for _, row in imp_df.head(10).iterrows():
            bar = "█" * int(row["importance"] / 2)
            print(f"  {row['feature']:<30} {bar} {row['importance']:.1f}%")

    # ── Sauvegarde locale du meilleur modèle ──────────────────────────────
    section("Sauvegarde du modèle")
    joblib.dump(best_pipeline, MODEL_FILE)
    ok(f"Modèle sauvegardé → {MODEL_FILE.name}")

    # ── Enregistrement dans le Model Registry MLflow ─────────────────────
    section("Enregistrement dans le Model Registry MLflow")
    register_best_model(best_pipeline, best_name, run_ids[best_name], imp_df)

    section("PHASE 4 TERMINÉE")
    print(f"\n  Classement final :")
    for _, row in results_df.iterrows():
        print(f"  {row['Modele']:<35} MAE={row['MAE']:.2f}%  R²={row['R2']:.4f}")
    print(f"\n  Dashboard MLflow : http://localhost:5000")


if __name__ == "__main__":
    main()
