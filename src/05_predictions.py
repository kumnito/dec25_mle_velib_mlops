"""
05_predictions.py — Phase 5 : Prédictions sur le test set
==========================================================
Applique le meilleur modèle sauvegardé sur le test set et
reconstruit le taux de remplissage final.

Entrées  : data/processed/test_preprocessed.csv
           data/outputs/best_model.joblib
Sorties  : data/outputs/test_avec_predictions.csv

Colonnes ajoutées au test set :
    residual_pred  → résidu prédit par le modèle
    taux_pred      → taux reconstruit = station_trend_avg + residual_pred (clampé [0,100])
    erreur_absolue → |taux_reel - taux_pred|
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    TEST_FILE, MODEL_FILE, PREDICTIONS_FILE,
    FEATURES_FINAL, TARGET, TARGET_RAW,
    section, ok, info,
)


def main() -> None:
    section("PHASE 5 — PRÉDICTIONS")

    # ── Vérifications préalables ──────────────────────────────────────────
    for path, label in [(TEST_FILE, "test_preprocessed.csv"),
                        (MODEL_FILE, "best_model.joblib")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} non trouvé : {path}\n"
                "Lancer 02_preprocessing.py puis 04_training.py d'abord."
            )

    # ── Chargement ────────────────────────────────────────────────────────
    test_df = pd.read_csv(TEST_FILE)
    ok(f"Test set chargé : {len(test_df):,} lignes")

    model = joblib.load(MODEL_FILE)
    ok(f"Modèle chargé : {MODEL_FILE.name}")

    # Nom du modèle (pour l'affichage)
    model_name = type(model.named_steps["model"]).__name__

    # ── Sélection des features disponibles ───────────────────────────────
    available = [f for f in FEATURES_FINAL if f in test_df.columns]
    missing   = [f for f in FEATURES_FINAL if f not in test_df.columns]
    if missing:
        info(f"Features absentes ignorées : {missing}")

    X_test = test_df[available]

    # ── Prédiction du résidu ──────────────────────────────────────────────
    info(f"Prédiction en cours ({model_name})...")
    residual_pred = model.predict(X_test)

    # ── Reconstruction du taux de remplissage réel ────────────────────────
    # taux_prédit = station_trend_avg + résidu_prédit
    # np.clip garantit que le taux reste dans [0, 100]
    taux_pred = np.clip(
        test_df["station_trend_avg"].values + residual_pred,
        0, 100
    )
    taux_reel = test_df[TARGET_RAW].values

    # ── Calcul des métriques finales ──────────────────────────────────────
    mae_residu  = mean_absolute_error(test_df[TARGET].values, residual_pred)
    rmse_residu = np.sqrt(mean_squared_error(test_df[TARGET].values, residual_pred))
    r2_residu   = r2_score(test_df[TARGET].values, residual_pred)

    mae_taux    = mean_absolute_error(taux_reel, taux_pred)
    rmse_taux   = np.sqrt(mean_squared_error(taux_reel, taux_pred))
    r2_taux     = r2_score(taux_reel, taux_pred)

    baseline_mae = mean_absolute_error(taux_reel, np.full_like(taux_reel, taux_reel.mean()))
    gain_pct     = (baseline_mae - mae_taux) / baseline_mae * 100

    print(f"\n  {'─' * 50}")
    print(f"  MÉTRIQUES FINALES — {model_name}")
    print(f"  {'─' * 50}")
    print(f"  Sur le résidu (cible du modèle) :")
    print(f"    MAE  : {mae_residu:.2f}%")
    print(f"    RMSE : {rmse_residu:.2f}%")
    print(f"    R²   : {r2_residu:.4f}")
    print(f"  Sur le taux reconstitué (0-100%) :")
    print(f"    MAE  : {mae_taux:.2f}%")
    print(f"    RMSE : {rmse_taux:.2f}%")
    print(f"    R²   : {r2_taux:.4f}")
    print(f"  Gain vs baseline (prédire la moyenne) : {gain_pct:.0f}%")
    print(f"  {'─' * 50}")

    # ── Ajout des colonnes de prédiction au test set ──────────────────────
    test_df["residual_pred"]  = residual_pred.round(2)
    test_df["taux_pred"]      = taux_pred.round(2)
    test_df["erreur_absolue"] = np.abs(taux_reel - taux_pred).round(2)

    # ── Export ────────────────────────────────────────────────────────────
    test_df.to_csv(PREDICTIONS_FILE, index=False)
    ok(f"Prédictions exportées → {PREDICTIONS_FILE.name}")
    info(f"  Shape : {test_df.shape}")
    info(f"  Colonnes ajoutées : residual_pred, taux_pred, erreur_absolue")

    # ── Analyse rapide des erreurs par météo ─────────────────────────────
    if "weather_severity" in test_df.columns:
        print(f"\n  MAE par condition météo :")
        SEV_MAP = {0: "Clair", 1: "Nuageux", 2: "Pluie légère",
                   3: "Pluie forte", 4: "Extrême"}
        mae_sev = test_df.groupby("weather_severity")["erreur_absolue"].apply(
            lambda x: round(x.mean(), 2)
        )
        for sev, mae in mae_sev.items():
            label = SEV_MAP.get(sev, str(sev))
            print(f"    {label:<16} : {mae:.2f}%")

    section("PHASE 5 TERMINÉE")


if __name__ == "__main__":
    main()
