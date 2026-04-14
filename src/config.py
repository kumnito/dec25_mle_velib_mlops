"""
config.py — Constantes partagées entre toutes les phases du pipeline Vélib'.

Ce module centralise :
  - les chemins des fichiers de données
  - la liste ordonnée des features
  - les noms des colonnes cibles
  - les paramètres globaux du projet

Tous les scripts src/*.py importent depuis ce module pour garantir
la cohérence (un seul endroit à modifier si un nom change).
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS — lus depuis les variables d'environnement ou valeurs par défaut
# ─────────────────────────────────────────────────────────────────────────────

# Répertoire racine des données (injecté via .env → docker-compose)
RAW_DIR       = Path(os.getenv("DATA_RAW_DIR",       "/app/data/raw"))
PROCESSED_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "/app/data/processed"))
OUTPUTS_DIR   = Path(os.getenv("DATA_OUTPUTS_DIR",   "/app/data/outputs"))
PLOTS_DIR     = OUTPUTS_DIR / "plots"

# Création des dossiers si absents (utile lors du premier lancement)
for _dir in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Fichiers d'entrée ──────────────────────────────────────────────────────
RAW_FILE      = RAW_DIR / os.getenv("RAW_FILENAME", "dataset_velib_300326.csv")

# ── Fichiers produits par 02_preprocessing ────────────────────────────────
TRAIN_FILE    = PROCESSED_DIR / "train_preprocessed.csv"
TEST_FILE     = PROCESSED_DIR / "test_preprocessed.csv"
STATIONS_FILE = PROCESSED_DIR / "station_names.csv"

# ── Fichiers produits par 03_hyperparameter ───────────────────────────────
BEST_PARAMS_FILE = OUTPUTS_DIR / "best_params.json"

# ── Fichiers produits par 04_training ────────────────────────────────────
RESULTS_FILE    = OUTPUTS_DIR / "resultats_modeles_v2.csv"
IMPORTANCE_FILE = OUTPUTS_DIR / "feature_importance.csv"
MODEL_FILE      = OUTPUTS_DIR / "best_model.joblib"

# ── Fichiers produits par 05_predictions ─────────────────────────────────
PREDICTIONS_FILE = OUTPUTS_DIR / "test_avec_predictions.csv"

# ── Fichiers produits par 06_bilan ────────────────────────────────────────
BILAN_FILE = OUTPUTS_DIR / "bilan_final.txt"

# ─────────────────────────────────────────────────────────────────────────────
# COLONNES CIBLES
# ─────────────────────────────────────────────────────────────────────────────

# Taux de remplissage brut calculé (0-100%)
TARGET_RAW = "capacity_status_calc"

# Cible du modèle : résidu = taux_réel - station_trend_avg
TARGET = "residual"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES — 24 variables d'entrée des modèles ML
# L'ordre est fixé ici et utilisé partout (train, test, CV, prédiction)
# ─────────────────────────────────────────────────────────────────────────────

FEATURES_FINAL = [
    # Comportement historique de la station (feature la plus importante)
    "station_trend_avg",

    # Temporel brut
    "hour",
    "day_of_week",
    "month",

    # Encodage cyclique : garantit la continuité 23h→0h et dim→lun
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",

    # Calendrier
    "is_holiday",
    "is_vacation",

    # Météo
    "apparent_temperature",
    "weather_severity",
    "is_frozen",
    "is_stormy",

    # Caractéristiques de la station
    "capacity",
    "capacity_group",

    # Série temporelle (lags)
    "lag_60min",
    "lag_240min",
    "lag_res_240min",

    # Features enrichies
    "temp_anomalie",
    "is_peak_hour",
    "is_friday_evening",
    "is_monday_morning",
    "morning_evening_ratio",
]

# Features utilisées pendant la recherche d'hyperparamètres
# (station_trend_avg et lag_res_240min exclus → calculés sur le train, fuite si gardés)
FEATURES_SEARCH = [
    f for f in FEATURES_FINAL
    if f not in ["station_trend_avg", "lag_res_240min"]
]

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES GLOBAUX
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED      = int(os.getenv("RANDOM_SEED", 42))
TRAIN_RATIO      = 0.80        # 80% train / 20% test (split temporel)

# Hyperparameter tuning
OPTUNA_N_TRIALS  = int(os.getenv("OPTUNA_N_TRIALS", 50))
TUNING_SAMPLE_FRAC = float(os.getenv("TUNING_SAMPLE_FRAC", 0.5))

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES D'AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    """Affiche un séparateur de section dans les logs console."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def ok(msg: str) -> None:
    """Affiche un message de succès."""
    print(f"  ✓ {msg}")


def info(msg: str) -> None:
    """Affiche une information."""
    print(f"  · {msg}")
