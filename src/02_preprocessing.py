"""
02_preprocessing.py — Phase 2 : Preprocessing & Feature Engineering
=====================================================================
Transforme les données brutes en dataset propre et enrichi pour la modélisation.

Entrées  : data/raw/dataset_velib_300326.csv
Sorties  : data/processed/train_preprocessed.csv
           data/processed/test_preprocessed.csv
           data/processed/station_names.csv

Ce que prédit le modèle :
    résidu = taux_réel - station_trend_avg
    → forcer le modèle à apprendre les ÉCARTS à la normale,
      pas le comportement moyen qu'il pourrait simplement mémoriser.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from config import (
    RAW_FILE, PROCESSED_DIR,
    TRAIN_FILE, TEST_FILE, STATIONS_FILE,
    TARGET_RAW, TARGET, TRAIN_RATIO, RANDOM_SEED,
    section, ok, info,
)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path) -> pd.DataFrame:
    """Charge le dataset brut et convertit les types."""
    df = pd.read_csv(path, parse_dates=["datetime"])

    # Conversion des booléens stockés en 't'/'f' (format PostgreSQL)
    for col in ["is_renting", "is_holiday", "is_vacation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"t": True, "f": False})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — Calcul du taux et nettoyage
# ─────────────────────────────────────────────────────────────────────────────

def compute_fill_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le taux de remplissage et supprime les lignes invalides.

    Logique :
        total_bikes    = bikes_mechanical + bikes_ebike
        total_capacity = total_bikes + numdocksavailable
        taux           = total_bikes / total_capacity × 100

    Lignes supprimées :
        - capacité totale nulle (station vide ET aucun dock libre)
        - station hors service (is_renting = False)
    """
    df = df.copy()
    n_total = len(df)

    df["total_bikes"]    = df["bikes_mechanical"] + df["bikes_ebike"]
    df["total_capacity"] = df["total_bikes"] + df["numdocksavailable"]

    # Suppression des capacités nulles
    n_cap_nulle = (df["total_capacity"] == 0).sum()
    df = df[df["total_capacity"] > 0].copy()

    # Filtre is_renting — accepte bool natif, entier et texte
    n_hors_service = 0
    if "is_renting" in df.columns:
        mask = (
            df["is_renting"].eq(True) | df["is_renting"].eq(1) |
            df["is_renting"].astype(str).str.lower().isin(["true", "t", "1"])
        )
        n_hors_service = (~mask).sum()
        df = df[mask].copy()

    df[TARGET_RAW] = (df["total_bikes"] / df["total_capacity"] * 100).round(2)

    info(f"Lignes conservées : {len(df):,} / {n_total:,}")
    info(f"  Capacité nulle supprimée  : {n_cap_nulle:,}")
    info(f"  Hors service supprimées   : {n_hors_service:,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — Features temporelles
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les features temporelles brutes et leur encodage cyclique.

    Encodage cyclique — pourquoi ?
        Pour un modèle, hour=23 et hour=0 semblent distants de 23 unités.
        En projetant sur un cercle (sin/cos), 23h et 0h deviennent voisins.
        Même logique pour dimanche→lundi.
    """
    df = df.copy()

    # Temporel brut
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=lundi, 6=dimanche
    df["month"]       = df["datetime"].dt.month

    # Encodage cyclique de l'heure (période = 24h)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).round(4)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).round(4)

    # Encodage cyclique du jour de la semaine (période = 7 jours)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).round(4)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).round(4)

    # Features d'interaction temporelle
    # heure de pointe : 7h-9h (matin) ou 17h-19h (soir)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Vendredi soir : comportement distinct (sorties, loisirs)
    df["is_friday_evening"] = ((df["day_of_week"] == 4) & (df["hour"] >= 17)).astype(int)

    # Lundi matin : reprise des déplacements domicile-travail
    df["is_monday_morning"] = ((df["day_of_week"] == 0) & (df["hour"] < 10)).astype(int)

    ok("Features temporelles créées")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — Profil fonctionnel des stations
# ─────────────────────────────────────────────────────────────────────────────

def add_station_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le profil de taille des stations.

    capacity_group :
        Catégorie de taille : 0=petite (≤20), 1=moyenne (21-35),
                              2=grande (36-50), 3=très grande (>50)

    Note : morning_evening_ratio est calculé après le split sur le train
    uniquement (voir add_post_split_features) pour éviter toute fuite.
    """
    df = df.copy()

    df["capacity_group"] = pd.cut(
        df["capacity"],
        bins=[0, 20, 35, 50, 999],
        labels=[0, 1, 2, 3],
    ).astype(int)

    ok("Profils stations — capacity_group créé")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — Features météo
# ─────────────────────────────────────────────────────────────────────────────

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les features météo à partir du code WMO :

    weather_severity (0-4) :
        0 = Clair / 1 = Nuageux / 2 = Pluie légère
        3 = Pluie forte / 4 = Conditions extrêmes (orage, neige, verglas)

    is_frozen  : neige, verglas, grêle → forte réduction d'usage prévisible
    is_stormy  : orage, vent violent   → forte réduction d'usage prévisible
    temp_anomalie : écart à la normale mensuelle
        5°C en janvier = normal  → peu d'impact
        5°C en mai = choc froid  → impact comportemental fort
    """
    df = df.copy()

    # weather_severity
    df["weather_severity"] = 0
    df.loc[(df["weather_code"] >= 4) & (df["weather_code"] <= 19), "weather_severity"] = 1
    m2 = ((df["weather_code"] >= 20) & (df["weather_code"] <= 29)) | \
         ((df["weather_code"] >= 40) & (df["weather_code"] <= 59))
    df.loc[m2, "weather_severity"] = 2
    m3 = ((df["weather_code"] >= 60) & (df["weather_code"] <= 69)) | \
         ((df["weather_code"] >= 80) & (df["weather_code"] <= 84))
    df.loc[m3, "weather_severity"] = 3
    m4 = ((df["weather_code"] >= 30) & (df["weather_code"] <= 39)) | \
         ((df["weather_code"] >= 70) & (df["weather_code"] <= 79)) | \
          (df["weather_code"] >= 85)
    df.loc[m4, "weather_severity"] = 4

    # Conditions gelées (neige, verglas, grêle) — aligné notebook
    df["is_frozen"] = df["weather_code"].isin(
        [56, 57, 66, 67] + list(range(70, 80)) + [85, 86]
    ).astype(int)

    # Conditions orageuses — aligné notebook
    df["is_stormy"] = df["weather_code"].isin(
        [17, 18, 19, 29] + list(range(90, 100))
    ).astype(int)

    # Anomalie thermique — calculée après le split sur le train uniquement
    # (voir add_post_split_features)

    ok("Features météo créées")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — Encodage des variables calendaires
# ─────────────────────────────────────────────────────────────────────────────

def encode_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit is_holiday et is_vacation en entier 0/1."""
    df = df.copy()
    for col in ["is_holiday", "is_vacation"]:
        if col in df.columns:
            df[col] = (
                df[col].eq(True) | df[col].eq(1) |
                df[col].astype(str).str.lower().isin(["true", "t", "1"])
            ).astype(int)
    ok("Variables calendaires encodées")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 — Lags temporels avec merge_asof
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée deux lags temporels :
        lag_60min  : taux de la station 1h avant
        lag_240min : taux de la station 4h avant

    Pourquoi merge_asof plutôt que shift(N) ?
        shift(N) décale de N LIGNES, pas de N minutes.
        Avec des intervalles de collecte variables (médiane ≈ 69 min, max > 12h),
        shift(8) peut représenter 96 min ou 6h selon les jours.
        merge_asof cherche le relevé le plus proche d'une DURÉE fixe dans le passé,
        garantissant un lag temporel réel.
    """
    df = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    def lag_temporel(minutes: int, col_name: str) -> pd.Series:
        """
        Pour chaque relevé, trouve le taux le plus proche
        de `minutes` minutes dans le passé, par station.

        Correction bug : on préserve l'index original du df via reset_index()
        avant le merge_asof, puis on le restaure avec set_index().
        Sans ça, merge_asof réinitialise l'index à 0..n pour chaque groupe
        → pd.concat produit des index dupliqués → reindex échoue.
        """
        lag_time = pd.Timedelta(minutes=minutes)
        results  = []

        for _, group in df.groupby("station_id", sort=False):
            group = group.sort_values("datetime")
            ref   = group.copy()
            ref["datetime_lag"] = ref["datetime"] + lag_time

            # reset_index() : sauvegarde l'index original dans une colonne "_idx"
            merged = pd.merge_asof(
                group[["datetime", TARGET_RAW]].reset_index().rename(
                    columns={"index": "_idx"}
                ),
                ref[["datetime_lag", TARGET_RAW]].rename(
                    columns={"datetime_lag": "datetime", TARGET_RAW: col_name}
                ),
                on="datetime",
                direction="forward",
                tolerance=pd.Timedelta(minutes=30),
            )
            # Restaure l'index original avant d'ajouter aux résultats
            merged = merged.set_index("_idx")
            results.append(merged[col_name])

        return pd.concat(results).reindex(df.index)

    df["lag_60min"]  = lag_temporel(60,  "lag_60min")
    df["lag_240min"] = lag_temporel(240, "lag_240min")

    # Suppression des lignes sans historique suffisant
    n_avant = len(df)
    df = df.dropna(subset=["lag_60min", "lag_240min"])
    ok(f"Lags créés — lignes supprimées (historique insuffisant) : {n_avant - len(df):,}")
    ok(f"Lignes restantes : {len(df):,}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 — Split temporel train / test
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split 80/20 basé sur le temps (pas aléatoire).

    Méthode : quantile(0.80) sur la colonne datetime.
    Alignée avec le notebook 2_preprocessing.ipynb pour produire
    des résultats identiques sur les mêmes données.

    Pourquoi temporel ?
        Un split aléatoire permettrait au modèle de s'entraîner sur
        des données "du futur". En production, on prédit toujours à partir
        du passé — le split temporel simule cette condition réelle.
    """
    cutoff = df["datetime"].quantile(0.80)

    train = df[df["datetime"] <= cutoff].copy()
    test  = df[df["datetime"] >  cutoff].copy()

    info(f"Coupure temporelle : {cutoff}")
    ok(f"Train : {len(train):,} lignes ({len(train)/len(df)*100:.0f}%)")
    ok(f"Test  : {len(test):,}  lignes ({len(test)/len(df)*100:.0f}%)")

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 9 — station_trend_avg (calculé sur le TRAIN uniquement)
# ─────────────────────────────────────────────────────────────────────────────

def add_station_trend(train: pd.DataFrame,
                      test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pour chaque combinaison (station × heure × jour × mois), calcule la
    moyenne historique du taux — UNIQUEMENT sur le train pour éviter la fuite.

    Fallback à 2 niveaux — aligné avec le notebook 2_preprocessing.ipynb :
        Niveau 1 : station × heure × jour × mois  (≥ 3 obs)
        Niveau 2 : station × heure × jour          (si niveau 1 inconnu)
        Niveau 3 : moyenne globale du train        (dernier recours)

    Corrélation typique station_trend_avg / taux_réel ≈ 0.87
    → La feature la plus importante du modèle.
    """
    GROUPBY = ["station_id", "day_of_week", "hour", "month"]
    global_mean = train[TARGET_RAW].mean()

    # ── Niveau 1 : station × heure × jour × mois (≥ 3 obs) ───────────────
    s_mean  = train.groupby(GROUPBY)[TARGET_RAW].mean().round(2)
    s_count = train.groupby(GROUPBY)[TARGET_RAW].count()

    trend_raw = s_mean.to_frame("trend_mean").join(
        s_count.to_frame("trend_count")
    ).reset_index()

    trend_raw["station_trend_avg"] = np.where(
        trend_raw["trend_count"] >= 3,
        trend_raw["trend_mean"],
        np.nan,
    )
    trend_fin = trend_raw[GROUPBY + ["station_trend_avg"]]

    n_ignored = (trend_raw["trend_count"] < 3).sum()
    info(f"Combinaisons avec < 3 obs ignorées : {n_ignored:,}")
    info(f"Combinaisons valides (niveau 1) : {(trend_raw['trend_count'] >= 3).sum():,}")

    # ── Niveau 2 : station × heure × jour (fallback) ─────────────────────
    s_fallback = (
        train.groupby(["station_id", "day_of_week", "hour"])[TARGET_RAW]
        .mean().round(2)
        .to_frame("station_trend_fallback").reset_index()
    )

    # ── Application sur train et test ─────────────────────────────────────
    def apply_trend(df_):
        df_ = df_.merge(trend_fin, on=GROUPBY, how="left")
        df_ = df_.merge(s_fallback, on=["station_id", "day_of_week", "hour"], how="left")
        df_["station_trend_avg"] = (
            df_["station_trend_avg"]
            .fillna(df_["station_trend_fallback"])
            .fillna(global_mean)
        )
        return df_.drop(columns=["station_trend_fallback"])

    train = apply_trend(train)
    test  = apply_trend(test)

    corr = train["station_trend_avg"].corr(train[TARGET_RAW])
    ok(f"station_trend_avg créé — corrélation avec taux réel : {corr:.2f}")

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 10 — lag_res_240min et cible résiduelle
# ─────────────────────────────────────────────────────────────────────────────

def add_residual_features(train: pd.DataFrame,
                          test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    lag_res_240min = lag_240min - station_trend_avg
    Lecture : "La station était X points au-dessus/en-dessous de sa normale 4h avant."

    Cible (résidu) = taux_réel - station_trend_avg
    En production : taux_prédit = station_trend_avg + résidu_prédit

    Intérêt : réduit la variance de la cible d'un facteur ≈ 2.7
    (σ du taux brut ≈ 29% → σ du résidu ≈ 11%)
    Ce problème plus ciblé est plus facile à apprendre pour le modèle.
    """
    for df_ in [train, test]:
        df_["lag_res_240min"] = (df_["lag_240min"] - df_["station_trend_avg"]).round(2)
        df_[TARGET]           = (df_[TARGET_RAW]   - df_["station_trend_avg"]).round(2)

    ok(f"lag_res_240min créé")
    ok(f"Cible résiduelle '{TARGET}' créée")
    info(f"  σ taux brut : {train[TARGET_RAW].std():.2f}%")
    info(f"  σ résidu    : {train[TARGET].std():.2f}%")
    info(f"  Réduction   : ×{train[TARGET_RAW].std() / train[TARGET].std():.1f}")

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 9bis — Features calculées après split (sans fuite)
# ─────────────────────────────────────────────────────────────────────────────

def add_post_split_features(train: pd.DataFrame,
                             test:  pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcule morning_evening_ratio et temp_anomalie UNIQUEMENT sur le train,
    puis applique au test — élimine les fuites de ces deux features.

    morning_evening_ratio :
        ratio taux_moyen_matin / taux_moyen_soir par station.
        Calculé sur le train → appliqué au test via map(station_id).
        Fallback : 1.0 (profil mixte) pour les stations absentes du train.

    temp_anomalie :
        écart à la normale mensuelle de température.
        Normale calculée sur le train → appliquée au test via map(month).
    """
    train = train.copy()
    test  = test.copy()

    # ── morning_evening_ratio ─────────────────────────────────────────────
    morning_avg   = train[train["hour"].isin([7, 8, 9])].groupby("station_id")[TARGET_RAW].mean()
    evening_avg   = train[train["hour"].isin([17, 18, 19])].groupby("station_id")[TARGET_RAW].mean()
    station_ratio = (morning_avg / evening_avg.replace(0, np.nan)).dropna().round(3)

    for df_ in [train, test]:
        df_["morning_evening_ratio"] = (
            df_["station_id"].map(station_ratio).fillna(1.0).round(3)
        )

    n_res = int((station_ratio > 1.2).sum())
    n_bur = int((station_ratio < 0.8).sum())
    n_mix = int(((station_ratio >= 0.8) & (station_ratio <= 1.2)).sum())
    ok(f"morning_evening_ratio — Résidentiel:{n_res}  Bureaux:{n_bur}  Mixte:{n_mix}")

    # ── temp_anomalie ─────────────────────────────────────────────────────
    monthly_mean = train.groupby("month")["apparent_temperature"].mean()

    for df_ in [train, test]:
        df_["temp_anomalie"] = (
            df_["apparent_temperature"] - df_["month"].map(monthly_mean)
        ).round(2)

    ok(f"temp_anomalie — σ train : {train['temp_anomalie'].std():.2f}°C")

    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 11 — Export
# ─────────────────────────────────────────────────────────────────────────────

def export_datasets(df_raw: pd.DataFrame,
                    train: pd.DataFrame,
                    test: pd.DataFrame) -> None:
    """Sauvegarde les datasets préparés et la table de correspondance stations."""

    # Arrondi final
    train = train.round(2)
    test  = test.round(2)

    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE,   index=False)
    ok(f"Exporté : {TRAIN_FILE.name}  ({train.shape[0]:,} × {train.shape[1]})")
    ok(f"Exporté : {TEST_FILE.name}   ({test.shape[0]:,} × {test.shape[1]})")

    # Table de correspondance station_id ↔ nom + GPS (pour Streamlit)
    geo_cols = ["station_id", "name"]
    for lat_col in ["lat", "latitude", "Lat"]:
        if lat_col in df_raw.columns:
            geo_cols.append(lat_col)
            break
    for lon_col in ["lon", "lng", "longitude", "Lon"]:
        if lon_col in df_raw.columns:
            geo_cols.append(lon_col)
            break

    stations_geo = df_raw[geo_cols].drop_duplicates("station_id")
    stations_geo.to_csv(STATIONS_FILE, index=False)
    ok(f"Exporté : {STATIONS_FILE.name}  ({len(stations_geo)} stations)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    section("PHASE 2 — PREPROCESSING & FEATURE ENGINEERING")

    # Étape 1 — Chargement
    section("Étape 1 — Chargement")
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Dataset introuvable : {RAW_FILE}")
    df = load_raw(RAW_FILE)
    ok(f"Dataset chargé : {len(df):,} lignes × {df.shape[1]} colonnes")

    # Étape 2 — Taux de remplissage
    section("Étape 2 — Calcul du taux de remplissage")
    df = compute_fill_rate(df)

    # Étape 3 — Features temporelles
    section("Étape 3 — Features temporelles")
    df = add_temporal_features(df)

    # Étape 4 — Profil des stations
    section("Étape 4 — Profil fonctionnel des stations")
    df = add_station_profiles(df)

    # Étape 5 — Météo
    section("Étape 5 — Features météo")
    df = add_weather_features(df)

    # Étape 6 — Calendrier
    section("Étape 6 — Encodage calendaire")
    df = encode_calendar(df)

    # Étape 7 — Lags temporels
    section("Étape 7 — Lags temporels (merge_asof)")
    df = add_temporal_lags(df)

    # Étape 8 — Split temporel
    section("Étape 8 — Split temporel 80/20")
    train, test = temporal_split(df)

    # Étape 9bis — Features post-split (sans fuite)
    section("Étape 9bis — morning_evening_ratio & temp_anomalie (sur train uniquement)")
    train, test = add_post_split_features(train, test)

    # Étape 9 — station_trend_avg
    section("Étape 9 — Comportement historique (station_trend_avg)")
    train, test = add_station_trend(train, test)

    # Étape 10 — Résidus
    section("Étape 10 — Cible résiduelle")
    train, test = add_residual_features(train, test)

    # Étape 11 — Export
    section("Étape 11 — Export")
    export_datasets(load_raw(RAW_FILE), train, test)

    section("PHASE 2 TERMINÉE")
    info(f"Fichiers dans : {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
