"""
01_dataviz.py — Phase 1 : Exploration & DataViz
================================================
Analyse exploratoire du dataset brut Vélib'.

Entrées  : data/raw/dataset_velib_300326.csv
Sorties  : data/outputs/plots/*.png  (graphiques sauvegardés)

Ce script ne modifie pas les données — il produit uniquement
des visualisations pour comprendre le dataset avant tout traitement.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import (
    RAW_FILE, PLOTS_DIR, TARGET_RAW,
    section, ok, info,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION MATPLOTLIB
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":       110,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "savefig.bbox":     "tight",
    "savefig.dpi":      110,
})


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, name: str) -> None:
    """Sauvegarde une figure dans PLOTS_DIR et ferme pour libérer la mémoire."""
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    ok(f"Plot sauvegardé → {path.name}")


def apply_weather_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traduit le code météo WMO brut en niveau de sévérité (0-4).

    Référentiel WMO :
        0  → Clair/dégagé
        1  → Nuageux/brumeux
        2  → Précipitations récentes / brouillard
        3  → Pluie modérée / averse
        4  → Orage / neige / conditions extrêmes
    """
    df = df.copy()
    df["weather_severity"] = 0
    df.loc[(df["weather_code"] >= 4)  & (df["weather_code"] <= 19), "weather_severity"] = 1
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
    return df


def prepare_for_viz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcul minimal du taux de remplissage et des features temporelles/météo
    pour la DataViz (sans feature engineering complet du preprocessing).
    """
    df = df.copy()

    # Conversion des booléens stockés comme 't'/'f'
    for col in ["is_renting", "is_holiday", "is_vacation"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"t": True, "f": False})

    # Calcul du taux de remplissage
    df["total_bikes"]    = df["bikes_mechanical"] + df["bikes_ebike"]
    df["total_capacity"] = df["total_bikes"] + df["numdocksavailable"]

    # Suppression des capacités nulles (stations hors service ou données corrompues)
    df = df[df["total_capacity"] > 0].copy()

    # Filtre is_renting : garde uniquement les stations en service
    if "is_renting" in df.columns:
        mask = (
            df["is_renting"].eq(True) | df["is_renting"].eq(1) |
            df["is_renting"].astype(str).str.lower().isin(["true", "t", "1"])
        )
        df = df[mask].copy()

    df[TARGET_RAW] = (df["total_bikes"] / df["total_capacity"] * 100).round(2)

    # Features temporelles
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"]       = df["datetime"].dt.month

    # Météo
    df = apply_weather_severity(df)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────

def plot_temporal_coverage(df_raw: pd.DataFrame) -> None:
    """
    Graphique 1 — Répartition temporelle des relevés.
    Col 1 : relevés par jour calendaire  (détecte les trous de collecte)
    Col 2 : relevés par jour de semaine  (équilibre de la couverture)
    """
    daily   = df_raw.groupby(df_raw["datetime"].dt.date).size()
    dow     = df_raw.groupby(df_raw["datetime"].dt.dayofweek).size()

    n_days  = round((df_raw["datetime"].max() - df_raw["datetime"].min()).days, 1)
    n_weeks = round(n_days / 7, 1)
    DAY_FR  = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5),
                             gridspec_kw={"width_ratios": [2.5, 1]})

    # ── Col 1 : relevés par jour calendaire ───────────────────────────────
    axes[0].fill_between(range(len(daily)), daily.values, alpha=0.4, color="steelblue")
    axes[0].plot(range(len(daily)), daily.values, color="steelblue", linewidth=1)
    axes[0].set_title(f"Relevés par jour calendaire — {n_days} jours ({n_weeks} semaines)")
    axes[0].set_xlabel("Jours depuis le début de la collecte")
    axes[0].set_ylabel("Nombre de relevés")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ── Col 2 : relevés par jour de semaine ───────────────────────────────
    colors = ["steelblue"] * 5 + ["darkorange"] * 2   # weekend en orange
    bars   = axes[1].bar(DAY_FR, dow.values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, dow.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + dow.values.max() * 0.01,
                     f"{val:,}", ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Relevés par jour de semaine")
    axes[1].set_ylabel("Nombre de relevés")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    save(fig, "01_temporal_coverage")


def plot_fill_rate_distribution(df: pd.DataFrame) -> None:
    """
    Graphique 2 — Distribution du taux de remplissage.
    Montre la forme globale de la target avant toute modélisation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histogramme
    axes[0].hist(df[TARGET_RAW], bins=50, color="steelblue",
                 edgecolor="white", linewidth=0.3)
    axes[0].axvline(df[TARGET_RAW].mean(),   color="crimson",  linestyle="--",
                    label=f"Moyenne : {df[TARGET_RAW].mean():.1f}%")
    axes[0].axvline(df[TARGET_RAW].median(), color="darkorange", linestyle=":",
                    label=f"Médiane : {df[TARGET_RAW].median():.1f}%")
    axes[0].set_title("Distribution du taux de remplissage")
    axes[0].set_xlabel("Taux (%)")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()

    # Boîte à moustaches par station (échantillon)
    sample_stations = df["station_id"].value_counts().head(20).index
    sample = df[df["station_id"].isin(sample_stations)]
    axes[1].boxplot(
        [sample[sample["station_id"] == s][TARGET_RAW].values for s in sample_stations],
        patch_artist=True,
        medianprops=dict(color="crimson", linewidth=2),
    )
    axes[1].set_title("Distribution par station (top 20)")
    axes[1].set_xlabel("Station (index)")
    axes[1].set_ylabel("Taux (%)")

    save(fig, "02_fill_rate_distribution")


def plot_temporal_patterns(df: pd.DataFrame) -> None:
    """
    Graphique 3 — Profils temporels : taux moyen par heure et par jour.
    Met en évidence les heures de pointe et les différences weekend/semaine.
    """
    DAY_FR  = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    hourly  = df.groupby("hour")[TARGET_RAW].mean()
    daily   = df.groupby("day_of_week")[TARGET_RAW].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Taux par heure
    axes[0].plot(hourly.index, hourly.values, color="steelblue",
                 linewidth=2, marker="o", markersize=4)
    # Zones de pointe
    for h_start, h_end, label in [(7, 9, "Pointe matin"), (17, 19, "Pointe soir")]:
        axes[0].axvspan(h_start, h_end, alpha=0.12, color="green", label=label)
    axes[0].set_title("Taux moyen par heure")
    axes[0].set_xlabel("Heure")
    axes[0].set_ylabel("Taux moyen (%)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(fontsize=9)

    # Taux par jour
    colors = ["steelblue"] * 5 + ["darkorange"] * 2   # weekend en orange
    bars = axes[1].bar(DAY_FR, daily.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, daily.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Taux moyen par jour de la semaine")
    axes[1].set_ylabel("Taux moyen (%)")

    save(fig, "03_temporal_patterns")


def plot_weather_impact(df: pd.DataFrame) -> None:
    """
    Graphique 4 — Impact de la météo sur le taux de remplissage.
    Compare la répartition des conditions et leur effet sur l'usage.
    """
    SEV_LABELS = ["0 Clair", "1 Nuageux", "2 Pluie légère",
                  "3 Pluie forte", "4 Extrême"]
    SEV_COLORS = ["#f39c12", "#95a5a6", "#3498db", "#2980b9", "#8e44ad"]

    sev_counts  = df["weather_severity"].value_counts().sort_index()
    mean_by_sev = df.groupby("weather_severity")[TARGET_RAW].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Répartition des conditions météo
    idx = sev_counts.index.tolist()
    axes[0].bar([SEV_LABELS[i] for i in idx], sev_counts.values,
                color=[SEV_COLORS[i] for i in idx], edgecolor="white")
    axes[0].set_title("Répartition des conditions météo")
    axes[0].set_ylabel("Nombre de relevés")
    axes[0].tick_params(axis="x", rotation=20)

    # Taux moyen par condition
    idx2 = mean_by_sev.index.tolist()
    bars = axes[1].bar([SEV_LABELS[i] for i in idx2], mean_by_sev.values,
                       color=[SEV_COLORS[i] for i in idx2], edgecolor="white")
    for bar, val in zip(bars, mean_by_sev.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Taux moyen par condition météo")
    axes[1].set_ylabel("Taux moyen (%)")
    axes[1].tick_params(axis="x", rotation=20)

    save(fig, "04_weather_impact")


def plot_temp_anomaly(df: pd.DataFrame) -> None:
    """
    Graphique 5 — Anomalie thermique.
    5°C en janvier = normal. 5°C en mai = choc froid → impact sur l'usage.
    L'anomalie = température ressentie - moyenne mensuelle.
    """
    monthly_mean    = df.groupby("month")["apparent_temperature"].transform("mean")
    df              = df.copy()
    df["temp_anom"] = df["apparent_temperature"] - monthly_mean

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Distribution de l'anomalie thermique
    axes[0].hist(df["temp_anom"], bins=60, color="steelblue",
                 edgecolor="white", linewidth=0.3)
    axes[0].axvline(0, color="crimson", linestyle="--", label="0 = normale du mois")
    axes[0].set_title("Distribution de l'anomalie thermique")
    axes[0].set_xlabel("Écart à la normale (°C)")
    axes[0].set_ylabel("Fréquence")
    axes[0].legend()

    # Corrélation anomalie thermique / taux de remplissage (échantillon)
    sample = df.sample(min(8000, len(df)), random_state=42)
    axes[1].scatter(sample["temp_anom"], sample[TARGET_RAW],
                    alpha=0.05, s=3, color="steelblue")
    axes[1].set_title("Anomalie thermique vs taux de remplissage")
    axes[1].set_xlabel("Anomalie (°C)")
    axes[1].set_ylabel("Taux (%)")

    save(fig, "05_temp_anomaly")


def plot_station_profiles(df: pd.DataFrame) -> None:
    """
    Graphique 6 — Profil fonctionnel des stations (morning_evening_ratio).
    Résidentiel : se vide le matin (ratio > 1.2)
    Bureaux     : se remplit le matin (ratio < 0.8)
    Mixte       : équilibré
    """
    morning_avg   = df[df["hour"].isin([7, 8, 9])].groupby("station_id")[TARGET_RAW].mean()
    evening_avg   = df[df["hour"].isin([17, 18, 19])].groupby("station_id")[TARGET_RAW].mean()
    station_ratio = (morning_avg / evening_avg.replace(0, np.nan)).dropna()

    n_res = int((station_ratio > 1.2).sum())
    n_bur = int((station_ratio < 0.8).sum())
    n_mix = int(((station_ratio >= 0.8) & (station_ratio <= 1.2)).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Distribution du ratio
    axes[0].hist(station_ratio.values, bins=40, color="steelblue",
                 edgecolor="white", linewidth=0.3)
    axes[0].axvline(1.2, color="#2ecc71",  linestyle="--", label=f"Résidentiel >1.2 ({n_res})")
    axes[0].axvline(0.8, color="#e74c3c",  linestyle="--", label=f"Bureaux <0.8 ({n_bur})")
    axes[0].axvline(1.0, color="gray",     linestyle=":",  label=f"Mixte ({n_mix})")
    axes[0].set_title("Distribution du ratio matin/soir par station")
    axes[0].set_xlabel("Ratio matin / soir")
    axes[0].set_ylabel("Nb stations")
    axes[0].legend(fontsize=9)

    # Camembert des profils
    axes[1].pie(
        [n_res, n_mix, n_bur],
        labels=[f"Résidentiel\n{n_res}", f"Mixte\n{n_mix}", f"Bureaux\n{n_bur}"],
        colors=["#2ecc71", "#3498db", "#e67e22"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[1].set_title("Répartition des profils fonctionnels")

    save(fig, "06_station_profiles")


def plot_vacation_impact(df: pd.DataFrame) -> None:
    """
    Graphique 7 — Impact des vacances scolaires sur le taux de remplissage.
    """
    vac_int = (
        df["is_vacation"].eq(True) | df["is_vacation"].eq(1) |
        df["is_vacation"].astype(str).str.lower().isin(["true", "t", "1"])
    ).astype(int)

    vac_mean = df.groupby(vac_int)[TARGET_RAW].mean()

    if len(vac_mean) < 2:
        info("Pas assez de données vacances/hors-vacances pour le graphique 7")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Hors vacances", "Vacances scolaires"],
                  vac_mean.values.round(1),
                  color=["steelblue", "darkorange"],
                  width=0.4, edgecolor="white")
    for bar, val in zip(bars, vac_mean.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_title("Impact des vacances scolaires sur le taux de remplissage")
    ax.set_ylabel("Taux moyen (%)")
    save(fig, "07_vacation_impact")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    section("PHASE 1 — DATAVIZ")

    # ── Chargement ────────────────────────────────────────────────────────
    info(f"Chargement : {RAW_FILE}")
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {RAW_FILE}\n"
            f"Placer le fichier dans data/raw/"
        )

    df_raw = pd.read_csv(RAW_FILE, parse_dates=["datetime"])
    ok(f"Dataset chargé : {len(df_raw):,} lignes × {df_raw.shape[1]} colonnes")

    # ── Préparation pour la visualisation ─────────────────────────────────
    df = prepare_for_viz(df_raw)
    ok(f"Lignes après filtrage : {len(df):,}")
    info(f"Taux moyen   : {df[TARGET_RAW].mean():.1f}%")
    info(f"Taux médian  : {df[TARGET_RAW].median():.1f}%")
    info(f"Nb stations  : {df['station_id'].nunique():,}")

    # ── Génération des graphiques ─────────────────────────────────────────
    section("Génération des graphiques")

    plot_temporal_coverage(df_raw)
    plot_fill_rate_distribution(df)
    plot_temporal_patterns(df)
    plot_weather_impact(df)
    plot_temp_anomaly(df)
    plot_station_profiles(df)
    plot_vacation_impact(df)

    section("PHASE 1 TERMINÉE")
    info(f"7 graphiques sauvegardés dans : {PLOTS_DIR}")


if __name__ == "__main__":
    main()
