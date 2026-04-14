"""
streamlit/app.py — Dashboard Vélib' (version locale Docker)
============================================================
Visualisation complète des 6 phases du pipeline ML.
Lit les fichiers depuis le volume Docker monté en /app/data
(au lieu de HuggingFace dans la version notebook).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — CHEMINS LOCAUX
# ─────────────────────────────────────────────────────────────────────────────

DATA_RAW       = Path(os.getenv("DATA_RAW_DIR",       "/app/data/raw"))
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED_DIR", "/app/data/processed"))
DATA_OUTPUTS   = Path(os.getenv("DATA_OUTPUTS_DIR",   "/app/data/outputs"))

FILES = {
    "raw":        DATA_RAW       / "dataset_velib_300326.csv",
    "train":      DATA_PROCESSED / "train_preprocessed.csv",
    "test":       DATA_PROCESSED / "test_preprocessed.csv",
    "stations":   DATA_PROCESSED / "station_names.csv",
    "resultats":  DATA_OUTPUTS   / "resultats_modeles_v2.csv",
    "importance": DATA_OUTPUTS   / "feature_importance.csv",
    "preds":      DATA_OUTPUTS   / "test_avec_predictions.csv",
    "bilan":      DATA_OUTPUTS   / "bilan_final.txt",
}

TARGET     = "capacity_status_calc"
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Projet Vélib' — ML Pipeline",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary":   "#3498db",
    "secondary": "#2ecc71",
    "warning":   "#e67e22",
    "danger":    "#e74c3c",
    "dark":      "#2c3e50",
    "purple":    "#9b59b6",
}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚲 Projet Vélib'")
    st.markdown("Pipeline ML — Docker")
    st.divider()
    st.subheader("Navigation")
    section = st.radio(
        "Phase",
        [
            "Phase 1 — DataViz",
            "Phase 2 — Preprocessing",
            "Phase 3 — Modélisation",
            "Bilan — R² Reconstruit",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    # Statut des fichiers disponibles
    st.caption("Fichiers disponibles :")
    for label, path in [
        ("Dataset brut",  FILES["raw"]),
        ("Train",         FILES["train"]),
        ("Test",          FILES["test"]),
        ("Résultats",     FILES["resultats"]),
        ("Prédictions",   FILES["preds"]),
    ]:
        icon = "✅" if path.exists() else "❌"
        st.caption(f"{icon} {label}")


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT (mis en cache pour éviter de relire à chaque interaction)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement en cours...")
def load_csv(path: Path, parse_dates=None, optional: bool = False):
    """Charge un CSV depuis le volume local."""
    if not path.exists():
        if not optional:
            st.error(f"Fichier manquant : {path.name} — Lancer le pipeline d'abord.")
        return None
    try:
        return pd.read_csv(str(path), parse_dates=parse_dates or [])
    except Exception as e:
        if not optional:
            st.error(f"Erreur de lecture ({path.name}) : {e}")
        return None


def apply_weather_severity(df: pd.DataFrame) -> pd.DataFrame:
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


@st.cache_data(show_spinner="Préparation des données brutes...")
def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule taux et features temporelles/météo depuis le dataset brut."""
    df = df.copy()
    df["total_bikes"]    = df["bikes_mechanical"] + df["bikes_ebike"]
    df["total_capacity"] = df["total_bikes"] + df["numdocksavailable"]
    df = df[df["total_capacity"] > 0].copy()

    if "is_renting" in df.columns:
        mask = (
            df["is_renting"].eq(True) | df["is_renting"].eq(1) |
            df["is_renting"].astype(str).str.lower().isin(["true", "t", "1"])
        )
        df = df[mask].copy()

    df[TARGET]          = (df["total_bikes"] / df["total_capacity"] * 100).round(2)
    df["hour"]          = df["datetime"].dt.hour
    df["day_of_week"]   = df["datetime"].dt.dayofweek
    df["month"]         = df["datetime"].dt.month
    df                  = apply_weather_severity(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATAVIZ
# ─────────────────────────────────────────────────────────────────────────────

def page_dataviz():
    st.title("Phase 1 — Exploration & DataViz")
    st.markdown("Analyse exploratoire du dataset brut avant toute transformation.")

    df_raw = load_csv(FILES["raw"], parse_dates=["datetime"])
    if df_raw is None:
        st.info("Placer `dataset_velib_300326.csv` dans `data/raw/` pour voir cette phase.")
        return

    df     = prepare_raw(df_raw)
    n_days = (df_raw["datetime"].max() - df_raw["datetime"].min()).days

    # Métriques rapides
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevés totaux", f"{len(df_raw):,}")
    c2.metric("Stations",       f"{df_raw['station_id'].nunique():,}")
    c3.metric("Durée (jours)",  f"{n_days}")
    c4.metric("Taux moyen",     f"{df[TARGET].mean():.1f}%")
    st.divider()

    # Carte des stations
    with st.expander("0. Carte des stations Vélib'", expanded=True):
        lat_col  = next((c for c in ["lat","latitude","Lat"]  if c in df_raw.columns), None)
        lon_col  = next((c for c in ["lon","lng","longitude"] if c in df_raw.columns), None)
        name_col = next((c for c in ["name","nom"]            if c in df_raw.columns), None)

        if lat_col and lon_col:
            morning_avg = df[df["hour"].isin([7,8,9])].groupby("station_id")[TARGET].mean()
            evening_avg = df[df["hour"].isin([17,18,19])].groupby("station_id")[TARGET].mean()
            ratio       = (morning_avg / evening_avg.replace(0, np.nan))

            stations_map = df_raw[["station_id", lat_col, lon_col]].drop_duplicates("station_id").copy()
            if name_col:
                names = df_raw[["station_id", name_col]].drop_duplicates("station_id")
                stations_map = stations_map.merge(names, on="station_id", how="left")
            stations_map["ratio"]  = stations_map["station_id"].map(ratio)
            stations_map["profil"] = stations_map["ratio"].apply(
                lambda r: "Résidentiel" if pd.notna(r) and r > 1.2
                     else ("Bureaux" if pd.notna(r) and r < 0.8 else "Mixte")
            )
            taux_moyen = df.groupby("station_id")[TARGET].mean().reset_index()
            taux_moyen.columns = ["station_id", "taux_moyen"]
            stations_map = stations_map.merge(taux_moyen, on="station_id", how="left")

            PROFIL_COLORS = {"Résidentiel": "#2ecc71", "Mixte": "#3498db", "Bureaux": "#e67e22"}
            fig = go.Figure()
            for profil, color in PROFIL_COLORS.items():
                sub = stations_map[stations_map["profil"] == profil]
                if len(sub) == 0:
                    continue
                nom_col = sub[name_col] if name_col and name_col in sub.columns \
                          else sub["station_id"].astype(str)
                fig.add_trace(go.Scattermapbox(
                    lat=sub[lat_col], lon=sub[lon_col],
                    mode="markers",
                    marker=dict(size=9, color=color, opacity=1.0),
                    name=f"{profil} ({len(sub)})",
                    text=nom_col,
                    hovertemplate="<b>%{text}</b><br>"
                                  f"Profil : {profil}<br>"
                                  "Taux moyen : %{customdata[0]:.1f}%<extra></extra>",
                    customdata=sub[["taux_moyen"]].fillna(0).values,
                ))
            fig.update_layout(
                mapbox=dict(style="carto-positron",
                         center=dict(lat=48.856, lon=2.352), zoom=12),
                legend=dict(orientation="h", y=0, bgcolor="rgba(255,255,255,0.8)"),
                height=500, margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🟢 Résidentiel — se vide le matin  |  🔵 Mixte — flux équilibré  |  🟠 Bureaux — se remplit le matin")
        else:
            st.info("Coordonnées GPS non trouvées dans le dataset (colonnes lat/lon absentes).")

    # Répartition temporelle
    with st.expander("1. Répartition temporelle des relevés", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            daily = df_raw.groupby(df_raw["datetime"].dt.date).size().reset_index()
            daily.columns = ["date", "n"]
            fig = go.Figure(go.Scatter(
                x=daily["date"], y=daily["n"],
                fill="tozeroy", fillcolor="rgba(52,152,219,0.3)",
                line=dict(color=COLORS["primary"], width=2),
            ))
            fig.update_layout(title="Nombre de relevés par jour calendaire",
                              xaxis_title="Date", yaxis_title="Relevés",
                              height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            DAY_FR_REP = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
            dow_counts = df_raw.groupby(df_raw["datetime"].dt.dayofweek).size().reset_index()
            dow_counts.columns = ["dow", "n"]
            fig = go.Figure(go.Bar(
                x=[DAY_FR_REP[d] for d in dow_counts["dow"]],
                y=dow_counts["n"],
                marker_color=[COLORS["primary"]] * 5 + [COLORS["warning"]] * 2,
                text=[f"{v:,}" for v in dow_counts["n"]],
                textposition="outside",
            ))
            fig.update_layout(title="Relevés par jour de semaine",
                              yaxis_title="Relevés", height=280,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Distribution taux
    with st.expander("2. Distribution du taux de remplissage", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Histogram(
                x=df[TARGET], nbinsx=50,
                marker_color=COLORS["primary"], marker_line_color="white",
            ))
            fig.add_vline(x=df[TARGET].mean(), line_dash="dash",
                          line_color=COLORS["danger"],
                          annotation_text=f"Moyenne : {df[TARGET].mean():.1f}%")
            fig.add_vline(x=df[TARGET].median(), line_dash="dot",
                          line_color=COLORS["warning"],
                          annotation_text=f"Médiane : {df[TARGET].median():.1f}%")
            fig.update_layout(title="Distribution du taux",
                              height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            tranches = ["Vide\n(0-20%)", "Peu fournie\n(20-40%)",
                        "Normale\n(40-60%)", "Bien fournie\n(60-80%)", "Pleine\n(80-100%)"]
            counts = pd.cut(df[TARGET], bins=[0, 20, 40, 60, 80, 100]).value_counts(sort=False)
            fig = go.Figure(go.Bar(
                x=tranches, y=counts.values,
                marker_color=[COLORS["danger"], COLORS["warning"], COLORS["secondary"],
                              COLORS["primary"], COLORS["dark"]],
                text=[f"{v:,}" for v in counts.values], textposition="outside",
            ))
            fig.update_layout(title="Répartition par niveau",
                              height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Profil temporel
    with st.expander("3. Profil temporel", expanded=True):
        DAY_FR  = ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"]
        hourly  = df.groupby("hour")[TARGET].mean()
        daily_v = df.groupby("day_of_week")[TARGET].mean()
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Scatter(
                x=hourly.index, y=hourly.values,
                mode="lines+markers", line=dict(color=COLORS["primary"], width=2),
                fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
            ))
            fig.add_vrect(x0=7,  x1=9,  fillcolor="rgba(46,204,113,0.15)",
                          line_width=0, annotation_text="Pointe matin")
            fig.add_vrect(x0=17, x1=19, fillcolor="rgba(230,126,34,0.15)",
                          line_width=0, annotation_text="Pointe soir")
            fig.update_layout(title="Taux moyen selon l'heure",
                              height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(go.Bar(
                x=DAY_FR, y=daily_v.values,
                marker_color=[COLORS["primary"]]*5 + [COLORS["warning"]]*2,
                text=[f"{v:.1f}%" for v in daily_v.values], textposition="outside",
            ))
            fig.update_layout(title="Taux moyen par jour",
                              height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Impact météo
    with st.expander("4. Impact météo", expanded=False):
        SEV_LABELS = ["0 Clair","1 Nuageux","2 Pluie légère","3 Pluie forte","4 Extrême"]
        SEV_COLORS = ["#f39c12","#95a5a6","#3498db","#2980b9","#8e44ad"]
        sev_counts  = df["weather_severity"].value_counts().sort_index()
        mean_by_sev = df.groupby("weather_severity")[TARGET].mean()
        col1, col2 = st.columns(2)
        with col1:
            idx = sev_counts.index.tolist()
            fig = go.Figure(go.Bar(
                x=[SEV_LABELS[i] for i in idx], y=sev_counts.values,
                marker_color=[SEV_COLORS[i] for i in idx],
            ))
            fig.update_layout(title="Répartition météo", height=300,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            idx2 = mean_by_sev.index.tolist()
            fig = go.Figure(go.Bar(
                x=[SEV_LABELS[i] for i in idx2], y=mean_by_sev.values.round(1),
                marker_color=[SEV_COLORS[i] for i in idx2],
                text=[f"{v:.1f}%" for v in mean_by_sev.values], textposition="outside",
            ))
            fig.update_layout(title="Taux moyen par météo", height=300,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def page_preprocessing():
    st.title("Phase 2 — Preprocessing & Feature Engineering")

    train_df = load_csv(FILES["train"])
    test_df  = load_csv(FILES["test"])
    if train_df is None or test_df is None:
        st.info("Lancer `02_preprocessing.py` pour générer les datasets préparés.")
        return

    n_feats = len([c for c in train_df.columns
                   if c not in ["residual", TARGET, "station_id"]])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train", f"{len(train_df):,}")
    c2.metric("Test",  f"{len(test_df):,}")
    c3.metric("Features", f"{n_feats}")
    c4.metric("Résidu σ", f"{train_df['residual'].std():.1f}%")
    st.divider()

    with st.expander("1. Split temporel train / test", expanded=True):
        total   = len(train_df) + len(test_df)
        n_train = len(train_df)
        n_test  = len(test_df)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=["Dataset complet"], x=[n_train], orientation="h",
            name=f"Train ({n_train/total*100:.0f}%)",
            marker_color=COLORS["primary"],
            text=f"Train  {n_train/total*100:.0f}%",
            textposition="inside", insidetextanchor="middle",
        ))
        fig.add_trace(go.Bar(
            y=["Dataset complet"], x=[n_test], orientation="h",
            name=f"Test ({n_test/total*100:.0f}%)",
            marker_color=COLORS["danger"],
            text=f"Test  {n_test/total*100:.0f}%",
            textposition="inside", insidetextanchor="middle",
        ))
        fig.update_layout(
            barmode="stack",
            height=120,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.35),
            xaxis_title="Nombre de relevés",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Split temporel 80/20 basé sur le temps — "
            f"train : {n_train:,} relevés | test : {n_test:,} relevés. "
            "Le split est temporel (pas aléatoire) pour simuler la production : "
            "toujours prédire vers le futur."
        )

    with st.expander("2. Comportement historique — station_trend_avg", expanded=True):
        if "station_trend_avg" in train_df.columns:
            corr = train_df["station_trend_avg"].corr(train_df[TARGET])
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=train_df[TARGET], nbinsx=50, name="Taux réel",
                marker_color=COLORS["danger"], opacity=0.6,
            ))
            fig.add_trace(go.Histogram(
                x=train_df["station_trend_avg"], nbinsx=50, name="Trend historique",
                marker_color=COLORS["primary"], opacity=0.6,
            ))
            fig.update_layout(
                barmode="overlay",
                title=f"Taux réel vs trend historique — corrélation = {corr:.2f}",
                height=360, margin=dict(l=0, r=0, t=50, b=0),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Message dynamique basé sur la corrélation réelle
            if corr >= 0.85:
                st.success(
                    f"✅ **Corrélation excellente ({corr:.2f})** — "
                    f"`station_trend_avg` capture très fidèlement les habitudes de chaque station. "
                    f"C'est la feature la plus puissante du pipeline."
                )
            elif corr >= 0.70:
                st.info(
                    f"ℹ️ **Corrélation bonne ({corr:.2f})** — "
                    f"`station_trend_avg` apporte un signal utile mais le modèle devra "
                    f"compenser les écarts via le résidu."
                )
            else:
                st.warning(
                    f"⚠️ **Corrélation faible ({corr:.2f})** — "
                    f"Les habitudes historiques ne prédisent pas bien le taux actuel. "
                    f"Le résidu aura plus de travail à faire."
                )

    with st.expander("3. Cible résiduelle — réduction de variance", expanded=True):
        if "residual" in train_df.columns:
            std_brut = train_df[TARGET].std() if TARGET in train_df.columns else 29.0
            std_res  = train_df["residual"].std()
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Histogram(
                    x=train_df["residual"].clip(-50, 50), nbinsx=60,
                    marker_color=COLORS["primary"], marker_line_color="white",
                ))
                fig.add_vline(x=0, line_dash="dash", line_color=COLORS["danger"],
                              annotation_text="0 = normale")
                fig.update_layout(title="Distribution du résidu (cible du modèle)",
                                  height=340, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(go.Bar(
                    x=["Taux brut", "Résidu"],
                    y=[round(std_brut, 1), round(std_res, 1)],
                    marker_color=[COLORS["danger"], COLORS["secondary"]],
                    text=[f"σ={std_brut:.1f}%", f"σ={std_res:.1f}%"],
                    textposition="outside", width=0.4,
                ))
                fig.update_layout(title=f"Réduction ×{std_brut/std_res:.1f}",
                                  height=340, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
            st.success(f"Variance réduite ×{std_brut/std_res:.1f} — tâche plus facile pour le modèle ✓")
            reduction = std_brut / std_res
            if reduction >= 2.5:
                st.success(
                    f"✅ **Réduction majeure (×{reduction:.1f})** — "
                    f"La variance passe de σ={std_brut:.1f}% à σ={std_res:.1f}%. "
                    f"Le modèle prédit un signal beaucoup plus concentré, "
                    f"ce qui améliore significativement la précision."
                )
            elif reduction >= 1.5:
                st.info(
                    f"ℹ️ **Réduction modérée (×{reduction:.1f})** — "
                    f"La variance passe de σ={std_brut:.1f}% à σ={std_res:.1f}%. "
                    f"Le trend historique capture une partie du signal, "
                    f"mais des patterns complexes restent à apprendre."
                )
            else:
                st.warning(
                    f"⚠️ **Réduction faible (×{reduction:.1f})** — "
                    f"La variance reste élevée (σ={std_res:.1f}%). "
                    f"Le trend historique n'explique pas bien la variabilité — "
                    f"envisager d'enrichir `station_trend_avg` avec plus de données."
                )

    with st.expander("4. Récapitulatif des 24 features", expanded=False):
        rows = [
            ("station_trend_avg",     "Comportement historique", "Habitudes station × heure × jour × mois"),
            ("hour",                  "Temporel",                "Heure de la journée (0-23)"),
            ("day_of_week",           "Temporel",                "Jour de la semaine (0=lundi)"),
            ("month",                 "Temporel",                "Mois (1-12)"),
            ("hour_sin / hour_cos",   "Cyclique",                "Encodage cyclique — 23h et 0h sont proches"),
            ("dow_sin / dow_cos",     "Cyclique",                "Encodage cyclique du jour"),
            ("is_holiday",            "Calendrier",              "Jour férié (0/1)"),
            ("is_vacation",           "Calendrier",              "Vacances scolaires (0/1)"),
            ("apparent_temperature",  "Météo",                   "Température ressentie (°C)"),
            ("weather_severity",      "Météo",                   "Sévérité 0 (clair) à 4 (extrême)"),
            ("is_frozen",             "Météo",                   "Conditions glaciales (0/1)"),
            ("is_stormy",             "Météo",                   "Orage (0/1)"),
            ("capacity",              "Station",                 "Capacité totale"),
            ("capacity_group",        "Station",                 "Catégorie taille (0-3)"),
            ("lag_60min",             "Série temporelle",        "Taux 1h avant"),
            ("lag_240min",            "Série temporelle",        "Taux 4h avant"),
            ("lag_res_240min",        "Série temporelle",        "Écart à la normale 4h avant"),
            ("temp_anomalie",         "Météo enrichi",           "Écart de temp. à la normale du mois"),
            ("is_peak_hour",          "Temporel enrichi",        "Heure de pointe (0/1)"),
            ("is_friday_evening",     "Temporel enrichi",        "Vendredi soir (0/1)"),
            ("is_monday_morning",     "Temporel enrichi",        "Lundi matin (0/1)"),
            ("morning_evening_ratio", "Station",                 "Profil résidentiel / bureaux / mixte"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Feature", "Type", "Ce qu'elle capte"]),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — MODÉLISATION
# ─────────────────────────────────────────────────────────────────────────────

def page_modeling():
    st.title("Phase 3 — Modélisation")

    results_df = load_csv(FILES["resultats"])
    if results_df is None:
        st.info("Lancer `04_training.py` pour générer les résultats.")
        return

    if "Modèle" in results_df.columns:
        results_df = results_df.rename(columns={"Modèle": "Modele"})

    best         = results_df.sort_values("MAE").iloc[0]
    baseline_row = results_df[results_df["Modele"] == "Baseline (moyenne)"]
    baseline_mae = float(baseline_row["MAE"].values[0]) if len(baseline_row) \
                   else float(results_df["MAE"].max())
    gain = (baseline_mae - float(best["MAE"])) / baseline_mae * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleur modèle",  best["Modele"])
    c2.metric("MAE",              f"{best['MAE']:.2f}%")
    c3.metric("R²",               f"{best['R2']:.4f}")
    c4.metric("Gain vs baseline", f"{gain:.0f}%")
    st.divider()

    with st.expander("1. Benchmark — comparaison des modèles", expanded=True):
        plot_df   = results_df.sort_values("MAE")
        n         = len(plot_df)
        bar_colors = [
            COLORS["secondary"] if i == 0 else
            COLORS["primary"]   if i < n * 0.35 else
            COLORS["warning"]   if i < n * 0.70 else
            COLORS["danger"]
            for i in range(n)
        ]
        fig = go.Figure(go.Bar(
            x=plot_df["MAE"], y=plot_df["Modele"],
            orientation="h", marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in plot_df["MAE"]], textposition="outside",
        ))
        fig.add_vline(x=baseline_mae, line_dash="dash", line_color=COLORS["danger"],
                      annotation_text=f"Baseline : {baseline_mae:.2f}%")
        fig.update_layout(title="Benchmark — MAE par modèle",
                          xaxis_title="MAE (plus c'est court, mieux c'est)",
                          height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(plot_df[["Modele","MAE","RMSE","R2","Temps"]]
                     .style.background_gradient(subset=["MAE"], cmap="RdYlGn_r"),
                     use_container_width=True, hide_index=True)

        # Message dynamique basé sur le gain et l'écart entre modèles
        second = results_df.iloc[1] if len(results_df) > 1 else None
        ecart  = float(second["MAE"]) - float(best["MAE"]) if second is not None else 0

        if gain >= 60:
            gain_msg = f"✅ **Gain élevé ({gain:.0f}%)** vs baseline"
        elif gain >= 30:
            gain_msg = f"ℹ️ **Gain modéré ({gain:.0f}%)** vs baseline"
        else:
            gain_msg = f"⚠️ **Gain faible ({gain:.0f}%)** vs baseline — à investiguer"

        if ecart < 0.1:
            ecart_msg = (
                f"Les modèles **{best['Modele']}** et **{second['Modele']}** "
                f"sont quasi-équivalents (écart {ecart:.2f}%) — "
                f"le choix final peut se faire sur le temps d'entraînement."
            )
        else:
            ecart_msg = (
                f"**{best['Modele']}** domine clairement "
                f"avec {ecart:.2f}% d'avance sur **{second['Modele'] if second is not None else '—'}**."
            )

        st.info(f"{gain_msg}. {ecart_msg}")

    with st.expander("2. Importance des features", expanded=True):
        imp_df = load_csv(FILES["importance"])
        if imp_df is None:
            st.info("feature_importance.csv non trouvé — lancer 04_training.py")
        else:
            imp_df = imp_df.sort_values("importance", ascending=True)
            feat_colors = [
                COLORS["secondary"] if v > 5 else
                COLORS["primary"]   if v > 1 else "#bdc3c7"
                for v in imp_df["importance"]
            ]
            fig = go.Figure(go.Bar(
                x=imp_df["importance"], y=imp_df["feature"],
                orientation="h", marker_color=feat_colors,
                text=[f"{v:.1f}%" for v in imp_df["importance"]],
                textposition="outside",
            ))
            fig.update_layout(title="Quels facteurs influencent le plus les prédictions ?",
                              xaxis_title="Influence (%)",
                              height=560, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# BILAN — R² RECONSTRUIT
# ─────────────────────────────────────────────────────────────────────────────

def page_bilan():
    st.title("Bilan — R² Reconstruit")
    st.markdown("`taux_prédit = station_trend_avg + résidu_prédit`")

    results_df = load_csv(FILES["resultats"])
    test_df    = load_csv(FILES["test"])
    preds_df   = load_csv(FILES["preds"], optional=True)

    if results_df is None or test_df is None:
        st.info("Lancer 02_preprocessing.py et 04_training.py pour générer les données.")
        return

    if "Modèle" in results_df.columns:
        results_df = results_df.rename(columns={"Modèle": "Modele"})

    best = results_df.sort_values("MAE").iloc[0]

    if preds_df is not None and "taux_pred" in preds_df.columns:
        taux_reel = preds_df[TARGET].values if TARGET in preds_df.columns else None
        taux_pred = preds_df["taux_pred"].values
    elif "station_trend_avg" in test_df.columns and "residual" in test_df.columns:
        taux_reel = test_df[TARGET].values if TARGET in test_df.columns else None
        taux_pred = np.clip(
            test_df["station_trend_avg"].values + test_df["residual"].values, 0, 100
        )
        st.info("test_avec_predictions.csv non trouvé — approximation avec residual réel.")
    else:
        st.error("Données insuffisantes pour la reconstruction.")
        return

    if taux_reel is None:
        st.error(f"Colonne {TARGET} manquante.")
        return

    r2_reconstruit = float(r2_score(taux_reel, taux_pred))
    r2_residu      = float(best["R2"])
    mae_f          = float(best["MAE"])
    bl_f           = float(np.mean(np.abs(taux_reel - taux_reel.mean())))
    gain_f         = (bl_f - mae_f) / bl_f * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE taux réel",    f"{mae_f:.2f}%")
    c2.metric("R² taux réel",     f"{r2_reconstruit:.4f}")
    c3.metric("R² résidu",        f"{r2_residu:.4f}")
    c4.metric("Gain vs baseline", f"{gain_f:.0f}%")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        idx = np.random.default_rng(RANDOM_SEED).choice(
            len(taux_pred), size=min(4000, len(taux_pred)), replace=False
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=taux_pred[idx], y=taux_reel[idx], mode="markers",
            marker=dict(color=COLORS["primary"], size=3, opacity=0.2),
            name="Relevés test",
        ))
        fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines",
                                 line=dict(color=COLORS["danger"], dash="dash"),
                                 name="Parfait"))
        fig.update_layout(
            title=f"Prédictions vs réalité — R²={r2_reconstruit:.4f}",
            xaxis_title="Taux prédit (%)", yaxis_title="Taux réel (%)",
            height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=["Sans modèle\n(prédiction naïve)", f"Avec {best['Modele']}"],
            y=[round(bl_f, 1), round(mae_f, 1)],
            marker_color=[COLORS["danger"], COLORS["secondary"]],
            text=[f"{bl_f:.1f}%", f"{mae_f:.1f}%"],
            textposition="outside", width=0.4,
        ))
        fig.update_layout(title=f"Gain ML : {gain_f:.0f}%",
                          yaxis_title="Erreur moyenne (%)",
                          height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    if r2_reconstruit >= 0.85:
        st.success(
            f"✅ **Excellent** — Le pipeline explique **{r2_reconstruit*100:.1f}%** de la variance "
            f"du taux réel avec une erreur moyenne de **{mae_f:.2f}%** "
            f"({gain_f:.0f}% de réduction vs prédiction naïve)."
        )
    elif r2_reconstruit >= 0.70:
        st.info(
            f"ℹ️ **Bon** — Le pipeline explique **{r2_reconstruit*100:.1f}%** de la variance "
            f"du taux réel. MAE = **{mae_f:.2f}%**, gain = **{gain_f:.0f}%** vs baseline. "
            f"Des données supplémentaires (>3 mois) permettraient d'atteindre 85%+."
        )
    else:
        st.warning(
            f"⚠️ **Résultat partiel** — R²={r2_reconstruit:.4f}, MAE={mae_f:.2f}%. "
            f"Le modèle améliore la baseline de {gain_f:.0f}% mais reste perfectible. "
            f"Investiguer : durée du dataset, features manquantes, ou données aberrantes."
        )

    # Rapport texte (si disponible)
    if FILES["bilan"].exists():
        with st.expander("📄 Rapport bilan_final.txt", expanded=False):
            st.code(FILES["bilan"].read_text(encoding="utf-8"))

    st.divider()

    # ── Décisions méthodologiques ─────────────────────────────────────────
    with st.expander("🔬 Décisions méthodologiques clés", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Lags temporels — merge_asof**")
            st.markdown(
                "`shift(N)` décale de N **lignes**, pas de N **minutes**. "
                "Avec des intervalles de collecte variables (médiane ≈ 69 min), "
                "`shift(8)` peut représenter 96 min ou 6h selon les jours. "
                "`merge_asof` garantit un lag temporel réel avec ±30 min de tolérance."
            )
            st.markdown("**Cible résiduelle**")
            st.markdown(
                "On prédit `résidu = taux_réel − station_trend_avg` plutôt que le taux brut. "
                "Réduction de variance ×2.7 (σ 29% → 11%) — "
                "la tâche est plus ciblée et plus facile à apprendre."
            )

        with col2:
            st.markdown("**Split temporel 80/20**")
            st.markdown(
                "Split basé sur le temps, pas aléatoire. "
                "Simule la condition de production : toujours prédire vers le futur. "
                "La CV a été retirée — avec ~1 mois de données, les folds (~6 jours) "
                "sont trop courts pour capturer les habitudes hebdomadaires. "
                "Le test set 80/20 est la métrique de référence."
            )
            st.markdown("**Modèles exclus définitivement**")
            st.markdown(
                "- **KNN** : malédiction de la dimensionnalité (24 features, 1.2M lignes) \n"
                "- **Extra Trees** : seuils aléatoires contre-productifs sur features propres \n"
                "- **SVR RBF** : O(n²) mémoire, contraint à 10k lignes"
            )

    # ── Limites et perspectives ───────────────────────────────────────────
    with st.expander("🚀 Limites et perspectives", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Limite principale**")
            st.markdown(
                "Période de données courte (~1 mois). "
                "Sur 12-24 mois, la saisonnalité (été vs hiver, vacances scolaires) "
                "développerait un signal bien plus fort. "
                "R² reconstruit estimé à **0.90+** avec 1 an de données."
            )

        with col2:
            st.markdown("**Pistes d'amélioration**")
            st.markdown(
                "- Ajouter des événements externes (concerts, grèves, météo extrême) \n"
                "- Features de voisinage (stations proches — effets de report) \n"
                "- Modèle LSTM pour capturer les dépendances temporelles long-terme \n"
                "- Réentraînement mensuel glissant sur données cumulées"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

if section == "Phase 1 — DataViz":
    page_dataviz()
elif section == "Phase 2 — Preprocessing":
    page_preprocessing()
elif section == "Phase 3 — Modélisation":
    page_modeling()
else:
    page_bilan()
