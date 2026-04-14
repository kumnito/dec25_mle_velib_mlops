"""
06_bilan.py — Phase 6 : Bilan final du pipeline
================================================
Agrège tous les résultats du pipeline et génère un rapport complet.

Entrées  : data/outputs/resultats_modeles_v2.csv
           data/outputs/feature_importance.csv
           data/outputs/test_avec_predictions.csv
Sorties  : data/outputs/bilan_final.txt   (rapport texte)
           Console (affichage complet)
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from config import (
    RESULTS_FILE, IMPORTANCE_FILE,
    PREDICTIONS_FILE, BILAN_FILE, TARGET_RAW,
    section, ok, info,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def load_if_exists(path: Path, label: str) -> pd.DataFrame | None:
    """Charge un CSV si le fichier existe, sinon affiche un avertissement."""
    if not path.exists():
        print(f"  ⚠  {label} non trouvé — section ignorée")
        return None
    return pd.read_csv(path)


def bar_chart(value: float, max_val: float = 100.0,
              width: int = 30, fill: str = "█") -> str:
    """Génère une barre de progression ASCII."""
    n = int((value / max_val) * width)
    return fill * n + "░" * (width - n)


# ─────────────────────────────────────────────────────────────────────────────
# SECTIONS DU RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def section_benchmark(results_df: pd.DataFrame) -> list[str]:
    """Résumé du benchmark de tous les modèles."""
    lines = [
        "═" * 60,
        "  1. BENCHMARK — COMPARAISON DES MODÈLES",
        "═" * 60,
        f"  {'Modèle':<35} {'MAE':>7} {'RMSE':>7} {'R²':>7}",
        "  " + "─" * 56,
    ]

    baseline_row = results_df[results_df["Modele"] == "Baseline (moyenne)"]
    baseline_mae = float(baseline_row["MAE"].values[0]) if len(baseline_row) else 99.0

    for _, row in results_df.iterrows():
        gain = (baseline_mae - row["MAE"]) / baseline_mae * 100
        marker = " ← MEILLEUR" if _ == 0 else ""
        lines.append(
            f"  {row['Modele']:<35} {row['MAE']:>6.2f}% {row['RMSE']:>6.2f}% {row['R2']:>6.4f}{marker}"
        )

    best = results_df.iloc[0]
    gain = (baseline_mae - float(best["MAE"])) / baseline_mae * 100
    lines += [
        "  " + "─" * 56,
        f"  Baseline MAE : {baseline_mae:.2f}%",
        f"  Gain obtenu  : {gain:.0f}%  (réduction d'erreur vs prédiction naïve)",
    ]
    return lines


def section_features(imp_df: pd.DataFrame) -> list[str]:
    """Top 10 des features les plus importantes."""
    lines = [
        "",
        "═" * 60,
        "  2. IMPORTANCE DES FEATURES (top 10)",
        "═" * 60,
    ]
    for _, row in imp_df.head(10).iterrows():
        bar = bar_chart(row["importance"], max_val=imp_df["importance"].max(), width=25)
        lines.append(f"  {row['feature']:<30} {bar}  {row['importance']:>5.1f}%")
    return lines


def section_predictions(preds_df: pd.DataFrame) -> list[str]:
    """Métriques finales sur le taux de remplissage reconstitué."""
    taux_reel = preds_df[TARGET_RAW].values
    taux_pred = preds_df["taux_pred"].values

    mae      = mean_absolute_error(taux_reel, taux_pred)
    r2       = r2_score(taux_reel, taux_pred)
    baseline = mean_absolute_error(taux_reel, np.full_like(taux_reel, taux_reel.mean()))
    gain     = (baseline - mae) / baseline * 100

    erreurs = taux_reel - taux_pred
    p90     = np.percentile(np.abs(erreurs), 90)

    lines = [
        "",
        "═" * 60,
        "  3. PERFORMANCES SUR LE TAUX RÉEL (0-100%)",
        "═" * 60,
        f"  MAE  : {mae:.2f}%  — erreur absolue moyenne",
        f"  R²   : {r2:.4f}  — variance expliquée du taux réel",
        f"  P90  : {p90:.2f}%  — 90% des erreurs sont inférieures à cette valeur",
        f"  Gain : {gain:.0f}%   — vs prédire toujours la moyenne",
        "",
        "  Distribution des erreurs :",
    ]

    bins   = [-30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30]
    labels = ["<-30", "-30:-20", "-20:-10", "-10:-5", "-5:-2",
              "-2:0", "0:2", "2:5", "5:10", "10:20", ">20"]
    counts, _ = np.histogram(erreurs, bins=bins + [999])
    total      = len(erreurs)
    for label, count in zip(labels, counts):
        pct = count / total * 100
        bar = bar_chart(pct, max_val=30, width=20)
        lines.append(f"    {label:>10}  {bar}  {pct:.1f}%")

    return lines


def section_decisions() -> list[str]:
    """Récapitulatif des décisions méthodologiques clés."""
    return [
        "",
        "═" * 60,
        "  4. DÉCISIONS MÉTHODOLOGIQUES CLÉS",
        "═" * 60,
        "  • Lags temporels via merge_asof",
        "    shift(N) décale de N lignes, pas de N minutes.",
        "    merge_asof garantit un lag temporel réel (±30min de tolérance).",
        "",
        "  • Cible résiduelle (residual)",
        "    Prédit l'écart à la normale station, pas le taux brut.",
        "    Réduction de variance ×2.7 → tâche plus facile pour le modèle.",
        "",
        "  • Split temporel 80/20 (pas aléatoire)",
        "    Simule la condition de production : toujours prédire vers le futur.",
        "    La CV a été retirée — avec ~1 mois de données, les folds sont",
        "    trop courts (~6j) pour capturer les habitudes hebdomadaires.",
        "    Le test set 80/20 est la métrique de référence.",
        "",
        "  • Modèles exclus définitivement",
        "    KNN         : malédiction dimensionnalité (24 features, 1.2M lignes)",
        "    Extra Trees : seuils aléatoires contre-productifs sur features propres",
        "    SVR RBF     : O(n²) mémoire, contraint à 10k lignes",
    ]


def section_limite() -> list[str]:
    """Limite principale et pistes d'amélioration."""
    return [
        "",
        "═" * 60,
        "  5. LIMITES ET PERSPECTIVES",
        "═" * 60,
        "  Limite principale : période de données courte (~1 mois).",
        "  Sur 12-24 mois, la saisonnalité développerait un signal",
        "  plus fort → R² estimé à 0.75-0.85.",
        "",
        "  Pistes d'amélioration :",
        "  • Ajouter des événements (concerts, grèves, météo extrême)",
        "  • Features de voisinage (stations proches)",
        "  • Modèle LSTM pour capturer les dépendances long-terme",
        "  • Rééquilibrage du dataset sur les événements rares",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    section("PHASE 6 — BILAN FINAL")

    # ── Chargement de tous les fichiers de résultats ──────────────────────
    results_df = load_if_exists(RESULTS_FILE,     "resultats_modeles_v2.csv")
    imp_df     = load_if_exists(IMPORTANCE_FILE,  "feature_importance.csv")
    preds_df   = load_if_exists(PREDICTIONS_FILE, "test_avec_predictions.csv")

    if results_df is None:
        raise FileNotFoundError(
            "resultats_modeles_v2.csv non trouvé. Lancer 04_training.py d'abord."
        )

    # ── Construction du rapport ───────────────────────────────────────────
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "╔" + "═" * 58 + "╗",
        "║  BILAN FINAL — PROJET VÉLIB' ML PIPELINE" + " " * 16 + "║",
        f"║  Généré le : {now}" + " " * (44 - len(now)) + "║",
        "╚" + "═" * 58 + "╝",
        "",
    ]

    lines += section_benchmark(results_df)

    if imp_df is not None:
        lines += section_features(imp_df)

    if preds_df is not None:
        lines += section_predictions(preds_df)

    lines += section_decisions()
    lines += section_limite()

    lines += [
        "",
        "═" * 60,
        "  FIN DU RAPPORT",
        "═" * 60,
    ]

    # ── Affichage console ─────────────────────────────────────────────────
    report = "\n".join(lines)
    print(report)

    # ── Sauvegarde du rapport ─────────────────────────────────────────────
    with open(BILAN_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    ok(f"Rapport sauvegardé → {BILAN_FILE.name}")

    section("PHASE 6 TERMINÉE — PIPELINE COMPLET")


if __name__ == "__main__":
    main()
