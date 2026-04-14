"""
load_all_raw.py — Reconstruction du dataset complet depuis HuggingFace
=======================================================================
Concatène tous les fichiers dataset_velib_raw_XX.csv depuis HuggingFace
pour reconstituer l'historique complet avant preprocessing.

Usage dans les notebooks / scripts :
    from load_all_raw import load_all_raw
    df = load_all_raw()

Ou directement :
    python load_all_raw.py
"""

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

HF_REPO     = "voroman/velib-ml-data"
HF_BASE     = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
FILE_PREFIX = "dataset_velib_raw_"


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def list_raw_files_hf() -> list[str]:
    """Détecte les fichiers dataset_velib_raw_XX.csv disponibles sur HuggingFace.

    Stratégie : tester les noms séquentiels jusqu'au premier 404.
    Évite d'avoir besoin d'un token pour lister les fichiers d'un dataset public.

    retour : liste triée des noms de fichiers disponibles
    """
    files = []
    for i in range(1, 100):   # max 99 fichiers × 4GB = 396GB
        fname = f"{FILE_PREFIX}{i:02d}.csv"
        url   = HF_BASE + fname
        try:
            resp = requests.head(url, timeout=5)
            if resp.status_code == 200:
                size_mb = int(resp.headers.get("content-length", 0)) / (1024 ** 2)
                files.append(fname)
                print(f"  Trouvé : {fname}  ({size_mb:.1f}MB)")
            else:
                break   # premier fichier manquant = fin de la séquence
        except Exception:
            break
    return sorted(files)


def load_all_raw(verbose: bool = True) -> pd.DataFrame:
    """Charge et concatène tous les fichiers raw depuis HuggingFace.

    param verbose : afficher les logs de chargement
    retour        : DataFrame complet trié par station_id et datetime
    """
    if verbose:
        print(f"Recherche des fichiers raw sur HuggingFace ({HF_REPO})...")

    files = list_raw_files_hf()
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier {FILE_PREFIX}*.csv trouvé sur HuggingFace.\n"
            "Vérifier que la collecte a bien été lancée."
        )

    if verbose:
        print(f"\n{len(files)} fichier(s) trouvé(s) — chargement...")

    dfs = []
    for fname in files:
        url = HF_BASE + fname
        df  = pd.read_csv(url, parse_dates=["datetime"])
        dfs.append(df)
        if verbose:
            print(f"  {fname} → {len(df):,} lignes")

    # Concaténation + dédoublonnage + tri
    combined = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["station_id", "datetime"])
        .sort_values(["station_id", "datetime"])
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\nDataset complet : {len(combined):,} relevés")
        print(f"Période         : {combined['datetime'].min()} → {combined['datetime'].max()}")
        print(f"Stations        : {combined['station_id'].nunique():,}")

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_raw()
    # Sauvegarde optionnelle en local pour éviter de re-télécharger
    df.to_csv("dataset_velib_complet.csv", index=False)
    print(f"\ndataset_velib_complet.csv sauvegardé : {df.shape}")
