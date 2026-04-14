"""
ingestion_hf.py — Collecte des données Vélib' vers HuggingFace Datasets
========================================================================
Adapté depuis ingestion_data_workflow.py (version Supabase).
Remplace INSERT Supabase par append CSV rotatif sur HuggingFace.

Différences vs version Supabase :
    - Pas de table station_information séparée
      → les infos GPS sont dans le CSV raw (colonnes lat, lon, capacity, name)
    - Pas de contrainte de clé étrangère
      → les nouvelles stations sont insérées directement
    - Pas de batch de 500 (pas de limite API)
      → tout le relevé est uploadé en une fois
    - Fichiers rotatifs : nouveau fichier créé automatiquement si taille >= MAX_SIZE_GB

Variables d'environnement requises :
    HF_TOKEN  : token HuggingFace avec accès Write
    HF_REPO   : ex "voroman/velib-ml-data"
"""

import io
import os
import time
import requests
import holidays
import pandas as pd
from datetime import datetime, timezone
from huggingface_hub import HfApi

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

HF_REPO     = os.environ.get("HF_REPO",  "voroman/velib-ml-data")
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
FILE_PREFIX = "dataset_velib_raw_"
MAX_SIZE_GB = 4.0
MAX_SIZE_B  = int(MAX_SIZE_GB * 1024 ** 3)

VELIB_STATUS_URL = (
    "https://velib-metropole-opendata.smovengo.cloud"
    "/opendata/Velib_Metropole/station_status.json"
)
VELIB_INFO_URL = (
    "https://velib-metropole-opendata.smovengo.cloud"
    "/opendata/Velib_Metropole/station_information.json"
)
WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=48.8566&longitude=2.3522"
    "&current=apparent_temperature,weather_code"
)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_with_retry(url: str, retries: int = 3, delay: int = 10):
    """Récupère une URL avec plusieurs tentatives.

    param url     : URL à récupérer
    param retries : nombre maximum de tentatives
    param delay   : délai en secondes entre les tentatives
    retour        : JSON parsé ou None si toutes les tentatives échouent
    """
    for i in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"  Tentative {i+1}/{retries} échouée : {e}")
            if i < retries - 1:
                time.sleep(delay)
    return None


def get_vacation_status() -> bool:
    """Vérifie les vacances scolaires à Paris (Zone C) via l'API Education nationale.

    retour : True si vacances scolaires, False sinon ou si API indisponible
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = (
        f"https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
        f"fr-en-calendrier-scolaire/records?"
        f"where=start_date%20%3C%3D%20%27{today}%27%20AND%20"
        f"end_date%20%3E%3D%20%27{today}%27%20AND%20location%3D%27Paris%27"
    )
    data = fetch_with_retry(url)
    if data is None:
        print("  API vacances scolaires indisponible — is_vacation=False par défaut")
        return False
    return data.get("total_count", 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# FICHIERS ROTATIFS HUGGINGFACE
# ─────────────────────────────────────────────────────────────────────────────

def list_raw_files(api: HfApi) -> list:
    """Liste et trie les fichiers dataset_velib_raw_XX.csv sur HuggingFace.

    param api  : instance HfApi authentifiée
    retour     : liste triée des noms de fichiers existants
    """
    try:
        files = [
            f.path for f in api.list_repo_tree(
                repo_id=HF_REPO, repo_type="dataset", token=HF_TOKEN
            )
            if hasattr(f, "path")
            and f.path.startswith(FILE_PREFIX)
            and f.path.endswith(".csv")
        ]
        return sorted(files)
    except Exception:
        return []


def get_file_size_bytes(filename: str) -> int:
    """Récupère la taille en octets d'un fichier sur HuggingFace.

    param filename : nom du fichier
    retour         : taille en octets, 0 si non trouvé
    """
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"
    try:
        resp = requests.head(
            url,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            timeout=10,
        )
        return int(resp.headers.get("content-length", 0))
    except Exception:
        return 0


def get_current_file(api: HfApi) -> str:
    """Détermine le fichier actif — crée le suivant si taille >= MAX_SIZE_GB.

    param api  : instance HfApi authentifiée
    retour     : nom du fichier à utiliser pour l'écriture
    """
    existing = list_raw_files(api)

    if not existing:
        print(f"  Aucun fichier trouvé — création : {FILE_PREFIX}01.csv")
        return f"{FILE_PREFIX}01.csv"

    current    = existing[-1]
    size_bytes = get_file_size_bytes(current)
    size_gb    = size_bytes / (1024 ** 3)

    if size_bytes >= MAX_SIZE_B:
        num      = int(current.replace(FILE_PREFIX, "").replace(".csv", ""))
        new_file = f"{FILE_PREFIX}{num + 1:02d}.csv"
        print(f"  {current} plein ({size_gb:.2f}GB) → nouveau fichier : {new_file}")
        return new_file

    print(f"  Fichier actif : {current} ({size_gb:.3f}GB / {MAX_SIZE_GB}GB)")
    return current


def load_existing(filename: str) -> pd.DataFrame:
    """Charge le CSV existant depuis HuggingFace.

    param filename : nom du fichier à charger
    retour         : DataFrame existant ou vide si fichier absent
    """
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"
    try:
        df = pd.read_csv(url)
        print(f"  {len(df):,} lignes existantes chargées")
        return df
    except Exception:
        print(f"  Nouveau fichier — aucune ligne existante")
        return pd.DataFrame()


def upload_to_hf(api: HfApi, df: pd.DataFrame, filename: str) -> None:
    """Sérialise et uploade le DataFrame sur HuggingFace.

    param api      : instance HfApi authentifiée
    param df       : DataFrame complet à uploader
    param filename : nom du fichier cible sur HuggingFace
    """
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    api.upload_file(
        path_or_fileobj=buffer.getvalue().encode("utf-8"),
        path_in_repo=filename,
        repo_id=HF_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTE ET CONSTRUCTION DES RECORDS
# ─────────────────────────────────────────────────────────────────────────────

def build_records(stations_status: list,
                  stations_info:   dict,
                  temp:            float,
                  w_code:          int,
                  is_holiday_:     bool,
                  is_vacation_:    bool,
                  now_iso:         str) -> list:
    """Construit la liste des relevés à insérer.

    Fusionne les données de statut (status.json) et d'information
    (station_information.json — GPS, capacité, nom) en une seule ligne
    par station.

    param stations_status : liste des stations depuis station_status.json
    param stations_info   : dict {station_id: {name, lat, lon, capacity}}
    param temp            : température ressentie (°C)
    param w_code          : code météo WMO
    param is_holiday_     : True si jour férié
    param is_vacation_    : True si vacances scolaires
    param now_iso         : datetime UTC ISO 8601
    retour                : liste de dicts prêts à être insérés dans le CSV
    """
    records = []
    for s in stations_status:
        sid = int(s["station_id"])

        # Extraction des types de vélos
        bike_data = s.get("num_bikes_available_types", [])
        meca, ebike = 0, 0
        for item in bike_data:
            if "mechanical" in item:
                meca  = int(item["mechanical"])
            if "ebike" in item:
                ebike = int(item["ebike"])

        # Extraction des docks (deux noms possibles dans l'API)
        docks = int(s.get("num_docks_available", s.get("numDocksAvailable", 0)))

        # Calcul du taux de remplissage
        total_items = meca + ebike + docks
        capa_pct    = float((meca + ebike) / total_items * 100) if total_items > 0 else 0.0

        # Informations statiques depuis station_information.json
        info = stations_info.get(sid, {})

        records.append({
            # Identifiants
            "station_id":           sid,
            "name":                 info.get("name", ""),
            "lat":                  info.get("lat",  None),
            "lon":                  info.get("lon",  None),
            "capacity":             info.get("capacity", 0),
            # Statut temps réel
            "datetime":             now_iso,
            "bikes_mechanical":     meca,
            "bikes_ebike":          ebike,
            "numdocksavailable":    docks,
            "is_renting":           bool(s.get("is_renting", 1) == 1),
            "capacity_status":      capa_pct,
            # Contexte
            "is_holiday":           bool(is_holiday_),
            "is_vacation":          bool(is_vacation_),
            "apparent_temperature": float(temp),
            "weather_code":         int(w_code),
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def ingest() -> dict:
    """Exécute un cycle complet de collecte et d'upload vers HuggingFace.

    Étapes :
        1. Récupérer statut Vélib' + infos stations + météo + contexte (vacances, férié)
        2. Construire les relevés (fusion statut + infos GPS)
        3. Déterminer le fichier HuggingFace actif (rotation si nécessaire)
        4. Charger l'existant + append + dédoublonnage
        5. Uploader le CSV mis à jour

    retour : dict avec les métriques de la collecte
    """
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN manquant — définir la variable d'environnement")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Démarrage de la collecte Vélib'...")

    now     = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    fr_holidays_ = holidays.France()

    # ── Collecte des données ──────────────────────────────────────────────
    r_status  = fetch_with_retry(VELIB_STATUS_URL)
    r_info    = fetch_with_retry(VELIB_INFO_URL)
    r_weather = fetch_with_retry(WEATHER_URL)

    is_holiday_  = now.replace(tzinfo=None) in fr_holidays_
    is_vacation_ = get_vacation_status()

    if not r_status:
        print("Impossible de récupérer les données Vélib. Abandon.")
        return {"status": "error", "message": "Vélib API unavailable"}

    if not r_weather:
        print("Données météo indisponibles — valeurs par défaut utilisées (0.0 / 0)")
    temp   = r_weather["current"]["apparent_temperature"] if r_weather else 0.0
    w_code = r_weather["current"]["weather_code"]         if r_weather else 0

    # ── Construction de la table d'infos stations ─────────────────────────
    # {station_id: {name, lat, lon, capacity}}
    stations_info = {}
    if r_info:
        for s in r_info["data"]["stations"]:
            stations_info[int(s["station_id"])] = {
                "name":     s.get("name", ""),
                "lat":      float(s.get("lat",      0)),
                "lon":      float(s.get("lon",      0)),
                "capacity": int(s.get("capacity", 0)),
            }
    else:
        print("station_information.json indisponible — colonnes GPS seront vides")

    # ── Construction des relevés ──────────────────────────────────────────
    records = build_records(
        r_status["data"]["stations"],
        stations_info,
        temp, w_code,
        is_holiday_, is_vacation_,
        now_iso,
    )

    # Log de contrôle
    if records:
        t = records[0]
        print(f"  Contrôle station {t['station_id']} : "
              f"{t['bikes_mechanical']} meca, {t['bikes_ebike']} ebike, "
              f"{t['numdocksavailable']} docks | "
              f"temp={temp}°C | férié={is_holiday_} | vacances={is_vacation_}")

    new_df = pd.DataFrame(records)

    # ── Fichier actif + rotation ──────────────────────────────────────────
    api      = HfApi()
    filename = get_current_file(api)

    # ── Chargement + append + dédoublonnage ───────────────────────────────
    existing = load_existing(filename)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["station_id", "datetime"])
    else:
        combined = new_df

    # ── Upload ────────────────────────────────────────────────────────────
    upload_to_hf(api, combined, filename)
    size_mb = len(combined.to_csv(index=False).encode()) / (1024 ** 2)

    print(f"  Upload OK → {filename} | {len(new_df)} nouveaux relevés | "
          f"total {len(combined):,} | ~{size_mb:.1f}MB")

    return {
        "status":  "ok",
        "file":    filename,
        "n_new":   len(new_df),
        "n_total": len(combined),
        "size_mb": round(size_mb, 1),
    }


if __name__ == "__main__":
    result = ingest()
    print(f"\nRésultat : {result}")
