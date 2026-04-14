# 🚲 Projet Vélib' — ML Pipeline Docker

Prédiction du taux de remplissage des stations Vélib' avec un pipeline ML complet conteneurisé sous Docker.

---

## Architecture du projet

```
velib/
├── Dockerfile               ← Image pipeline ML (phases 1-6)
├── Dockerfile.streamlit     ← Image dashboard Streamlit
├── docker-compose.yml       ← Orchestration
├── requirements.txt         ← Dépendances Python
├── .env                     ← Variables d'environnement
│
├── data/
│   ├── raw/                 ← Placer ici : dataset_velib_300326.csv
│   ├── processed/           ← Généré par 02_preprocessing
│   └── outputs/             ← Généré par 04/05/06 + plots
│
├── src/
│   ├── config.py            ← Constantes partagées
│   ├── 01_dataviz.py        ← EDA → plots PNG
│   ├── 02_preprocessing.py  ← Feature engineering + split
│   ├── 03_hyperparameter.py ← Tuning XGBoost (Optuna)
│   ├── 04_training.py       ← Entraînement + CV + export
│   ├── 05_predictions.py    ← Prédictions sur test set
│   └── 06_bilan.py          ← Rapport final
│
└── streamlit/
    └── app.py               ← Dashboard (lecture locale)
```

---

## Prérequis

- **Docker** ≥ 24
- **Docker Compose** ≥ 2.20
- Le fichier `dataset_velib_300326.csv` dans `data/raw/`

---

## Démarrage rapide

### 1. Placer le dataset
```bash
cp /chemin/vers/dataset_velib_300326.csv data/raw/
```

### 2. Build des images
```bash
docker compose --profile pipeline --profile streamlit build
```


### 3. MLflow (optionnel http://localhost:5000)
```bash
docker compose --profile mlflow up -d mlflow
```
### MLflow (à stopper manuellement)
```bash
docker compose --profile mlflow down
```


### 4. Lancer le pipeline complet (toutes les phases)
```bash
# à éviter car "best_params.json" est déjà disponible
# Phase 3 - Hyperparameter tuning : très long !!!

# docker compose run --rm --profile pipeline pipeline
```

### 5. Lancer le dashboard Streamlit
```bash
docker compose --profile streamlit up -d streamlit
# Ouvrir : http://localhost:8501
```

### Arrêter tous les containers
```bash
docker compose --profile pipeline --profile mlflow --profile streamlit down
```


---

## Exécution phase par phase

```bash
# Phase 1 — DataViz (exports PNG dans data/outputs/plots/)
docker compose --profile pipeline run --rm pipeline python src/01_dataviz.py

# Phase 2 — Preprocessing (→ train/test CSV dans data/processed/)
docker compose --profile pipeline run --rm pipeline python src/02_preprocessing.py

# Phase 3 — Hyperparameter tuning (→ best_params.json)
# sur mon pc avec le dataset velib = 30h
# docker compose --profile pipeline run --rm pipeline python src/03_hyperparameter.py

# Phase 4 — Entraînement (→ modèle + résultats)
docker compose --profile pipeline run --rm pipeline python src/04_training.py

# Phase 5 — Prédictions (→ test_avec_predictions.csv)
docker compose --profile pipeline run --rm pipeline python src/05_predictions.py

# Phase 6 — Bilan (→ bilan_final.txt + affichage console)
docker compose --profile pipeline run --rm pipeline python src/06_bilan.py
```

---

## Fichiers produits

| Phase | Fichier | Localisation |
|---|---|---|
| 01 | `*.png` (7 graphiques) | `data/outputs/plots/` |
| 02 | `train_preprocessed.csv` | `data/processed/` |
| 02 | `test_preprocessed.csv` | `data/processed/` |
| 02 | `station_names.csv` | `data/processed/` |
| 03 | `best_params.json` | `data/outputs/` |
| 04 | `resultats_modeles_v2.csv` | `data/outputs/` |
| 04 | `feature_importance.csv` | `data/outputs/` |
| 04 | `best_model.joblib` | `data/outputs/` |
| 05 | `test_avec_predictions.csv` | `data/outputs/` |
| 06 | `bilan_final.txt` | `data/outputs/` |


---

## Variables d'environnement (.env)

| Variable | Défaut | Description |
|---|---|---|
| `RAW_FILENAME` | `dataset_velib_300326.csv` | Nom du fichier source |
| `OPTUNA_N_TRIALS` | `50` | Essais Optuna pour XGBoost |
| `TUNING_SAMPLE_FRAC` | `0.5` | Fraction du train pour le tuning |
| `STREAMLIT_PORT` | `8501` | Port du dashboard |
| `RANDOM_SEED` | `42` | Reproductibilité |

---

## Reproductibilité — XGBoost Linux vs Windows

Les résultats XGBoost peuvent différer légèrement entre le Docker (Linux) 
et un environnement local Windows (notebook), 
même avec la même version de librairie et les mêmes hyperparamètres.

**Cause : parallélisme flottant**

`tree_method="hist"` construit des histogrammes en accumulant des sommes de gradients sur 1.2M de lignes en parallèle (`n_jobs=-1`). En virgule flottante, l'addition n'est pas associative :

```
(a + b) + c ≠ a + (b + c)
```

Avec plusieurs threads, l'ordre dans lequel les résultats partiels sont agrégés dépend du scheduler OS — différent entre Linux et Windows. Ces micro-erreurs d'arrondi s'accumulent sur des millions d'opérations et produisent des arbres légèrement différents.

**Ampleur observée sur ce projet**

|    Environnement   | XGBoost MAE | Random Forest MAE |
|--------------------|-------------|-------------------|
|   Docker (Linux)   |    7.95%    |       7.92%       |
| Notebook (Windows) |    7.89%    |       7.92%       |

Random Forest, Ridge et Baseline sont **strictement identiques** entre les deux environnements — ce qui confirme que le preprocessing et le pipeline sont parfaitement alignés. L'écart de 0.06% sur XGBoost est exclusivement dû au parallélisme flottant inter-OS.

**Référence** : ce comportement est documenté dans la documentation officielle XGBoost :
> *"Results may not be perfectly reproducible across different platforms due to floating point non-associativity in parallel reductions."*

---

## Collecte en continu — HuggingFace + cron-job.org + HuggingFace Spaces

### Architecture

```
cron-job.org (*/5 * * * *)
      ↓  GET /collect
HuggingFace Spaces (collector_app.py)
      ↓  ingestion_hf.py
HuggingFace Datasets
      ├── dataset_velib_raw_01.csv  (≤ 4GB)
      ├── dataset_velib_raw_02.csv  (≤ 4GB)
      └── dataset_velib_raw_03.csv  (en cours)
```

### Stratégie fichiers rotatifs

- Taille max par fichier : **4GB**
- Nouveau fichier créé automatiquement quand la limite est atteinte
- Aucune donnée supprimée — historique complet préservé

### Déploiement

**1. Variables d'environnement**
```bash
# Dans .env
HF_TOKEN=ton_token_hf_write
HF_REPO=voroman/velib-ml-data
```

**2. Test local**
```bash
docker compose --profile collector up collector
curl http://localhost:8080/collect
```

**3. Déploiement sur HuggingFace Spaces**
```
huggingface.co/spaces → New Web Service → connecter le repo GitHub
Build command : docker build -f Dockerfile.collector -t collector .
Start command : gunicorn render_app:app --bind 0.0.0.0:$PORT
Variables : HF_TOKEN, HF_REPO
URL obtenue : https://ton-app.onhuggingface.co/spaces
```

**4. Configurer cron-job.org**
```
URL    : https://ton-app.onhuggingface.co/spaces/collect
Méthode: GET
Cron   : */5 * * * *
```

### Reconstruction du dataset complet

Pour entraîner les modèles sur l'historique complet :
```bash
# Depuis le container pipeline ou localement
python load_all_raw.py
# → dataset_velib_complet.csv reconstruit depuis tous les fichiers raw
```

Ou depuis un notebook :
```python
from load_all_raw import load_all_raw
df = load_all_raw()
```
