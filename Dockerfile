# ─────────────────────────────────────────────────────────────
# Image pipeline ML — Projet Vélib'
# Base : Python 3.11 slim pour limiter la taille de l'image
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Projet Vélib' ML Pipeline"
LABEL description="Pipeline ML : dataviz, preprocessing, tuning, training, predictions, bilan"

# Variables d'environnement Python
# PYTHONDONTWRITEBYTECODE : évite les fichiers .pyc inutiles
# PYTHONUNBUFFERED        : logs visibles en temps réel dans Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Répertoire de travail dans le container
WORKDIR /app

# Copie et installation des dépendances en premier
# (couche Docker cachée tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/

# Le dossier /app/data est monté en volume depuis l'hôte
# → ne pas COPY data/ ici, elle est injectée au runtime

# Point d'entrée par défaut : pipeline complet
# Surchargeable avec : docker compose run pipeline python src/02_preprocessing.py
CMD ["bash", "-c", \
     "python src/01_dataviz.py && \
      python src/02_preprocessing.py && \
      python src/03_hyperparameter.py && \
      python src/04_training.py && \
      python src/05_predictions.py && \
      python src/06_bilan.py"]
