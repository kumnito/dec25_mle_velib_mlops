"""
collector_app.py — Serveur Flask pour HuggingFace Spaces
=============================================
Expose un endpoint HTTP appelé par cron-job.org toutes les 5 minutes.
Hébergé gratuitement sur HuggingFace Spaces (SDK Docker).

HuggingFace Spaces :
    - Pas de mise en veille
    - Gratuit sans carte bancaire
    - RAM : 512MB — suffisant pour la collecte Vélib'

Démarrage local :
    python render_app.py

Démarrage sur HuggingFace Spaces :
    Start command : gunicorn collector_app:app --bind 0.0.0.0:7860
"""

import os
from datetime import datetime
from flask import Flask, jsonify

from ingestion_hf import collect

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def health():
    """Healthcheck — vérifie que le serveur est actif."""
    return jsonify({
        "status":  "ok",
        "service": "velib-collector",
        "time":    datetime.now().isoformat(),
    })


@app.route("/collect")
def run_collect():
    """Endpoint de collecte — appelé par cron-job.org toutes les 5 min.

    retour : JSON avec les métriques de la collecte
    """
    try:
        result = collect()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
