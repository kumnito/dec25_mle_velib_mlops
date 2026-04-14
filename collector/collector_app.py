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
import threading
from datetime import datetime
from flask import Flask, jsonify

from ingestion_hf import ingest

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

    Lance la collecte en arrière-plan et répond immédiatement 200
    pour éviter le timeout du worker gunicorn.
    """
    def run():
        try:
            ingest()
        except Exception as e:
            print(f"[ERREUR collecte] {e}")

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "time": datetime.now().isoformat()}), 200


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
