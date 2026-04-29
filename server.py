"""
server.py
Flask backend for the Blood Pressure Estimation App.

Endpoints:
  GET  /              → serve index.html
  GET  /api/status    → model/calibration loaded status
  POST /api/predict   → {ir: [...], red: [...], fs: 20} → predictions
  POST /api/calibrate → {sessions: [{csv_data, sbp, dbp}, ...]} → cal_obj
  POST /api/save_csv  → {csv_data: "...", label: "..."} → saved path
"""

import os
import io
import json
import time
import traceback
import numpy as np

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from pipeline_adapter import (
    load_models,
    predict,
    run_calibration,
    load_calibration,
    MODEL_DIR,
)

# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# ── Globals loaded at startup ─────────────────────────────────────────────────
MODELS: dict | None = None
CAL_OBJ: dict | None = None
MODEL_ERROR: str | None = None


def _boot():
    global MODELS, CAL_OBJ, MODEL_ERROR
    try:
        MODELS = load_models(MODEL_DIR)
        print(f"[server] Models loaded from {MODEL_DIR}")
    except Exception as exc:
        MODEL_ERROR = str(exc)
        print(f"[server] WARNING: Could not load models — {exc}")

    # ALWAYS clear old calibration file on startup so it starts fresh
    try:
        cal_path = os.path.join(os.path.dirname(__file__), "calibration", "calibration.joblib")
        if os.path.exists(cal_path):
            os.remove(cal_path)
            print("[server] Cleared old calibration file on startup.")
    except Exception as e:
        print(f"[server] Note: Could not clear calibration file: {e}")

    CAL_OBJ = load_calibration()
    if CAL_OBJ:
        print(f"[server] Calibration loaded: {CAL_OBJ}")
    else:
        print("[server] No calibration file found — using absolute model only.")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "index.html"))


@app.route("/api/status")
def status():
    return jsonify({
        "models_loaded":      MODELS is not None,
        "calibration_loaded": CAL_OBJ is not None,
        "model_error":        MODEL_ERROR,
        "cal_sessions":       CAL_OBJ.get("n_sessions") if CAL_OBJ else 0,
        "bias_sbp":           CAL_OBJ.get("bias_sbp")   if CAL_OBJ else None,
        "bias_dbp":           CAL_OBJ.get("bias_dbp")   if CAL_OBJ else None,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODELS is None:
        return jsonify({"error": f"Models not loaded: {MODEL_ERROR}"}), 503

    body = request.get_json(force=True)
    ir  = np.array(body.get("ir",  []), dtype=float)
    red = np.array(body.get("red", []), dtype=float)
    fs  = float(body.get("fs", 20))

    if len(ir) < 2 or len(red) < 2:
        return jsonify({"error": "Need at least 2 IR and Red samples"}), 400

    try:
        result = predict(ir, red, MODELS, cal_obj=CAL_OBJ, fs=fs)
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    global CAL_OBJ
    if MODELS is None:
        return jsonify({"error": f"Models not loaded: {MODEL_ERROR}"}), 503

    body = request.get_json(force=True)
    sessions = body.get("sessions", [])

    if len(sessions) < 1:
        return jsonify({"error": "Provide at least 1 session"}), 400

    try:
        fs = float(body.get("fs", 20))
        cal = run_calibration(sessions, MODELS, fs=fs)
        CAL_OBJ = cal
        return jsonify({
            "success":    True,
            "bias_sbp":   cal["bias_sbp"],
            "bias_dbp":   cal["bias_dbp"],
            "n_sessions": cal["n_sessions"],
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/save_csv", methods=["POST"])
def api_save_csv():
    body     = request.get_json(force=True)
    csv_data = body.get("csv_data", "")
    label    = body.get("label", "session")

    if not csv_data.strip():
        return jsonify({"error": "csv_data is empty"}), 400

    fname = f"{label}_{int(time.time())}.csv"
    path  = os.path.join(RECORDINGS_DIR, fname)
    with open(path, "w") as f:
        f.write(csv_data)

    return jsonify({"success": True, "filename": fname, "path": path})


# ──────────────────────────────────────────────────────────────────────────────
# Initialize models and calibration
_boot()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
