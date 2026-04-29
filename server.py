"""
server.py
Flask backend for the Blood Pressure Estimation App.

Endpoints:
  GET  /              → serve index.html
  GET  /api/status    → model/calibration loaded status
  POST /api/predict   → {ir: [...], red: [...], fs: 20} → predictions
  POST /api/calibrate → {sessions: [{csv_data, sbp, dbp}, ...]} → cal_obj
  POST /api/save_csv  → {csv_data: "...", label: "..."} → saved path
  POST /api/clear_calibration → clear guest calibration

All endpoints accept an optional `profile` query parameter (e.g. ?profile=anuk).
Profile calibrations are stored separately and are never cleared on refresh.
"""

import os
import io
import json
import time
import traceback
import numpy as np
import joblib

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
CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "calibration")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(CALIBRATION_DIR, exist_ok=True)

# ── Globals loaded at startup ─────────────────────────────────────────────────
MODELS: dict | None = None
MODEL_ERROR: str | None = None


def _cal_path_for_profile(profile: str | None) -> str:
    """Return the calibration file path for a given profile name.
    Guest/default uses 'calibration.joblib', named profiles use '<name>.joblib'."""
    if profile:
        safe = "".join(c for c in profile if c.isalnum() or c in ("_", "-")).lower()
        return os.path.join(CALIBRATION_DIR, f"{safe}.joblib")
    return os.path.join(CALIBRATION_DIR, "calibration.joblib")


def _load_cal(profile: str | None) -> dict | None:
    path = _cal_path_for_profile(profile)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _save_cal(cal_obj: dict, profile: str | None):
    path = _cal_path_for_profile(profile)
    joblib.dump(cal_obj, path)


def _boot():
    global MODELS, MODEL_ERROR
    try:
        MODELS = load_models(MODEL_DIR)
        print(f"[server] Models loaded from {MODEL_DIR}")
    except Exception as exc:
        MODEL_ERROR = str(exc)
        print(f"[server] WARNING: Could not load models — {exc}")

    # Clear guest calibration on startup (profiles are preserved)
    try:
        guest_path = _cal_path_for_profile(None)
        if os.path.exists(guest_path):
            os.remove(guest_path)
            print("[server] Cleared guest calibration file on startup.")
    except Exception as e:
        print(f"[server] Note: Could not clear calibration file: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(os.path.join(os.path.dirname(__file__), "index.html"))


@app.route("/api/status")
def status():
    profile = request.args.get("profile")
    cal_obj = _load_cal(profile)
    return jsonify({
        "models_loaded":      MODELS is not None,
        "calibration_loaded": cal_obj is not None,
        "model_error":        MODEL_ERROR,
        "cal_sessions":       cal_obj.get("n_sessions") if cal_obj else 0,
        "bias_sbp":           cal_obj.get("bias_sbp")   if cal_obj else None,
        "bias_dbp":           cal_obj.get("bias_dbp")   if cal_obj else None,
        "profile":            profile or "guest",
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODELS is None:
        return jsonify({"error": f"Models not loaded: {MODEL_ERROR}"}), 503

    profile = request.args.get("profile")
    body = request.get_json(force=True)
    ir  = np.array(body.get("ir",  []), dtype=float)
    red = np.array(body.get("red", []), dtype=float)
    fs  = float(body.get("fs", 20))

    if len(ir) < 2 or len(red) < 2:
        return jsonify({"error": "Need at least 2 IR and Red samples"}), 400

    try:
        cal_obj = _load_cal(profile)
        result = predict(ir, red, MODELS, cal_obj=cal_obj, fs=fs)
        return jsonify(result)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    if MODELS is None:
        return jsonify({"error": f"Models not loaded: {MODEL_ERROR}"}), 503

    profile = request.args.get("profile")
    body = request.get_json(force=True)
    sessions = body.get("sessions", [])

    if len(sessions) < 1:
        return jsonify({"error": "Provide at least 1 session"}), 400

    try:
        fs = float(body.get("fs", 20))
        cal = run_calibration(sessions, MODELS, fs=fs)
        # Save to the profile-specific path
        _save_cal(cal, profile)
        return jsonify({
            "success":    True,
            "bias_sbp":   cal["bias_sbp"],
            "bias_dbp":   cal["bias_dbp"],
            "n_sessions": cal["n_sessions"],
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/clear_calibration", methods=["POST"])
def api_clear_calibration():
    """Only clears guest calibration. Named profiles are never auto-cleared."""
    try:
        guest_path = _cal_path_for_profile(None)
        if os.path.exists(guest_path):
            os.remove(guest_path)
    except Exception as e:
        print(f"[server] Note: Could not clear calibration file: {e}")
    return jsonify({"success": True})


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
