"""
pipeline_adapter.py
Thin adapter used by server.py to interact with ppg_bp_pipeline_v3.py.
Keeps the server import-clean and makes unit testing easy.
"""

from ppg_bp_pipeline_v3 import (
    load_models,
    predict,
    run_calibration,
    load_calibration,
    MODEL_DIR,
)

__all__ = ["load_models", "predict", "run_calibration", "load_calibration", "MODEL_DIR"]
