# Draft of the new ppg_bp_pipeline_v3.py
import os
import io
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.signal import find_peaks, butter, filtfilt, hilbert
from scipy.stats import skew, kurtosis, iqr, trim_mean
from scipy.fft import rfft, rfftfreq

# ──────────────────────────────────────────────────────────────────────────────
# StackedEnsemble — must be defined here so joblib can unpickle the .joblib files.
# ──────────────────────────────────────────────────────────────────────────────
class StackedEnsemble:
    def __init__(self, base=None, meta=None):
        self.base = base or {}
        self.meta = meta

    def predict(self, X):
        import numpy as _np
        base_preds = _np.column_stack([
            model.predict(X).reshape(-1) for model in self.base.values()
        ])
        return self.meta.predict(base_preds)

    def predict_with_uncertainty(self, X):
        import numpy as _np
        bp = _np.column_stack([m.predict(X).reshape(-1) for m in self.base.values()])
        return self.meta.predict(bp), _np.std(bp, axis=1)

    def fit(self, X, y):
        raise NotImplementedError("Fit not supported — use pre-trained model.")

import sys as _sys
_sys.modules['__main__'].StackedEnsemble = StackedEnsemble

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "calibration")
CALIBRATION_PATH = os.path.join(CALIBRATION_DIR, "calibration.joblib")

FS = 20          
WINDOW_SEC = 20    # from user script
STEP_SEC = 5       
MIN_BEATS_PER_WINDOW = 5     
MIN_HR, MAX_HR = 38, 185  
MAX_CV_RR = 0.45  
EPS = 1e-8

# ──────────────────────────────────────────────────────────────────────────────
# Signal pre-processing & Features (from OriginalFeatureTrainingCalibration.py)
# ──────────────────────────────────────────────────────────────────────────────
def bandpass(x, fs, low=0.5):
    high = min(8.0, fs/2.0-0.5)
    if high <= low: return x.astype(float)
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, x.astype(float))

def normalise(x):
    lo, hi = np.min(x), np.max(x)
    return (x - lo) / (hi - lo + EPS)

def _safe(fn, x): 
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return fn(x) if len(x) else np.nan

def s_mean(x):  return _safe(np.mean, x)
def s_std(x):   return _safe(np.std, x)
def s_med(x):   return _safe(np.median, x)
def s_min(x):   return _safe(np.min, x)
def s_max(x):   return _safe(np.max, x)
def s_iqr(x):   return _safe(iqr, x)
def s_rms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x**2))) if len(x) else np.nan
def s_mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan
def s_skew(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(skew(x)) if (len(x) >= 3 and np.std(x) > EPS) else np.nan
def s_kurt(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(kurtosis(x)) if (len(x) >= 4 and np.std(x) > EPS) else np.nan
def s_corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    return float(np.corrcoef(a, b)[0, 1]) if (len(a) >= 2 and np.std(a) > EPS and np.std(b) > EPS) else np.nan
def zcr(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return 0
    s = np.sign(x); s[s == 0] = 1; return int(np.sum(s[1:] != s[:-1]))

def agg_stats(feats, prefix, vals):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    for nm, fn in [("mean", np.mean), ("std", np.std), ("med", np.median),
                   ("min", np.min), ("max", np.max), ("iqr", iqr)]:
        feats[f"{prefix}_{nm}"] = fn(vals) if len(vals) else np.nan
    feats[f"{prefix}_cv"] = np.std(vals) / (np.mean(vals) + EPS) if len(vals) else np.nan

def detect_peaks_ppg(ppg, fs):
    md = max(int(0.33 * fs), 1)
    prom1 = max(np.std(ppg) * 0.30, EPS)
    peaks1, props1 = find_peaks(ppg, distance=md, prominence=prom1)
    if len(peaks1) >= MIN_BEATS_PER_WINDOW:
        return peaks1, props1
    prom2 = max(np.std(ppg) * 0.10, EPS)
    height = np.percentile(ppg, 20)
    peaks2, props2 = find_peaks(ppg, distance=md, prominence=prom2, height=height)
    return (peaks2, props2) if len(peaks2) > len(peaks1) else (peaks1, props1)

def window_ok(ppg, fs):
    peaks, _ = detect_peaks_ppg(ppg, fs)
    if len(peaks) < MIN_BEATS_PER_WINDOW: return False
    rr = np.diff(peaks) / fs
    if len(rr) < 2: return False
    hr = 60.0 / (np.mean(rr) + EPS)
    if not (MIN_HR <= hr <= MAX_HR): return False
    cv = np.std(rr) / (np.mean(rr) + EPS)
    if cv <= MAX_CV_RR: return True
    try:
        nyq = fs / 2.0
        high = min(8.0, nyq - 0.5)
        if high > 1.0:
            b, a = butter(4, [1.0 / nyq, high / nyq], btype="band")
            ppg_hp = filtfilt(b, a, ppg)
            ppg_hp = normalise(ppg_hp)
            peaks2, _ = detect_peaks_ppg(ppg_hp, fs)
            if len(peaks2) >= MIN_BEATS_PER_WINDOW:
                rr2 = np.diff(peaks2) / fs
                hr2 = 60.0 / (np.mean(rr2) + EPS)
                cv2 = np.std(rr2) / (np.mean(rr2) + EPS)
                if (MIN_HR <= hr2 <= MAX_HR) and cv2 <= MAX_CV_RR:
                    return True
    except Exception:
        pass
    return False

def resample1d(x, n=100):
    x = np.asarray(x, float)
    if len(x) < 2: return np.full(n, np.nan)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)

def spectral_feats(x, fs):
    feats = {}
    keys = ["spec_dom_freq", "spec_centroid", "spec_bw", "spec_entropy", "spec_power",
            "spec_p05_3", "spec_p3_8", "spec_ratio", "spec_fund", "spec_h2", "spec_h3",
            "spec_harm", "spec_h3_ratio"]
    if len(x) < 16 or np.std(x) < EPS:
        for k in keys: feats[k] = np.nan
        return feats
    x0 = x - np.mean(x); yf = np.abs(rfft(x0))**2; xf = rfftfreq(len(x0), d=1/fs)
    pt = np.sum(yf) + EPS; df_ = xf[np.argmax(yf[1:]) + 1]
    cent = np.sum(xf * yf) / pt; bw = np.sqrt(np.sum(((xf - cent)**2) * yf) / pt)
    p = yf / pt; ent = -np.sum(p * np.log(p + EPS))
    b1 = (xf >= 0.5) & (xf <= 3.0); b2 = (xf > 3.0) & (xf <= 8.0)
    fi = np.argmin(np.abs(xf - df_))
    h2i = np.argmin(np.abs(xf - 2*df_)) if np.isfinite(df_) else 0
    h3i = np.argmin(np.abs(xf - 3*df_)) if np.isfinite(df_) else 0
    feats.update({"spec_dom_freq": df_, "spec_centroid": cent, "spec_bw": bw,
                  "spec_entropy": ent, "spec_power": pt, "spec_p05_3": np.sum(yf[b1]),
                  "spec_p3_8": np.sum(yf[b2]), "spec_ratio": np.sum(yf[b1]) / (np.sum(yf[b2]) + EPS),
                  "spec_fund": yf[fi], "spec_h2": yf[h2i] if h2i < len(yf) else np.nan,
                  "spec_h3": yf[h3i] if h3i < len(yf) else np.nan,
                  "spec_harm": yf[h2i] / (yf[fi] + EPS) if h2i < len(yf) else np.nan,
                  "spec_h3_ratio": yf[h3i] / (yf[fi] + EPS) if h3i < len(yf) else np.nan})
    return feats

def apg_features(ppg, fs):
    feats = {}
    try:
        d1 = np.gradient(ppg) * fs; d2 = np.gradient(d1) * fs
        peaks_ppg, _ = detect_peaks_ppg(ppg, fs)
        if len(peaks_ppg) < 3: raise ValueError()
        ba_ratios = []; cda_ratios = []; aix_vals = []; ai2_vals = []
        back = max(int(0.45 * fs), 1)
        for i, pk in enumerate(peaks_ppg):
            ss = max(0, pk - back); seg = ppg[ss:pk+1]
            if len(seg) < 3: continue
            foot = ss + np.argmin(seg)
            nxt = peaks_ppg[i+1] if i < len(peaks_ppg) - 1 else min(len(ppg) - 1, pk + int(0.9 * fs))
            if nxt <= foot + 4: continue
            beat_d2 = d2[foot:nxt]; bl = len(beat_d2)
            if bl < 10: continue
            pos_a, _ = find_peaks(beat_d2[:int(bl * 0.40)], prominence=np.std(beat_d2) * 0.1 + EPS)
            if not len(pos_a): continue
            a_idx = pos_a[0]; a_val = beat_d2[a_idx]
            if a_val < EPS: continue
            neg_b, _ = find_peaks(-beat_d2[a_idx:int(bl * 0.60)], prominence=np.std(beat_d2) * 0.05 + EPS)
            if not len(neg_b): continue
            b_val = beat_d2[a_idx + neg_b[0]]
            ba_ratios.append(b_val / (a_val + EPS))
            beat_ppg = ppg[foot:nxt]; p1i = np.argmax(beat_ppg); p1v = beat_ppg[p1i]
            ds = min(p1i + max(int(0.10 * len(beat_ppg)), 2), len(beat_ppg) - 1)
            if ds < len(beat_ppg):
                p2c, _ = find_peaks(beat_ppg[ds:], prominence=np.std(beat_ppg) * 0.05 + EPS)
                if len(p2c):
                    p2v = beat_ppg[ds + p2c[0]]
                    aix_vals.append(float((p2v - p1v) / (p1v - beat_ppg[0] + EPS)))
                    ai2_vals.append(float(p2v / (p1v + EPS)))
        agg_stats(feats, "apg_ba", ba_ratios); agg_stats(feats, "apg_cda", cda_ratios)
        agg_stats(feats, "apg_aix", aix_vals); agg_stats(feats, "apg_ai2", ai2_vals)
        agg_stats(feats, "apg_d2", d2); feats["apg_d2_pos_frac"] = float(np.mean(d2 > 0))
    except Exception:
        for p in ["apg_ba", "apg_cda", "apg_aix", "apg_ai2", "apg_d2"]:
            for s in ["_mean", "_std", "_med", "_min", "_max", "_iqr", "_cv"]: feats[f"{p}{s}"] = np.nan
        feats["apg_d2_pos_frac"] = np.nan
    return feats

def extract_features(ppg, fs):
    feats = {}
    x = np.asarray(ppg, float); xc = x - np.mean(x)
    d1 = np.gradient(x) * fs; d2 = np.gradient(d1) * fs
    for nm, arr in [("sig", x), ("sigc", xc), ("d1", d1), ("d2", d2)]:
        for stat, fn in [("mean", s_mean), ("std", s_std), ("med", s_med), ("min", s_min),
                         ("max", s_max), ("iqr", s_iqr), ("mad", s_mad), ("rms", s_rms),
                         ("skew", s_skew), ("kurt", s_kurt),
                         ("ptp", lambda a: float(np.ptp(a))), ("zcr", zcr)]:
            feats[f"{nm}_{stat}"] = fn(arr)
    feats["d1_pos_frac"] = float(np.mean(d1 > 0))
    feats["d1_pos_area"] = float(np.trapezoid(np.clip(d1, 0, None), dx=1/fs))
    feats["d1_neg_area"] = float(np.trapezoid(np.clip(-d1, 0, None), dx=1/fs))
    feats["d2_pos_area"] = float(np.trapezoid(np.clip(d2, 0, None), dx=1/fs))
    feats["d2_neg_area"] = float(np.trapezoid(np.clip(-d2, 0, None), dx=1/fs))
    feats["d1_energy"] = float(np.mean(d1**2)); feats["d2_energy"] = float(np.mean(d2**2))
    feats["d1_d2_ratio"] = float(np.mean(d1**2) / (np.mean(d2**2) + EPS))
    try:
        env = np.abs(hilbert(xc))
        feats.update({"env_mean": float(np.mean(env)), "env_std": float(np.std(env)),
                      "env_cv": float(np.std(env) / (np.mean(env) + EPS)),
                      "env_range": float(np.max(env) - np.min(env))})
    except:
        for k in ["env_mean", "env_std", "env_cv", "env_range"]: feats[k] = np.nan
    feats.update(spectral_feats(x, fs)); feats.update(apg_features(x, fs))
    peaks, props = detect_peaks_ppg(x, fs); dur = len(x) / fs
    feats["peak_count"] = len(peaks); feats["peak_rate"] = len(peaks) / dur if dur > 0 else np.nan
    agg_stats(feats, "peak_prom", props.get("prominences", np.array([])) if props else [])
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        feats.update({"rr_mean": float(np.mean(rr)), "rr_std": float(np.std(rr)),
                      "rr_med": float(np.median(rr)), "rr_iqr": float(iqr(rr)),
                      "rr_cv": float(np.std(rr) / (np.mean(rr) + EPS)),
                      "hr": float(60.0 / (np.mean(rr) + EPS)),
                      "rmssd": float(np.sqrt(np.mean(np.diff(rr)**2))),
                      "sdnn": float(np.std(rr)), "pnn50": float(np.mean(np.abs(np.diff(rr)) > 0.05))})
        drr = np.diff(rr)
        feats.update({"drr_mean": s_mean(drr), "drr_std": s_std(drr),
                      "drr_abs": s_mean(np.abs(drr)), "drr_rms": s_rms(drr)})
    else:
        for k in ["rr_mean", "rr_std", "rr_med", "rr_iqr", "rr_cv", "hr", "rmssd", "sdnn", "pnn50",
                  "drr_mean", "drr_std", "drr_abs", "drr_rms"]: feats[k] = np.nan
    amps = []; rises = []; decays = []; w25 = []; w50 = []; w75 = []
    bareas = []; sareas = []; dareas = []; sup = []; sdn = []; bdurs = []; sym = []
    nttp = []; nw50 = []; inflection_ratios = []; diastolic_fracs = []
    beat_skews = []; beat_kurts = []; beat_entropies = []; tc = []; tr = []; beats_n = []
    back = max(int(0.45 * fs), 1)
    for i, pk in enumerate(peaks):
        ss = max(0, pk - back); seg = x[ss:pk+1]
        if len(seg) < 3: continue
        foot = ss + np.argmin(seg)
        nxt = peaks[i+1] if i < len(peaks) - 1 else min(len(x) - 1, pk + int(0.9 * fs))
        if nxt <= foot + 2: continue
        beat = x[foot:nxt]
        if len(beat) < 6: continue
        amp = x[pk] - x[foot]; rt = (pk - foot) / fs; dt = (nxt - pk) / fs; bd = (nxt - foot) / fs
        if amp <= 0 or not np.isfinite(rt) or rt <= 0: continue
        amps.append(amp); rises.append(rt); decays.append(dt); bdurs.append(bd)
        sup.append(amp / (rt + EPS)); sdn.append(amp / (dt + EPS))
        ba = float(np.trapezoid(beat, dx=1/fs)); sa = float(np.trapezoid(x[foot:pk+1], dx=1/fs))
        if nxt > pk: da = float(np.trapezoid(x[pk:nxt], dx=1/fs)); dareas.append(da); diastolic_fracs.append(da / (ba + EPS))
        bareas.append(ba); sareas.append(sa); sym.append(rt / (dt + EPS))
        d1b = np.gradient(beat) * fs
        nc, _ = find_peaks(-d1b[int(len(d1b) * 0.3):int(len(d1b) * 0.7)], prominence=np.std(d1b) * 0.1 + EPS)
        if len(nc):
            n0 = int(len(d1b) * 0.3) + nc[0]
            if 0 < n0 < len(beat) - 1:
                as_ = float(np.trapezoid(beat[:n0], dx=1/fs)); ad_ = float(np.trapezoid(beat[n0:], dx=1/fs))
                inflection_ratios.append(as_ / (as_ + ad_ + EPS))
        b0 = beat - beat[0]; sc_ = np.max(b0) - np.min(b0)
        if sc_ > EPS:
            bn = b0 / sc_; beat_skews.append(float(skew(bn)) if len(bn) >= 3 else np.nan)
            beat_kurts.append(float(kurtosis(bn)) if len(bn) >= 4 else np.nan)
            h, _ = np.histogram(bn, bins=10, density=True); h = h[h > 0]
            beat_entropies.append(float(-np.sum(h * np.log(h + EPS))))
            bnr = resample1d(bn, 100); beats_n.append(bnr)
            nttp.append(float(np.argmax(bnr) / 99.0))
            i50 = np.where(bnr >= 0.5 * np.max(bnr))[0]
            if len(i50) > 1: nw50.append(float((i50[-1] - i50[0]) / 99.0))
        for frac, store in [(0.25, w25), (0.5, w50), (0.75, w75)]:
            lv = x[foot] + frac * amp; li = np.where(x[foot:pk+1] >= lv)[0]
            if not len(li): continue
            ri = np.where(x[pk:nxt] <= lv)[0]
            if not len(ri): continue
            if pk + ri[0] > foot + li[0]: store.append((pk + ri[0] - foot - li[0]) / fs)
    for nm, vals in [("beat_amp", amps), ("beat_rise", rises), ("beat_decay", decays),
                     ("beat_dur", bdurs), ("beat_w25", w25), ("beat_w50", w50), ("beat_w75", w75),
                     ("beat_area", bareas), ("beat_sa", sareas), ("beat_da", dareas),
                     ("beat_sup", sup), ("beat_sdn", sdn), ("beat_sym", sym), ("nttp", nttp),
                     ("nw50", nw50), ("beat_ipar", inflection_ratios), ("beat_dfrac", diastolic_fracs),
                     ("beat_skew", beat_skews), ("beat_kurt", beat_kurts), ("beat_ent", beat_entropies)]:
        agg_stats(feats, nm, vals)
    feats["beat_ar"] = feats.get("beat_sa_mean", np.nan) / (feats.get("beat_da_mean", np.nan) + EPS)
    feats["beat_wr2550"] = feats.get("beat_w25_mean", np.nan) / (feats.get("beat_w50_mean", np.nan) + EPS)
    feats["beat_wr5075"] = feats.get("beat_w50_mean", np.nan) / (feats.get("beat_w75_mean", np.nan) + EPS)
    feats["beat_ud"] = feats.get("beat_sup_mean", np.nan) / (feats.get("beat_sdn_mean", np.nan) + EPS)
    feats["beat_rd"] = feats.get("beat_rise_mean", np.nan) / (feats.get("beat_decay_mean", np.nan) + EPS)
    feats["beat_stiffness_proxy"] = feats.get("beat_w50_mean", np.nan) * feats.get("rr_mean", np.nan) / (feats.get("beat_amp_mean", np.nan) + EPS)
    if len(beats_n) >= 2:
        bna = np.asarray(beats_n, float)
        # Fix for nanmedian with all-NaN slice warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tpl = np.nanmedian(bna, axis=0)
        for b in bna: tc.append(s_corr(b, tpl)); tr.append(float(np.sqrt(np.nanmean((b - tpl)**2))))
        agg_stats(feats, "tmpl_corr", tc); agg_stats(feats, "tmpl_rmse", tr)
        feats.update({"tmpl_area": float(np.trapezoid(tpl, dx=1/99)), "tmpl_max": float(np.max(tpl)),
                      "tmpl_ptp": float(np.ptp(tpl)), "tmpl_ttp": float(np.argmax(tpl) / 99.0)})
    else:
        for k in ["tmpl_corr_mean", "tmpl_corr_std", "tmpl_corr_med", "tmpl_corr_min", "tmpl_corr_max",
                  "tmpl_corr_iqr", "tmpl_corr_cv", "tmpl_rmse_mean", "tmpl_rmse_std", "tmpl_rmse_med",
                  "tmpl_rmse_min", "tmpl_rmse_max", "tmpl_rmse_iqr", "tmpl_rmse_cv",
                  "tmpl_area", "tmpl_max", "tmpl_ptp", "tmpl_ttp"]: feats[k] = np.nan
    return feats

def extract_windows(ppg, fs):
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)
    if win < 2 or len(ppg) < win:
        # Fallback if recording is too short: return whole signal if possible
        if len(ppg) >= fs * 5 and window_ok(ppg, fs):
             return [extract_features(ppg, fs)], 1
        return [], 0
    windows = [ppg[s:s+win] for s in range(0, len(ppg) - win + 1, step)]
    good = [extract_features(w, fs) for w in windows if window_ok(w, fs)]
    return good, len(windows)

def aggregate_windows(feat_list):
    if not feat_list: return {}
    df = pd.DataFrame(feat_list)
    out = {}
    for col in df.columns:
        vals = df[col].astype(float).dropna().values
        out[col] = float(trim_mean(vals, 0.1)) if len(vals) >= 4 else (float(np.mean(vals)) if len(vals) else np.nan)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────
def load_models(model_dir: str = MODEL_DIR) -> dict:
    """
    Load saved models into a structure expected by prediction logic.
    Returns:
       dict with "abs" and "delta" sub-dicts containing full pipeline dict objects.
    """
    trained = {"abs": {}, "delta": {}}
    names = {
        "abs_sbp":   ("abs", "sbp"),
        "abs_dbp":   ("abs", "dbp"),
        "delta_sbp_delta": ("delta", "sbp_delta"),
        "delta_dbp_delta": ("delta", "dbp_delta"),
    }
    for file_base, (mode, target) in names.items():
        path = os.path.join(model_dir, f"{file_base}.joblib")
        if os.path.exists(path):
            obj = joblib.load(path)
            # Patch SimpleImputer for sklearn 1.6.1 -> 1.8.0 mismatch
            if "imp" in obj:
                imp = obj["imp"]
                if not hasattr(imp, "_fill_dtype") and hasattr(imp, "_fit_dtype"):
                    imp._fill_dtype = imp._fit_dtype
                if not hasattr(imp, "keep_empty_features"):
                    imp.keep_empty_features = False
            trained[mode][target] = obj
    if not trained["abs"]:
       raise FileNotFoundError(f"No abs models found in {model_dir}")
    return trained

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def _apply_model(obj, feat_dict):
    all_cols = obj["all_cols"]
    sel_idx = obj["sel_idx"]
    X_df = pd.DataFrame([feat_dict]).reindex(columns=all_cols, fill_value=np.nan)
    X_imp = obj["imp"].transform(X_df.values.astype(float))
    X_s = obj["scaler"].transform(X_imp)[:, sel_idx]
    if hasattr(obj["stack"], "predict_with_uncertainty"):
        pd_arr, unc_arr = obj["stack"].predict_with_uncertainty(X_s)
        return float(pd_arr[0]), float(unc_arr[0])
    return float(obj["stack"].predict(X_s)[0]), 0.0

def predict(
    ir: np.ndarray,
    red: np.ndarray,
    models: dict,
    cal_obj: dict | None = None,
    fs: float = FS,
) -> dict:
    """
    Run end-to-end BP prediction on the live buffer.
    We use the IR signal for primary extraction as per pipeline logic.
    """
    ppg = normalise(bandpass(ir, fs))
    good, n_total = extract_windows(ppg, fs)
    if not good:
        raise ValueError("No valid PPG signal detected. Ensure the sensor is worn correctly and you have at least 20 seconds of stable data.")

    trained = models
    ref_feat = cal_obj.get("ref_feat", {}) if cal_obj else {}
    
    results = {}
    for bp_key in ["sbp", "dbp"]:
        delta_key = f"{bp_key}_delta"
        ref_bp = cal_obj.get(f"ref_{bp_key}") if cal_obj else None
        
        cal_info = cal_obj["cal"].get(bp_key, {"method": "none"}) if cal_obj and "cal" in cal_obj else {"method": "none"}
        per_win = []
        per_unc = []
        per_raw = []

        for wf in good:
            pred_final = None
            unc = 0.0
            pa = None # Absolute prediction
            
            # Predict robust absolute first
            if bp_key in trained.get("abs", {}):
                pa, ua = _apply_model(trained["abs"][bp_key], wf)
                per_raw.append(pa)
                if cal_info["method"] == "offset_abs":
                    pred_final = pa + cal_info.get("offset_a", 0.0)
                else:
                    pred_final = pa
                unc = ua

            # Delta calibration if possible
            if delta_key in trained.get("delta", {}) and ref_feat and ref_bp is not None:
                feat_d = {}
                for k in ref_feat:
                    rv = ref_feat.get(k, np.nan)
                    sv = wf.get(k, np.nan)
                    feat_d[f"diff__{k}"] = sv - rv if (np.isfinite(rv) and np.isfinite(sv)) else np.nan
                    feat_d[f"ratio__{k}"] = sv / (rv + EPS) if (np.isfinite(rv) and np.isfinite(sv)) else np.nan
                    feat_d[f"raw__{k}"] = sv
                
                pd_val, ud_val = _apply_model(trained["delta"][delta_key], feat_d)
                
                m = cal_info["method"]
                if m == "ridge_delta" and "ridge_d" in cal_info:
                    pd_cal = float(cal_info["ridge_d"].predict([[pd_val]])[0])
                elif m == "offset_delta":
                    pd_cal = pd_val + cal_info.get("offset_d", 0.0)
                else:
                    pd_cal = pd_val
                
                pred_final = ref_bp + pd_cal
                unc = ud_val

            if pred_final is not None:
                per_win.append(pred_final)
                per_unc.append(unc)
                
        if not per_win:
             raise ValueError("Failed to run prediction on windows")

        preds = np.array(per_win)
        pred = float(np.mean(preds))
        std = float(np.std(preds))
        raws = np.array(per_raw) if per_raw else preds
        raw_pred = float(np.mean(raws))
        
        unc_mean = float(np.mean(per_unc)) if per_unc else std
        half_ci = max(1.96 * std, unc_mean)
        
        results[bp_key] = {
            "pred": round(pred, 1),
            "raw": round(raw_pred, 1),
            "std": std,
            "ci_low": round(pred - half_ci, 1),
            "ci_high": round(pred + half_ci, 1),
        }
        
    return {
        "sbp": results["sbp"]["pred"],
        "dbp": results["dbp"]["pred"],
        "sbp_raw": results["sbp"]["raw"],
        "dbp_raw": results["dbp"]["raw"],
        "ci_sbp_low": results["sbp"]["ci_low"],
        "ci_sbp_high": results["sbp"]["ci_high"],
        "ci_dbp_low": results["dbp"]["ci_low"],
        "ci_dbp_high": results["dbp"]["ci_high"],
        "n_windows": len(good),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import RidgeCV

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
MIN_CAL_FOR_RIDGE = 3

def _csv_to_arrays(csv_data: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(io.StringIO(csv_data))
    cols = [str(c).lower().strip() for c in df.columns]
    df.columns = cols
    
    ir_col = next((c for c in ["ir", "ppg", "ppg_wrist_ir", "ppg_wrist_g", "signal", "value"] if c in cols), cols[0])
    red_col = next((c for c in ["red", "ppg_red", "r"] if c in cols), ir_col)
    
    ir = df[ir_col].to_numpy(dtype=float)
    red = df[red_col].to_numpy(dtype=float)
    return ir, red

def run_calibration(
    sessions: list[dict],
    models: dict,
    fs: float = FS,
) -> dict:
    if len(sessions) < 1:
        raise ValueError("At least 1 calibration session required.")

    cal_records = []
    for idx, s in enumerate(sessions):
        ir, red = _csv_to_arrays(s["csv_data"])
        ppg = normalise(bandpass(ir, fs))
        good, n_total = extract_windows(ppg, fs)
        if not good:
            if len(ppg) > fs * 5: good = [extract_features(ppg, fs)]
            else: continue
            
        fd = aggregate_windows(good)
        cal_records.append({
            "feat": fd,
            "sbp": float(s["sbp"]),
            "dbp": float(s["dbp"])
        })

    if not cal_records:
        raise RuntimeError("No usable cal sessions extracted.")

    trained = models
    cal_obj = {"records": cal_records, "cal": {}}
    ref = cal_records[0]

    for bp_key in ["sbp", "dbp"]:
        delta_key = f"{bp_key}_delta"
        y_true_d = []; y_pred_d = []; y_true_a = []; y_pred_a = []
        
        for rec in cal_records[1:]:
            true_delta = rec[bp_key] - ref[bp_key]
            feat_d = {}
            for k in ref["feat"]:
                rv = ref["feat"].get(k, np.nan); sv = rec["feat"].get(k, np.nan)
                feat_d[f"diff__{k}"] = sv - rv if (np.isfinite(rv) and np.isfinite(sv)) else np.nan
                feat_d[f"ratio__{k}"] = sv / (rv + EPS) if (np.isfinite(rv) and np.isfinite(sv)) else np.nan
                feat_d[f"raw__{k}"] = sv
                
            if delta_key in trained.get("delta", {}):
                pred_d, _ = _apply_model(trained["delta"][delta_key], feat_d)
                y_pred_d.append(pred_d)
                y_true_d.append(true_delta)
                
            if bp_key in trained.get("abs", {}):
                pred_a, _ = _apply_model(trained["abs"][bp_key], rec["feat"])
                y_pred_a.append(pred_a)
                y_true_a.append(float(rec[bp_key]))

        n_d = len(y_true_d)
        if n_d >= MIN_CAL_FOR_RIDGE:
            y_t = np.array(y_true_d); y_p = np.array(y_pred_d)
            # Only fit Ridge if there is meaningful variance to avoid astronomical slopes
            if np.std(y_p) > 0.5:
                rc = RidgeCV(alphas=RIDGE_ALPHAS, cv=n_d)
                rc.fit(y_p.reshape(-1, 1), y_t)
                if abs(float(rc.coef_[0])) > 3.0:
                    offset_d = float(np.mean(y_t - y_p))
                    cal_obj["cal"][bp_key] = {"method": "offset_delta", "offset_d": offset_d, "n_d": n_d}
                else:
                    cal_obj["cal"][bp_key] = {"method": "ridge_delta", "ridge_d": rc, "n_d": n_d}
            else:
                offset_d = float(np.mean(y_t - y_p))
                cal_obj["cal"][bp_key] = {"method": "offset_delta", "offset_d": offset_d, "n_d": n_d}
        elif n_d > 0:
            offset_d = float(np.mean(np.array(y_true_d) - np.array(y_pred_d)))
            cal_obj["cal"][bp_key] = {"method": "offset_delta", "offset_d": offset_d, "n_d": n_d}
        elif len(y_true_a) > 0:
            offset_a = float(np.mean(np.array(y_true_a) - np.array(y_pred_a)))
            cal_obj["cal"][bp_key] = {"method": "offset_abs", "offset_a": offset_a}
        else:
            cal_obj["cal"][bp_key] = {"method": "none"}

        cal_obj[f"ref_{bp_key}"] = ref[bp_key]
        cal_obj["ref_feat"] = ref["feat"]

    cal_obj["n_sessions"] = len(cal_records)
    # Simple bias metrics for UI
    if "offset_abs" in cal_obj["cal"].get("sbp", {}).get("method", ""):
       cal_obj["bias_sbp"] = cal_obj["cal"]["sbp"].get("offset_a", 0)
       cal_obj["bias_dbp"] = cal_obj["cal"]["dbp"].get("offset_a", 0)
    else:
       cal_obj["bias_sbp"] = cal_obj["cal"].get("sbp", {}).get("offset_d", 0)
       cal_obj["bias_dbp"] = cal_obj["cal"].get("dbp", {}).get("offset_d", 0)

    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    joblib.dump(cal_obj, CALIBRATION_PATH)
    return cal_obj

def load_calibration() -> dict | None:
    if os.path.exists(CALIBRATION_PATH):
        return joblib.load(CALIBRATION_PATH)
    return None
