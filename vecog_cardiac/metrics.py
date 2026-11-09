"""Ground-truth metrics: residual error and preservation of motor features."""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .signals import FS, bandpass

BANDS = {"beta": (13.0, 30.0), "hg": (70.0, 150.0)}


def error_db(Xhat: np.ndarray, Xref: np.ndarray, band: tuple[float, float] | None = None) -> float:
    """20*log10(rms(Xhat - Xref) / rms(Xref)), optionally inside a band. Lower is better;
    -inf means a perfect recovery."""
    d, r = Xhat - Xref, Xref
    if band:
        d, r = bandpass(d, *band), bandpass(r, *band)
    return float(20 * np.log10(np.sqrt(np.mean(d**2)) / np.sqrt(np.mean(r**2))))


def band_power_trials(X: np.ndarray, starts: np.ndarray, length: int, band: tuple[float, float]) -> np.ndarray:
    """log mean power (Hilbert envelope squared) per trial and channel -> (trials, channels)."""
    env = np.abs(signal.hilbert(bandpass(X, *band), axis=-1)) ** 2
    idx = starts[:, None] + np.arange(length)[None, :]
    return np.log(env[:, idx].mean(axis=2).T + 1e-12)


def motor_features(X: np.ndarray, t_on: np.ndarray, fs: int = FS) -> dict:
    """Move windows: 0.5-3.0 s after cue. Rest windows: 2.5-0.5 s before cue."""
    move = t_on + int(0.5 * fs)
    rest = t_on - int(2.5 * fs)
    ok = rest >= 0
    out = {}
    for name, band in BANDS.items():
        m = band_power_trials(X, move[ok], int(2.5 * fs), band)
        r = band_power_trials(X, rest[ok], int(2.0 * fs), band)
        out[name] = (m, r)
    out["ok"] = ok
    return out


def activation_map(m: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-channel signed r^2 of move vs rest log power."""
    y = np.r_[np.ones(len(m)), np.zeros(len(r))]
    Z = np.r_[m, r]
    Zc = Z - Z.mean(0)
    yc = y - y.mean()
    rr = (Zc * yc[:, None]).sum(0) / (np.linalg.norm(Zc, axis=0) * np.linalg.norm(yc) + 1e-12)
    return np.sign(rr) * rr**2


def decode_accuracy(feats: dict, stim: np.ndarray, seed: int = 0, bands=("beta", "hg")) -> float:
    """Hand vs tongue, shrinkage LDA on move-window log power in the given bands, 5-fold CV."""
    ok = feats["ok"]
    Z = np.concatenate([feats[b][0] for b in bands], axis=1)
    y = stim[ok]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, Z, y, cv=cv).mean())


def map_fidelity(feats: dict, feats_ref: dict) -> dict:
    out = {}
    for b in BANDS:
        a = activation_map(*feats[b])
        r = activation_map(*feats_ref[b])
        out[f"map_corr_{b}"] = float(np.corrcoef(a, r)[0, 1])
    return out
