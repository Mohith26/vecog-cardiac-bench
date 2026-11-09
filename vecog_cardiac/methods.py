"""Cardiac artifact removal methods compared in Liyanage et al. 2025 (CAR, ICA with automated
ECG component selection, template-based removal) plus a band-limited ICA variant in the spirit
of Feldman et al. 2026. These are independent reimplementations from the papers' descriptions,
not the authors' code."""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA

from .signals import FS, bandpass

# ----------------------------------------------------------------------------- R-peak detection


def detect_rpeaks(x: np.ndarray, fs: int = FS) -> np.ndarray:
    """Polarity-agnostic QRS detector (Pan-Tompkins style) on a single 1-D signal."""
    y = bandpass(x, 5.0, 30.0, fs)
    e = np.gradient(y) ** 2
    win = int(0.12 * fs)
    e = np.convolve(e, np.ones(win) / win, mode="same")
    thr = 0.3 * np.percentile(e, 99.5)
    peaks, _ = signal.find_peaks(e, height=thr, distance=int(0.33 * fs))
    # refine to the largest absolute deflection of the band-passed signal within +-60 ms
    half = int(0.06 * fs)
    out = []
    for p in peaks:
        a, b = max(0, p - half), min(len(y), p + half)
        out.append(a + int(np.argmax(np.abs(y[a:b]))))
    return np.unique(np.asarray(out, dtype=int))


def cardiac_score(s: np.ndarray, fs: int = FS, rr_gate: bool = True) -> float:
    """How heartbeat-like a 1-D source is, in [0, 1].

    Requires a plausible heart rate (40-180 bpm) and regular rhythm (RR coefficient of
    variation < 0.25); the score is then the mean correlation of individual beats with the
    median beat template (morphology consistency). Fixed a priori; not tuned on ground truth.
    rr_gate=False drops the rhythm-regularity requirement (heart-rate range still applies),
    which matters for patients with frequent ectopic beats.
    """
    r = detect_rpeaks(s, fs)
    dur_min = len(s) / fs / 60
    if len(r) < 10:
        return 0.0
    hr = len(r) / dur_min
    rr = np.diff(r) / fs
    if not (40 <= hr <= 180) or (rr_gate and rr.std() / rr.mean() > 0.25):
        return 0.0
    pre, post = int(0.2 * fs), int(0.4 * fs)
    r = r[(r > pre) & (r < len(s) - post)]
    ep = np.stack([s[p - pre : p + post] for p in r])
    tmpl = np.median(ep, axis=0)
    ep_c = ep - ep.mean(axis=1, keepdims=True)
    t_c = tmpl - tmpl.mean()
    num = ep_c @ t_c
    den = np.linalg.norm(ep_c, axis=1) * np.linalg.norm(t_c) + 1e-12
    return float(np.clip(np.mean(num / den), 0.0, 1.0))


# ----------------------------------------------------------------------------- methods


def car(X: np.ndarray) -> np.ndarray:
    """Common average reference."""
    return X - X.mean(axis=0, keepdims=True)


def template_removal(
    X: np.ndarray, rpeaks: np.ndarray, n_beats: int = 30, correction_lowpass: float | None = None, fs: int = FS
) -> np.ndarray:
    """Template-based removal: per channel, subtract a moving median template of the
    neighbouring n_beats beats, least-squares scaled per beat, with a Tukey taper at the window
    edges. correction_lowpass (Hz) low-passes the subtracted estimate so that the method cannot
    inject template noise into high-frequency bands the artifact does not occupy."""
    pre, post = int(0.25 * fs), int(0.45 * fs)
    L = pre + post
    taper = signal.windows.tukey(L, 0.2)
    r = rpeaks[(rpeaks > pre) & (rpeaks < X.shape[1] - post)]
    out = X.copy()
    if len(r) < 3:
        return out
    idx = r[:, None] + np.arange(-pre, post)[None, :]
    for c in range(X.shape[0]):
        ep = X[c][idx]  # (beats, L)
        ep = ep - np.median(ep, axis=1, keepdims=True)
        for k in range(len(r)):
            lo = max(0, k - n_beats // 2)
            hi = min(len(r), lo + n_beats)
            lo = max(0, hi - n_beats)
            tmpl = np.median(ep[lo:hi], axis=0)
            scale = float(ep[k] @ tmpl / (tmpl @ tmpl + 1e-12))
            out[c, idx[k]] -= np.clip(scale, 0.0, 2.0) * tmpl * taper
    if correction_lowpass:
        return X - bandpass(X - out, None, correction_lowpass, fs)
    return out


def rpeaks_from_data(X: np.ndarray, fs: int = FS) -> np.ndarray:
    """Without an ECG channel: R-peaks from the 5-30 Hz first principal component."""
    Y = bandpass(X, 5.0, 30.0, fs)
    Y = Y - Y.mean(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(Y[:, :: 4] @ Y[:, :: 4].T)
    return detect_rpeaks(u[:, 0] @ Y, fs)


def ica_remove(
    X: np.ndarray,
    fit_band: tuple[float, float] | None = None,
    n_components: int | None = None,
    threshold: float = 0.5,
    max_remove: int = 4,
    oracle_ecg: np.ndarray | None = None,
    seed: int = 0,
    correction_lowpass: float | None = None,
    rr_gate: bool = True,
    fs: int = FS,
) -> tuple[np.ndarray, dict]:
    """FastICA, remove components selected automatically by cardiac_score >= threshold.

    fit_band=None fits ICA on the broadband data (as in Liyanage et al.); a band such as
    (2, 40) fits the unmixing on band-limited data and applies it to the broadband data
    (band-limited ICA, after Feldman et al.). correction_lowpass (Hz) low-passes the removed
    part (A_sel @ S_sel); the result is returned in info["lowpassed"]. If oracle_ecg is given, the components a
    ground-truth oracle would pick (|corr| with either ECG lead >= 0.3) are also reported.
    """
    n_ch = X.shape[0]
    n_components = n_components or min(n_ch, 40)
    Xf = bandpass(X, *fit_band, fs) if fit_band else X
    mu = X.mean(axis=1, keepdims=True)
    ica = FastICA(n_components=n_components, whiten="unit-variance", max_iter=600, tol=1e-4, random_state=seed)
    ica.fit((Xf[:, ::2] - Xf[:, ::2].mean(axis=1, keepdims=True)).T)
    W, A = ica.components_, ica.mixing_  # (k, ch), (ch, k)
    S = W @ (X - mu)
    S_fit = W @ (Xf - Xf.mean(axis=1, keepdims=True))
    scores = np.array([cardiac_score(S_fit[k], fs, rr_gate) for k in range(S.shape[0])])
    order = np.argsort(-scores)
    sel = [int(k) for k in order[:max_remove] if scores[k] >= threshold]
    info = {"n_components": int(S.shape[0]), "selected": sel, "scores": scores[order[:6]].round(3).tolist()}
    if oracle_ecg is not None:
        e = bandpass(oracle_ecg, 1.0, 40.0, fs)
        sb = bandpass(S, 1.0, 40.0, fs)
        sb = (sb - sb.mean(1, keepdims=True)) / (sb.std(1, keepdims=True) + 1e-12)
        e = (e - e.mean(1, keepdims=True)) / e.std(1, keepdims=True)
        corr = np.abs(sb @ e.T / sb.shape[1]).max(axis=1)
        info["oracle"] = [int(k) for k in np.where(corr >= 0.3)[0]]
    removed = A[:, sel] @ S[sel] if sel else np.zeros_like(X)
    if correction_lowpass:
        info["lowpassed"] = X - bandpass(removed, None, correction_lowpass, fs)
    return X - removed, info
