"""Data loading, filtering and the semi-synthetic cardiac artifact model."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy import signal

FS = 1000  # Hz, sampling rate of the Miller motor ECoG data

ECG_RECORDS = ["100", "101", "103", "105", "106", "112", "113"]  # MIT-BIH, one per subject


def bandpass(x: np.ndarray, lo: float | None, hi: float | None, fs: int = FS, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth filter along the last axis (sos form)."""
    if lo and hi:
        sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    elif lo:
        sos = signal.butter(order, lo, btype="highpass", fs=fs, output="sos")
    else:
        sos = signal.butter(order, hi, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def notch_line(x: np.ndarray, fs: int = FS, base: float = 60.0) -> np.ndarray:
    for f0 in (base, 2 * base, 3 * base):
        if f0 < fs / 2:
            b, a = signal.iirnotch(f0, 30.0, fs)
            x = signal.filtfilt(b, a, x, axis=-1)
    return x


@dataclass
class Recording:
    subject: int
    X: np.ndarray  # (n_channels, n_samples) microvolts, good channels only
    t_on: np.ndarray
    stim_id: np.ndarray
    good: np.ndarray  # indices of kept channels in the original montage


def load_ecog(npz_path: str, subject: int, block: int = 0) -> Recording:
    """Miller motor_imagery.npz (Neuromatch copy). block 0 = real movement."""
    dat = np.load(npz_path, allow_pickle=True)["dat"][subject][block]
    scale = np.asarray(dat["scale_uv"], dtype=float)
    good = np.where(scale > 5.0)[0]  # channels with scale ~0.9 uV are disconnected
    X = dat["V"].astype(np.float64).T[good] * scale[good, None]
    X = X - X.mean(axis=1, keepdims=True)
    X = bandpass(X, 0.5, None)
    X = notch_line(X)
    return Recording(subject, X, np.asarray(dat["t_on"]), np.asarray(dat["stim_id"]), good)


def load_ecg(record: str, n_samples: int, cache_dir: str, fs: int = FS, lowpass: float | None = 45.0) -> np.ndarray:
    """Two-lead real ECG from PhysioNet MIT-BIH, resampled to fs, unit RMS per lead.

    lowpass (default 45 Hz) strips the surface recording's EMG, quantization noise and mains
    hum, which are not beat-locked and would not reach an intravascular electrode the same way.
    Pass lowpass=None to keep the full 0.1-100 Hz MIT-BIH bandwidth (sensitivity check)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"mitdb_{record}.npy")
    if os.path.exists(path):
        raw = np.load(path)
    else:
        import wfdb  # optional dependency only needed for the first download

        rec = wfdb.rdrecord(record, pn_dir="mitdb")
        raw = rec.p_signal.T.astype(np.float64)
        np.save(path, raw)
    up = signal.resample_poly(raw, fs, 360, axis=1)
    up = bandpass(up, 0.5, lowpass) if lowpass else bandpass(up, 0.5, None)
    if up.shape[1] < n_samples:
        reps = int(np.ceil(n_samples / up.shape[1]))
        up = np.tile(up, reps)
    up = up[:, :n_samples]
    return up / up.std(axis=1, keepdims=True)


def make_artifact(
    ecg: np.ndarray,
    n_channels: int,
    neural_rms: float,
    sar_db: float,
    heterogeneity: float,
    rng: np.random.Generator,
    max_lag_ms: float = 4.0,
    fs: int = FS,
) -> np.ndarray:
    """Cardiac artifact projected onto every channel.

    heterogeneity = 0: one common waveform on every channel (pure common mode).
    heterogeneity > 0: per-channel log-normal gains (sigma = heterogeneity), a per-channel
    mix of the two ECG leads (different QRS morphology per channel) and a small per-channel lag.
    The overall artifact RMS is set so that 20*log10(artifact_rms / neural_rms) = sar_db.
    """
    n = ecg.shape[1]
    if heterogeneity == 0:
        art = np.tile(ecg[0], (n_channels, 1))
    else:
        gains = rng.lognormal(0.0, heterogeneity, n_channels)
        angles = rng.normal(0.0, heterogeneity * np.pi / 3, n_channels)
        lags = rng.integers(0, int(max_lag_ms * fs / 1000) + 1, n_channels)
        art = np.empty((n_channels, n))
        for c in range(n_channels):
            w = np.cos(angles[c]) * ecg[0] + np.sin(angles[c]) * ecg[1]
            art[c] = gains[c] * np.roll(w, lags[c])
    art *= neural_rms * 10 ** (sar_db / 20) / np.sqrt(np.mean(art**2))
    return art
