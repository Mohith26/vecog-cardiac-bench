"""Unit tests on synthetic data with known ground truth (no downloads needed)."""
import numpy as np
import pytest

from vecog_cardiac.methods import (
    car,
    cardiac_score,
    detect_rpeaks,
    ica_remove,
    rpeaks_from_data,
    template_removal,
)
from vecog_cardiac.metrics import activation_map, error_db
from vecog_cardiac.signals import FS, bandpass, make_artifact

DUR = 60  # seconds


def synthetic_ecg(n, hr_bpm=72.0, seed=0):
    """Two-lead toy ECG: Gaussian QRS + T wave on a slightly jittered rhythm."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    ecg = np.zeros((2, n))
    beat = 0.0
    peaks = []
    while beat < t[-1]:
        peaks.append(int(beat * FS))
        beat += 60.0 / hr_bpm * (1 + 0.03 * rng.standard_normal())
    peaks = np.array([p for p in peaks if 300 < p < n - 600])
    for p in peaks:
        qrs = np.exp(-0.5 * ((t - t[p]) / 0.012) ** 2)
        tw = 0.3 * np.exp(-0.5 * ((t - t[p] - 0.25) / 0.04) ** 2)
        ecg[0] += qrs + tw
        ecg[1] += -0.6 * qrs + 0.5 * tw
    ecg -= ecg.mean(1, keepdims=True)
    return ecg / ecg.std(1, keepdims=True), peaks


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = DUR * FS
    neural = bandpass(rng.standard_normal((16, n)), 1, 200) * 10
    ecg, peaks = synthetic_ecg(n)
    art = make_artifact(ecg, 16, float(np.sqrt(np.mean(neural**2))), 10.0, 0.5, rng)
    return neural, ecg, peaks, art


def test_detect_rpeaks_finds_beats_either_polarity(data):
    _, ecg, peaks, _ = data
    for sign in (1, -1):
        found = detect_rpeaks(sign * ecg[0])
        assert abs(len(found) - len(peaks)) <= 1
        d = np.abs(found[:, None] - peaks[None, :]).min(1)
        assert np.median(d) <= 5  # ms at 1 kHz


def test_cardiac_score_separates_ecg_from_noise(data):
    neural, ecg, _, _ = data
    assert cardiac_score(ecg[0]) > 0.8
    assert cardiac_score(neural[0]) < 0.3


def test_make_artifact_hits_requested_sar(data):
    neural, ecg, _, _ = data
    rms = float(np.sqrt(np.mean(neural**2)))
    for sar in (0.0, 10.0, 20.0):
        art = make_artifact(ecg, 16, rms, sar, 0.5, np.random.default_rng(0))
        got = 20 * np.log10(np.sqrt(np.mean(art**2)) / rms)
        assert got == pytest.approx(sar, abs=1e-6)


def test_common_mode_artifact_is_removed_exactly_by_car(data):
    neural, ecg, _, _ = data
    rms = float(np.sqrt(np.mean(neural**2)))
    art = make_artifact(ecg, 16, rms, 20.0, 0.0, np.random.default_rng(0))
    assert np.allclose(car(neural + art), car(neural))


def test_each_method_reduces_error(data):
    neural, ecg, peaks, art = data
    X = neural + art
    base = error_db(X, neural, (1, 40))
    assert error_db(template_removal(X, peaks), neural, (1, 40)) < base - 10
    rp = rpeaks_from_data(X)
    assert error_db(template_removal(X, rp), neural, (1, 40)) < base - 10
    Xi, info = ica_remove(X, n_components=16, oracle_ecg=ecg)
    assert info["selected"], info
    assert set(info["selected"]) & set(info["oracle"])
    assert error_db(Xi, neural, (1, 40)) < base - 10


def test_lowpassed_correction_leaves_high_gamma_untouched(data):
    neural, ecg, peaks, art = data
    art_lp = bandpass(art, None, 45.0)
    X = neural + art_lp
    hg = (70.0, 150.0)
    plain = error_db(template_removal(X, peaks), neural, hg)
    lp = error_db(template_removal(X, peaks, correction_lowpass=50.0), neural, hg)
    assert lp < plain - 10


def test_activation_map_sign_and_range():
    rng = np.random.default_rng(0)
    m = rng.normal(1.0, 0.1, (40, 3))
    r = rng.normal(0.0, 0.1, (40, 3))
    m[:, 2] = r[:, 2]
    a = activation_map(m, r)
    assert a[0] > 0.8 and a[1] > 0.8 and abs(a[2]) < 0.2
    assert np.all(np.abs(a) <= 1)
