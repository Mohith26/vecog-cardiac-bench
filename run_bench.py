"""Run the semi-synthetic cardiac artifact benchmark. Resumable: one JSON per subject.

usage: python run_bench.py [--subjects 0 1 ...] [--sar 0 10 20] [--het 0 0.5]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from vecog_cardiac.methods import car, detect_rpeaks, ica_remove, rpeaks_from_data, template_removal
from vecog_cardiac.metrics import BANDS, decode_accuracy, error_db, map_fidelity, motor_features
from vecog_cardiac.signals import ECG_RECORDS, load_ecg, load_ecog, make_artifact

HERE = os.path.dirname(os.path.abspath(__file__))


def peak_f1(found: np.ndarray, true: np.ndarray, tol: int = 50) -> float:
    if len(found) == 0 or len(true) == 0:
        return 0.0
    j = np.searchsorted(true, found)
    d = np.minimum(np.abs(found - true[np.clip(j - 1, 0, len(true) - 1)]), np.abs(found - true[np.clip(j, 0, len(true) - 1)]))
    tp = int((d <= tol).sum())
    p, r = tp / len(found), tp / len(true)
    return 0.0 if tp == 0 else 2 * p * r / (p + r)


def evaluate(Xhat, Xref, feats_ref, rec, extra=None):
    f = motor_features(Xhat, rec.t_on)
    row = {
        "err_db": error_db(Xhat, Xref),
        **{f"err_db_{b}": error_db(Xhat, Xref, band) for b, band in BANDS.items()},
        **map_fidelity(f, feats_ref),
        "decode_acc": decode_accuracy(f, rec.stim_id),
        "decode_acc_beta": decode_accuracy(f, rec.stim_id, bands=("beta",)),
    }
    if extra:
        row.update(extra)
    return row


def run_subject(s: int, sars, hets, out_dir: str, ecg_lowpass: float | None = 45.0, lag_ms: float = 4.0) -> None:
    path = os.path.join(out_dir, f"sub{s}.json")
    done = json.load(open(path)) if os.path.exists(path) else {"subject": s, "conditions": []}
    have = {(c["sar_db"], c["het"], c.get("lag_ms", 4.0)) for c in done["conditions"]}
    rec = load_ecog(os.path.join(HERE, "data", "motor_imagery.npz"), s)
    X0 = rec.X
    ecg = load_ecg(ECG_RECORDS[s], X0.shape[1], os.path.join(HERE, "data"), lowpass=ecg_lowpass)
    done["ecg_lowpass"] = ecg_lowpass
    true_r = detect_rpeaks(ecg[0])
    feats_clean = motor_features(X0, rec.t_on)
    feats_clean_car = motor_features(car(X0), rec.t_on)
    done["n_channels"] = int(X0.shape[0])
    done["clean_decode_acc"] = decode_accuracy(feats_clean, rec.stim_id)
    done["clean_car_decode_acc"] = decode_accuracy(feats_clean_car, rec.stim_id)
    done["clean_decode_acc_beta"] = decode_accuracy(feats_clean, rec.stim_id, bands=("beta",))
    done["clean_car_decode_acc_beta"] = decode_accuracy(feats_clean_car, rec.stim_id, bands=("beta",))
    neural_rms = float(np.sqrt(np.mean(X0**2)))
    for sar in sars:
        for het in hets:
            lag = lag_ms if het > 0 else 0.0
            if (sar, het, lag) in have or (het == 0 and (sar, het, 4.0) in have):
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 * s + int(10 * het) + int(sar))
            art = make_artifact(ecg, X0.shape[0], neural_rms, sar, het, rng, max_lag_ms=lag)
            Xc = X0 + art
            rows = {"contaminated": evaluate(Xc, X0, feats_clean, rec)}
            rows["car"] = evaluate(car(Xc), car(X0), feats_clean_car, rec)
            rp = rpeaks_from_data(Xc)
            rows["tbr_data_peaks"] = evaluate(template_removal(Xc, rp), X0, feats_clean, rec, {"peak_f1": peak_f1(rp, true_r)})
            rows["tbr_true_peaks"] = evaluate(template_removal(Xc, true_r), X0, feats_clean, rec)
            rows["tbr_data_peaks_lp50"] = evaluate(template_removal(Xc, rp, correction_lowpass=50.0), X0, feats_clean, rec)
            Xi, info = ica_remove(Xc, oracle_ecg=ecg, correction_lowpass=50.0)
            Xi_lp = info.pop("lowpassed")
            rows["ica_auto"] = evaluate(Xi, X0, feats_clean, rec, {"ica": info})
            rows["ica_auto_lp50"] = evaluate(Xi_lp, X0, feats_clean, rec)
            Xb, infob = ica_remove(Xc, fit_band=(2.0, 40.0), oracle_ecg=ecg, correction_lowpass=50.0)
            Xb_lp = infob.pop("lowpassed")
            rows["blica_auto"] = evaluate(Xb, X0, feats_clean, rec, {"ica": infob})
            rows["blica_auto_lp50"] = evaluate(Xb_lp, X0, feats_clean, rec)
            done["conditions"].append({"sar_db": sar, "het": het, "lag_ms": lag, "seconds": round(time.time() - t0, 1), "methods": rows})
            json.dump(done, open(path, "w"), indent=1)
            print(f"sub{s} sar={sar} het={het} lag={lag} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(7)))
    ap.add_argument("--sar", type=float, nargs="+", default=[0.0, 10.0, 20.0])
    ap.add_argument("--het", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--ecg-lowpass", type=float, default=45.0, help="0 = keep full MIT-BIH band")
    ap.add_argument("--max-lag-ms", type=float, default=4.0, help="per-channel lag for het > 0 (0 = instantaneous mixing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.subjects:
        run_subject(s, a.sar, a.het, a.out, a.ecg_lowpass or None, a.max_lag_ms)
