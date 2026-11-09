"""Why does automated ICA selection fail on one subject? Check the RR-regularity gate.

Runs subject 4 (paired with MIT-BIH record 106) at +10 dB, common-mode artifact, with the
selector's RR-regularity gate on and off, and counts ectopic beats in the record's
reference annotations. Writes results/ectopy_check.json.
"""
import json
import os

import numpy as np
import wfdb

from vecog_cardiac.methods import ica_remove
from vecog_cardiac.metrics import error_db
from vecog_cardiac.signals import ECG_RECORDS, load_ecg, load_ecog, make_artifact

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for s in (4, 0):
    rec = load_ecog(os.path.join(HERE, "data", "motor_imagery.npz"), s)
    X0 = rec.X
    ecg = load_ecg(ECG_RECORDS[s], X0.shape[1], os.path.join(HERE, "data"))
    ann = wfdb.rdann(ECG_RECORDS[s], "atr", pn_dir="mitdb", sampto=int(X0.shape[1] / 1000 * 360))
    beats = [a for a in ann.symbol if a in "NLRBAaJSVrFejnE/fQ"]
    pvc = sum(a == "V" for a in beats)
    art = make_artifact(ecg, X0.shape[0], float(np.sqrt(np.mean(X0**2))), 10.0, 0.0, np.random.default_rng(1000 * s + 10))
    Xc = X0 + art
    row = {"record": ECG_RECORDS[s], "beats": len(beats), "pvc_fraction": round(pvc / len(beats), 3)}
    for gate in (True, False):
        Xi, info = ica_remove(Xc, oracle_ecg=ecg, rr_gate=gate)
        row[f"rr_gate_{gate}"] = {
            "selected": info["selected"],
            "oracle": info["oracle"],
            "scores": info["scores"][:3],
            "err_db_beta": round(error_db(Xi, X0, (13.0, 30.0)), 2),
        }
    out[f"sub{s}"] = row
    print(row, flush=True)
json.dump(out, open(os.path.join(HERE, "results", "ectopy_check.json"), "w"), indent=1)
