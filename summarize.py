"""Aggregate results/sub*.json into RESULTS.md (medians across subjects)."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHODS = [
    "contaminated",
    "car",
    "tbr_data_peaks",
    "tbr_true_peaks",
    "tbr_data_peaks_lp50",
    "ica_auto",
    "ica_auto_lp50",
    "blica_auto",
    "blica_auto_lp50",
]
COLS = [
    ("err_db", "err all (dB)"),
    ("err_db_beta", "err beta (dB)"),
    ("err_db_hg", "err HG (dB)"),
    ("map_corr_beta", "beta map r"),
    ("map_corr_hg", "HG map r"),
    ("decode_acc_beta", "beta-only acc"),
]


def concordance(info: dict) -> tuple[int, int, int]:
    auto, oracle = set(info["selected"]), set(info.get("oracle", []))
    return len(auto & oracle), len(auto - oracle), len(oracle - auto)


def main(res_dir: str) -> str:
    subs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(res_dir, "sub*.json")))]
    for s in subs:  # drop duplicate conditions (identical seeds) if a run was ever repeated
        seen, keep = set(), []
        for c in s["conditions"]:
            key = (c["sar_db"], c["het"], c.get("lag_ms", 4.0 if c["het"] > 0 else 0.0))
            if key not in seen:
                seen.add(key)
                keep.append(c)
        s["conditions"] = keep
    lines = [f"Subjects: {len(subs)} ({', '.join(str(s['subject']) for s in subs)}); channels {[s['n_channels'] for s in subs]}", ""]
    clean_beta = [s["clean_decode_acc_beta"] for s in subs]
    clean_car_beta = [s["clean_car_decode_acc_beta"] for s in subs]
    lines.append(
        f"Clean data, hand vs tongue, beta-only accuracy: median {np.median(clean_beta):.3f} "
        f"(CAR-referenced clean: {np.median(clean_car_beta):.3f}); beta+HG: "
        f"{np.median([s['clean_decode_acc'] for s in subs]):.3f}"
    )
    lines.append("")
    def lag(c):
        return c.get("lag_ms", 4.0 if c["het"] > 0 else 0.0)

    conds = sorted({(c["sar_db"], c["het"], lag(c)) for s in subs for c in s["conditions"]})
    for sar, het, lg in conds:
        rows = [
            c for s in subs for c in s["conditions"] if c["sar_db"] == sar and c["het"] == het and lag(c) == lg
        ]
        kind = "common mode" if het == 0 else ("heterogeneous, instantaneous" if lg == 0 else f"heterogeneous + up to {lg:g} ms lags (stress test)")
        lines.append(f"### SAR {sar:+.0f} dB, {kind} (n={len(rows)} subjects)")
        lines.append("")
        lines.append("| method | " + " | ".join(h for _, h in COLS) + " |")
        lines.append("|---" * (len(COLS) + 1) + "|")
        for m in METHODS:
            vals = []
            for key, _ in COLS:
                v = [r["methods"][m][key] for r in rows if m in r["methods"]]
                vals.append(f"{np.median(v):.2f}" if v else "-")
            lines.append(f"| {m} | " + " | ".join(vals) + " |")
        f1 = [r["methods"]["tbr_data_peaks"]["peak_f1"] for r in rows]
        tp = fp = fn = 0
        for r in rows:
            a, b, c = concordance(r["methods"]["ica_auto"]["ica"])
            tp, fp, fn = tp + a, fp + b, fn + c
        lines.append("")
        lines.append(
            f"R-peak F1 from data (no ECG channel): median {np.median(f1):.3f}, min {np.min(f1):.3f}. "
            f"Automated ICA selection vs oracle (|r| >= 0.3 with true ECG): {tp} agree, {fp} extra, {fn} missed."
        )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out = main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results"))
    print(out)
