# vecog-cardiac-bench

A ground-truth benchmark for removing cardiac (ECG) artifact from intracranial recordings, built to test the methods compared in two recent endovascular BCI papers on data where the clean signal is known.

- Liyanage, Yoo, Grayden, Opie, Oxley (2025). *Artifact removal in electrocorticography devices with cardiac contamination.* IEEE TNSRE 33:3400-3408. Compares common average referencing (CAR), ICA with automated ECG component selection, and template-based removal (TBR) on Stentrode data.
- Feldman et al. (2026). *Preserving motor features by alternative re-referencing to remove heart artifact on the Stentrode.* Advanced Science. Re-referencing schemes and band-limited ICA (BL-ICA).

Stentrode recordings are not public, and on real data you cannot tell how much of the "cleaned" signal is still artifact or how much neural signal a method removed. So this repo builds a semi-synthetic version: real human ECoG with a motor task, plus real two-lead ECG injected with known gains, and scores every method against the untouched recording.

**This is an independent reimplementation from the papers' descriptions, not the authors' code, and not Stentrode data.** Read the limitations before drawing conclusions.

## Setup

- **Neural data:** Miller et al. motor ECoG (Neuromatch copy of `motor_imagery.npz`, https://osf.io/ksqv8), 7 subjects, 46-64 subdural channels at 1 kHz, real hand and tongue movement block (60 trials). Disconnected channels dropped; 0.5 Hz high-pass and 60/120/180 Hz notch.
- **ECG:** PhysioNet MIT-BIH Arrhythmia records 100, 101, 103, 105, 106, 112, 113 (one per subject), both leads, resampled to 1 kHz and low-passed at 45 Hz (removes the surface recording's EMG and quantization noise, which is not beat-locked).
- **Artifact models** (`vecog_cardiac/signals.py::make_artifact`):
  - *common mode*: the same ECG waveform on every channel;
  - *heterogeneous, instantaneous*: per-channel log-normal gains (sigma 0.5) and a per-channel mix of the two ECG leads (different QRS shape per channel). Physically the most plausible of the three, since the cardiac field reaches the electrodes by volume conduction;
  - *heterogeneous + lags (stress test)*: the same plus up to 4 ms per-channel delay. Included to show what breaks instantaneous-mixing assumptions, not because I think the Stentrode has such lags.
  - Overall level set by signal-to-artifact ratio: artifact RMS 0, +10 or +20 dB above neural RMS.
- **Methods** (`vecog_cardiac/methods.py`): CAR; TBR (per-channel moving median template over 30 beats, per-beat least-squares scaling) with R-peaks found from the data itself (first principal component, no ECG channel) or from the true ECG; ICA (FastICA, 40 components) with automatic removal of components whose cardiac score is at least 0.5 (heart rate 40-180 bpm, RR coefficient of variation under 0.25, mean beat-to-template correlation); BL-ICA (same, unmixing fitted on 2-40 Hz data and applied broadband). Each correction also has an `_lp50` variant where the subtracted estimate is low-passed at 50 Hz. All thresholds were fixed before looking at results.
- **Metrics** (`vecog_cardiac/metrics.py`): error vs ground truth in dB (`20 log10(rms(cleaned - clean) / rms(clean))`), overall and in beta (13-30 Hz) and high gamma (70-150 Hz); per-channel move-vs-rest activation maps (signed r^2 of log band power) correlated with the clean maps; hand-vs-tongue decoding with shrinkage LDA on beta power only (5-fold CV). CAR is scored against CAR of the clean data so it is not penalized for changing the reference.

## Results (medians over 7 subjects, one seed per condition)

Full tables for every condition: [RESULTS.md](RESULTS.md). Raw per-subject numbers: `results/sub*.json`. Clean-data beta-only decoding: 0.917.

Artifact at +10 dB:

| method | common mode: beta err (dB) / beta map r / beta acc | heterogeneous: beta err / map r / acc | HG map r (heterogeneous) |
|---|---|---|---|
| contaminated | 18.3 / 0.51 / 0.63 | 18.2 / 0.37 / 0.60 | 1.00 |
| CAR | exact / 1.00 / 0.90 | 14.8 / 0.54 / 0.73 | 1.00 |
| TBR, peaks from data | 2.4 / 0.79 / 0.75 | 2.5 / 0.73 / 0.75 | 0.86 |
| ICA, auto selection | -7.9 / 0.94 / 0.83 | 5.1 / 0.81 / 0.78 | 0.96 |
| BL-ICA, auto selection | -8.8 / 0.93 / 0.92 | 5.5 / 0.79 / 0.83 | 0.98 |
| any of the above, `_lp50` | same beta numbers | same beta numbers | 1.00 |

What I found:

1. **Every correction method injects noise into high gamma when the artifact does not reach it.** With the ECG band-limited to 45 Hz, the contaminated data's high-gamma activation maps are untouched (r = 1.00), but after TBR, ICA or BL-ICA they drop to a median r of 0.86 / 0.94 / 0.93 across all 63 subject-conditions (worst 0.23). TBR subtracts template noise; ICA removes components that also carry broadband neural activity. Low-passing the subtracted estimate at 50 Hz brings high-gamma maps back to r >= 0.99 in every case and leaves beta error unchanged. If the motor decoder uses high gamma, the cleanup should be band-limited to where the artifact lives.
2. **Common-mode artifact is the easy case.** CAR removes it exactly. ICA and BL-ICA recover beta well (about -8 to -9 dB error, beta decoding 0.83-0.92 vs 0.92 clean). TBR only knocks beta error 12-17 dB below the contaminated level, so it is fine at 0 dB but leaves +11.6 dB of beta error at +20 dB.
3. **Heterogeneous artifact is where methods disagree.** With different gains and QRS shapes per channel, CAR barely helps (14.8 dB at +10 dB), TBR is best on beta error (2.5 dB), and ICA/BL-ICA land in between on error but higher on map correlation. At +20 dB nothing gets beta error below 12 dB, and beta-only decoding stays at 0.63-0.70 vs 0.92 clean.
4. **Automated ECG component selection works for regular rhythms and fails with ectopy.** It matched the oracle component on 6 of 7 subjects in every common-mode condition. The failure is the subject paired with MIT-BIH record 106, whose reference annotations mark 14.3% of beats in the window as premature ventricular contractions: nothing was selected in all 9 conditions. Dropping the RR-regularity gate does not fix it (`check_ectopy.py`): the true cardiac component still scores lower than two wrong ones (~0.5), because a single median template fits a mix of normal and ventricular beats poorly. In heterogeneous conditions the selector also misses the weaker second cardiac component (6 of 14 oracle components missed at +10 dB).
5. **No ECG channel is needed to find beats.** R-peaks from the first principal component of the 5-30 Hz data had F1 1.000 (median; minimum 0.95) against the true ECG, and TBR with data-derived peaks matches TBR with true peaks.

## Limitations (please read)

- Not Stentrode data. Miller's grids are subdural, not endovascular, and the real Stentrode artifact (electronics in the chest, specific referencing) may differ in amplitude, spectrum and spatial structure. The three mixing models bracket plausible cases; they are not measurements.
- The ECG is a surface Holter recording (MIT-BIH), low-passed at 45 Hz. A real intravascular cardiac artifact may have energy above 45 Hz; `run_bench.py --ecg-lowpass 0` keeps the full band (not part of the reported results). Finding 1 depends on the artifact being band-limited.
- One random draw of gains per condition, 7 subjects. Medians, not confidence intervals.
- My ICA selector and TBR are reimplementations from text; the papers' exact parameters (template length, ICA variant, selection rule, BL-ICA band) may differ, and so may the results.
- Activation maps and decoding use one task and simple features.

## Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && curl -L -o data/motor_imagery.npz https://osf.io/ksqv8/download   # ~600 MB
.venv/bin/python run_bench.py                           # common mode + heterogeneous with lags
.venv/bin/python run_bench.py --het 0.5 --max-lag-ms 0  # heterogeneous, instantaneous
.venv/bin/python check_ectopy.py
.venv/bin/python summarize.py > RESULTS.md
.venv/bin/python -m pytest -q tests
```

MIT-BIH records download automatically through `wfdb` on first use. A full run took about 70-130 s per subject-condition on an Apple Silicon laptop (CPU only).

## Tests

`tests/test_methods.py` runs on synthetic data with known ground truth (no downloads): R-peak detection in both polarities, cardiac score separates ECG from noise, artifact scaling hits the requested SAR exactly, CAR removes a common-mode artifact exactly, every method cuts artifact error by more than 10 dB, the low-passed correction protects high gamma, and activation-map sanity checks.
