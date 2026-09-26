# Code Guide

This document explains the purpose of each script in the ELEC5305 project and how the scripts connect together.

## Overall Workflow

```text
LibriSpeech
    ↓
run_all_experiments.py
    ↓
WAV / MP3 / Opus versions
    ↓
Wav2Vec2 ASR
    ↓
WER + bitrate + compression results
    ↓
analyse_results.py
bootstrap_analysis.py
error_analysis.py
signal_distortion_analysis.py
representation_analysis.py
predictor_analysis.py
integrated_analysis.py
failure_case_analysis.py
    ↓
figures + summary CSV files
```

An optional neural-codec extension is implemented separately using:

```text
encodec_extension.py
```

---

## Main Experiment Scripts

### `run_all_experiments.py`

**Purpose:** Main MP3 / Opus experiment.

This script:

1. Loads a fixed set of LibriSpeech utterances.
2. Uses uncompressed WAV as the baseline.
3. Encodes the same utterance using MP3 or Opus at multiple bitrates.
4. Decodes the compressed audio back to waveform form.
5. Sends every waveform through the same pretrained Wav2Vec2 ASR model.
6. Calculates per-utterance and corpus-level Word Error Rate (WER).
7. Measures actual effective bitrate and compression ratio.
8. Saves detailed and summary CSV files.

Important settings:

```text
NUM_SAMPLES = 500
RANDOM_SEED = 5305
```

The main codec conditions are:

```text
WAV

MP3:
128k
64k
32k
24k
16k

Opus:
64k
32k
16k
12k
8k
6k
```

The same utterances are reused across all codec conditions within each dataset.

---

### `analyse_results.py`

**Purpose:** Generate the main WER and compression figures.

This script reads the saved summary CSV files and creates figures including:

- WER vs bitrate;
- ΔWER vs bitrate;
- WER vs compression ratio;
- `test-clean` vs `test-other`.

This script does not rerun Wav2Vec2.

---

### `bootstrap_analysis.py`

**Purpose:** Estimate uncertainty in WER differences.

This script performs paired bootstrap resampling between WAV and compressed conditions.

The same sampled utterance indices are used for the WAV and compressed predictions within each bootstrap iteration.

The output includes:

- ΔWER;
- lower 95% confidence bound;
- upper 95% confidence bound.

This allows small near-baseline changes to be separated from clearer and more consistent degradation.

---

### `error_analysis.py`

**Purpose:** Analyse how recognition errors change under compression.

The script calculates word-level:

- substitutions;
- deletions;
- insertions.

It also identifies utterances that are:

- newly incorrect;
- recovered;
- worsened;
- improved;
- unchanged.

The severe low-bitrate conditions show that most additional recognition errors are substitutions.

---

## Signal-Level Analysis

### `spectrogram_analysis.py`

**Purpose:** Produce sentence-level spectrogram case studies.

For selected utterances, the script compares:

- original WAV spectrogram;
- compressed spectrogram;
- spectral difference.

These plots provide visual examples of codec-induced acoustic changes.

---

### `local_spectrogram_analysis.py`

**Purpose:** Examine local regions around representative ASR errors.

The complete utterance is compressed first, then the same time region is cropped from WAV and compressed audio.

This avoids analysing a separately compressed short segment.

---

### `signal_distortion_analysis.py`

**Purpose:** Quantify codec-induced signal distortion across multiple utterances.

The script runs on the same 500 utterances per subset (`test-clean` and `test-other`) as the ASR experiment and calculates:

- mean squared log-spectral distortion `D_spec` (dB²);
- log-spectral distance `LSD` (dB): RMS over frequency, then mean over frames;
- frequency-dependent distortion `D(f)`;
- retained bandwidth: the highest frequency at which the codec output keeps long-term power within 20 dB of the WAV reference;
- power change in the 4–8 kHz band;
- estimated codec delay / alignment (FFT-based normalised cross-correlation).

The analysis uses:

```text
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
DYNAMIC_RANGE_DB = 80
BANDWIDTH_DROP_DB = 20
```

The original WAV waveform is used as the reference.

Log spectra are clipped at 80 dB below the reference spectrogram peak before comparison. Without this floor, bins that a codec sets to exactly zero (common for MP3) are mapped to `20·log10(ε)` and dominate the average. An earlier version used `ε = 1e-8` without a floor, which made MP3 128 kbps appear about four times more distorted than Opus 16 kbps; this was an artefact of `ε`, not a property of the codecs.

The retained-bandwidth measure replaces an earlier 95%-energy roll-off measure. Because speech energy is concentrated below about 3 kHz, the roll-off frequency barely moved when a codec removed 4–8 kHz content, so it did not detect the codec low-pass cutoff.

The output includes:

```text
results/signal_distortion_results.csv
results/signal_distortion_summary.csv
results/frequency_distortion/
results/figures/frequency_distortion_comparison.png
```

---

## Representation-Level Analysis

### `representation_analysis.py`

**Purpose:** Measure how compression changes Wav2Vec2 hidden representations.

The script compares original WAV and compressed audio at:

```text
conv       (convolutional feature-encoder output)
Layer 1 ... Layer 12  (every transformer layer)
```

for all 11 MP3 / Opus conditions.

Representation similarity is measured using cosine similarity.

Representation drift is defined as:

```text
drift = 1 - cosine similarity
```

The analysis uses the same 500 fixed utterances per subset (`test-clean` and `test-other`) as the ASR experiment, so every drift value can be paired with the WER of the same utterance and condition. The detailed CSV is stored in wide format (one drift column per layer).

The output includes:

```text
results/representation_similarity_results.csv
results/representation_similarity_summary.csv
results/figures/representation_drift_by_layer.png
```

The results show larger drift in early layers and smaller drift in deeper layers under moderate compression, while severe compression also increases deeper-layer drift.

---

## Predictor Analysis

### `predictor_analysis.py`

**Purpose:** Test which measurement best predicts ASR degradation (signal distortion or representation drift).

Predictors:

- signal level: LSD, `D_spec`, bandwidth loss, 4–8 kHz power loss;
- representation level: drift at the conv output and every transformer layer.

Outcome: WER change relative to the WAV version of the same utterance.

Three analyses are run per subset:

1. **condition level** – correlation between condition means and corpus ΔWER over the 11 codec conditions plus WAV;
2. **utterance level, pooled** – Spearman correlation over all (utterance, condition) pairs;
3. **within condition** – Spearman correlation across utterances inside one codec condition (which utterances fail at a fixed bitrate), averaged over conditions.

95% confidence intervals use a block bootstrap over speakers (Bisani & Ney, 2004; set `BOOTSTRAP_UNIT = "dataset_index"` to resample utterances). Because every predictor is evaluated on the same resamples, differences between predictors (e.g. layer-12 drift minus LSD) also get confidence intervals.

It does not rerun the ASR model.

The output includes:

```text
results/predictor_analysis/predictor_correlations.csv
results/predictor_analysis/predictor_differences_vs_lsd.csv
results/predictor_analysis/within_condition_correlations.csv
results/predictor_analysis/condition_level_predictors.csv
results/figures/predictor_condition_scatter.png
results/figures/predictor_correlation_by_layer.png
```

---

## Final Report Scripts

These scripts read existing results and do not train or fine-tune anything.

| Script | Purpose | Output |
|---|---|---|
| `anisotropy_check.py` | Cosine similarity between 50 pairs of *unrelated* utterances per layer, raw and standardised (justifies standardised drift) | `results/final_report/table_anisotropy.csv` |
| `opus_mode_check.py` | Reads the Opus TOC byte of every packet to report the coding mode and bandwidth libopus chose at each bitrate | `results/final_report/table_opus_modes.csv` |
| `final_report_tables.py` | Builds the authoritative report tables and runs the validation checks: WER recomputation, shared utterance sets, S/D/I totals, utterance- and speaker-level bootstrap, EnCodec baseline | `results/final_report/table_*.csv`, `validation_report.txt` |
| `final_case_studies.py` | Rule-based robust/failure case selection at Opus 8 kbps and the new-error rate by drift / LSD quartile | `case_study_selection.csv`, `table_drift_quartiles.csv`, `fig4_case_study.png` |
| `final_report_figures.py` | Report Figures 1–3 | `results/final_report/fig1–fig3` |
| `report/build_report.sh` | Builds `FINAL_REPORT.md` and `FINAL_REPORT.pdf` from `report/FINAL_REPORT.pandoc.md` with pandoc, the verified BibTeX file and the IEEE CSL style | report files |

---

## Integrated Analysis

### `integrated_analysis.py`

**Purpose:** Connect the three levels of the project:

```text
Signal
    ↓
Representation
    ↓
Recognition
```

The script reads previously generated results and combines, for every condition and both subsets:

- measured bitrate;
- log-spectral distance and retained bandwidth;
- conv, Layer 1, Layer 6 and Layer 12 drift;
- WER, ΔWER and its bootstrap confidence interval.

The figure plots every quantity against measured bitrate for MP3 and Opus (feedback point 33).

It does not rerun the ASR model.

The output includes:

```text
results/integrated_analysis_summary.csv
results/figures/integrated_analysis.png
```

This figure is one of the main summary figures of the project.

---

## Representative Failure Cases

### `failure_case_analysis.py`

**Purpose:** Select individual utterances where severe compression creates large recognition errors.

The script selects representative cases from the existing error-analysis results and reports:

- reference transcript;
- WAV prediction;
- compressed prediction;
- WAV WER;
- compressed WER;
- ΔWER;
- changes in substitutions, deletions and insertions.

Outputs:

```text
results/error_analysis/selected_failure_cases.csv
results/error_analysis/selected_failure_cases.txt
```

These examples complement the average quantitative results with concrete recognition failures.

---

## Optional Neural Codec Extension

### `encodec_extension.py`

**Purpose:** Test whether the main signal → representation → recognition pattern also appears under a neural audio codec.

The EnCodec extension uses:

```text
the same 500 fixed test-clean utterances as the main experiment (paired with MP3 / Opus)
```

Conditions:

```text
WAV
16 → 24 → 16 kHz resampling control
EnCodec 24 kbps
EnCodec 6 kbps
EnCodec 1.5 kbps
```

The resampling control is included because the mono EnCodec model operates at 24 kHz while LibriSpeech and Wav2Vec2 operate at 16 kHz.

The script calculates:

- corpus-level WER;
- ΔWER;
- log-spectral distortion;
- Layer 1 drift;
- Layer 6 drift;
- Layer 12 drift.

Outputs:

```text
results/encodec_extension_results.csv
results/encodec_extension_summary.csv
results/figures/encodec_extension.png
```

The EnCodec experiment is treated as an optional extension rather than part of the core MP3 / Opus codec comparison.

---

## Pilot and Supporting Scripts

### `baseline_asr.py`

Early development script used to confirm:

- LibriSpeech loading;
- Wav2Vec2 inference;
- CTC decoding;
- baseline WER calculation.

This script is retained as project development history.

---

### `experiment_mp3.py`

Early pilot used to test:

- FFmpeg MP3 encoding;
- MP3 decoding;
- compression-ratio calculation;
- compressed-audio ASR.

The final multi-codec experiment is implemented in `run_all_experiments.py`.

---

### `comparison_analysis.py`

Supporting analysis script used for codec and dataset comparisons.

It is retained as an additional analysis utility rather than the main final-analysis script.

---

## Recommended Execution Order

Run commands from the repository root.

Main experiment:

```bash
python src/run_all_experiments.py
```

Main analysis:

```bash
python src/analyse_results.py
python src/bootstrap_analysis.py
python src/error_analysis.py
python src/signal_distortion_analysis.py
python src/representation_analysis.py
python src/predictor_analysis.py
python src/integrated_analysis.py
python src/failure_case_analysis.py
```

Optional neural-codec extension:

```bash
python src/encodec_extension.py
```

Optional case-study scripts:

```bash
python src/spectrogram_analysis.py
python src/local_spectrogram_analysis.py
```

The pilot scripts:

```text
baseline_asr.py
experiment_mp3.py
```

are not required for the final analysis.

---

## Main Experimental Principle

The project is designed as a controlled experiment.

The following remain fixed across the main codec conditions:

- source utterance;
- Wav2Vec2 model;
- model weights;
- decoding method;
- sample selection;
- evaluation metric.

The main changing variables are:

- codec;
- bitrate.

This makes it possible to interpret recognition differences primarily in relation to codec-induced changes in the input signal.

---

## Three-Level Interpretation

The final project is organised around three connected levels:

```text
Level 1: Signal
log-spectral distance
frequency-dependent distortion
retained bandwidth

        ↓

Level 2: Representation
conv-feature drift
Layer 1 ... Layer 12 drift

        ↓

Level 3: Task
WER
ΔWER
substitutions
deletions
insertions
```

The central analysis asks whether measurable signal distortion can occur before higher-level Wav2Vec2 representations and recognition performance begin to degrade substantially.