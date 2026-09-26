# Why Is Wav2Vec2 Robust to Lossy Audio Compression? A Signal and Representation-Level Study of MP3 and Opus

**Final report:** [FINAL_REPORT.pdf](FINAL_REPORT.pdf) · [FINAL_REPORT.md](FINAL_REPORT.md) · reproducibility map and stale-file list: [FINAL_REPORT_CHECKLIST.md](FINAL_REPORT_CHECKLIST.md). The final report uses speaker-level block-bootstrap CIs (Bisani & Ney, 2004); the utterance-level ΔWER CIs quoted further down this README come from `results/bootstrap_results.csv` and lead to the same conclusions.

ELEC5305 project investigating why Wav2Vec2 remains robust under MP3 and Opus lossy audio compression, and how signal distortion and learned representation changes relate to the onset of ASR errors.

### Overall Objective

The overall objective is to explain the robustness of Wav2Vec2 to lossy audio compression by connecting codec-induced signal distortion, changes in internal learned representations, and the onset of recognition errors.

Rather than treating WER as the only outcome, the project investigates whether measurable acoustic degradation can occur while higher-level representations and recognition performance remain relatively stable, and how this relationship changes under severe compression.

**Project Site:**  
https://surnamemei.github.io/elec5305-project-540077463/

**Proposal:**  
[ELEC5305 Project Proposal v3.pdf](ELEC5305%20Project%20Proposal%20v3.pdf) (previous versions: [v2](ELEC5305%20Project%20Proposal%20v2.pdf), [v1](ELEC5305%20Project%20Proposal%20v1.pdf))

---

## Research Question

> **How do MP3 and Opus compression alter the acoustic signal and the internal representations of Wav2Vec2, and which codec-induced distortions are associated with the onset of ASR errors?**

Secondary question:

> **Is Wav2Vec2 robust because lossy codecs preserve the acoustic information important to the recogniser even when measurable waveform or spectral distortion is already substantial?**

---

## Working Hypothesis

Lossy compression may introduce measurable signal-level distortion before substantially affecting Wav2Vec2's higher-level internal representations or recognition performance. At sufficiently severe compression, representation drift may increase together with WER.

### Scope of Evaluation

The primary objective of this project is machine speech recognition robustness rather than human perceptual audio quality. Perceptual quality metrics may be considered as optional supporting measures, but the main analysis focuses on the relationship between codec-induced signal distortion, Wav2Vec2 representation drift and ASR performance.

## Experimental Setup

The project uses two LibriSpeech evaluation subsets:

- `test-clean`
- `test-other`

For each subset, **500 utterances** are selected using the fixed random seed `5305` for reproducibility.

The same pretrained **Wav2Vec2** ASR system is used for every condition.

### Fixed ASR Model

All compression conditions are evaluated using the same pretrained Wav2Vec2 model, `WAV2VEC2_ASR_BASE_960H`. The model weights, decoding method and sample-rate handling are kept fixed throughout the experiment. No retraining or fine-tuning is performed, so differences in recognition performance can be attributed to changes in the input audio rather than changes in the recogniser.

### Controlled Speech Corpus

LibriSpeech is used as the controlled evaluation corpus because it provides standard 16 kHz speech recordings together with reference transcripts.

The main experiment uses the `test-clean` subset, while `test-other` is used as a more challenging secondary robustness condition. The same fixed sampling procedure and random seed are used so that codec conditions are compared on identical source utterances.

### Evaluation Set Size

The initial pilot used a smaller number of utterances to verify feasibility. For the main experiment, the evaluation set was increased to 500 fixed utterances per subset.

This larger sample size reduces the influence of individual recognition errors and provides a more reliable basis for comparing small WER differences between compression conditions.

The same 500 utterances are reused across all codec conditions within each subset.

### Paired Evaluation

All codec conditions within each LibriSpeech subset are evaluated using exactly the same 500 source utterances.

The fixed random seed `5305` is used to select the utterance indices once, and the same indices are reused for WAV, MP3 and Opus conditions. This paired design allows each compressed result to be compared directly with the corresponding WAV baseline utterance and supports paired statistical analysis.

### Compression Conditions

| Codec | Bitrates |
|---|---|
| WAV | Uncompressed baseline |
| MP3 | 128, 64, 32, 24, 16 kbps |
| Opus | 64, 32, 16, 12, 8, 6 kbps |

### Why Lower Opus Bitrates Are Included

Opus is specifically designed for efficient speech coding, and 16 kbps is still within a practical operating range for wideband speech. Therefore, the lack of substantial WER degradation at 16 kbps is not unexpected.

To identify where ASR robustness begins to break down, lower Opus bitrates are included so that the experiment covers a robust region, a transition region, and a clearly degraded region.

### Main Metrics

- Word Error Rate (WER)
- WER change relative to WAV baseline (ΔWER)
- Average compression ratio
- 95% paired bootstrap confidence interval
- Substitution, deletion and insertion error counts
- Sentence-level and local spectrogram comparison

WER and recognition error statistics are calculated using JiWER. The same transcript normalisation procedure is applied consistently across all codec conditions to ensure fair comparison.

Substitution, deletion and insertion counts are analysed separately to determine how recognition failures change under severe compression. In the strongest low-bitrate conditions, the largest increase is observed in substitution errors, with smaller increases in deletions and insertions.

---

## Preliminary Observation

The initial experiments showed that substantial bitrate reduction did not immediately produce a corresponding increase in WER.

In particular, moderate MP3 and Opus compression remained close to the WAV baseline, while clearer degradation appeared only at more aggressive low-bitrate conditions.

This result changes the focus of the project. Rather than asking only at which bitrate WER begins to increase, the project now investigates why Wav2Vec2 remains robust despite measurable codec-induced signal distortion, and what signal or representation-level changes are associated with the onset of recognition errors.

## Three-Level Analysis Framework

The final analysis is organised across three connected levels:

### Level 1 – Signal

Codec-induced changes are measured against the aligned WAV signal using:

- log-spectral distance (LSD, dB) and mean squared log-spectral distortion `D_spec` (dB²), with log spectra clipped 80 dB below the reference peak;
- frequency-dependent spectral distortion `D(f)`;
- retained bandwidth: the highest frequency at which the codec keeps long-term power within 20 dB of the WAV reference.

### Level 2 – Learned Representation

Wav2Vec2 hidden representations are compared between the original WAV signal and compressed versions at the convolutional feature-encoder output and at every transformer layer (1–12). Drift is `1 − cosine similarity` of hidden states after per-dimension standardisation (see *Methodological Corrections*).

### Linking the levels

`src/predictor_analysis.py` tests which Level 1 or Level 2 measurement best predicts the Level 3 outcome, at the condition level, pooled over utterances, and within a single codec condition, with bootstrap confidence intervals.

### Level 3 – Recognition Task

ASR performance is evaluated using:

- Word Error Rate (WER);
- ΔWER relative to the WAV baseline;
- substitution, deletion and insertion error counts;
- paired bootstrap confidence intervals.

The central objective is to determine how signal-level distortion and learned-representation drift relate to the point at which recognition performance begins to degrade.

## Current Results

### test-clean

| Condition | WER | ΔWER | Compression Ratio |
|---|---:|---:|---:|
| WAV | 3.17% | 0.00 pp | 1.00× |
| MP3 128k | 3.15% | -0.02 pp | 1.95× |
| MP3 64k | 3.10% | -0.07 pp | 3.90× |
| MP3 32k | 3.27% | +0.11 pp | 7.78× |
| MP3 24k | 3.19% | +0.02 pp | 10.34× |
| MP3 16k | 4.08% | +0.91 pp | 15.40× |
| Opus 64k | 3.19% | +0.02 pp | 3.59× |
| Opus 32k | 3.13% | -0.04 pp | 8.15× |
| Opus 16k | 3.18% | +0.01 pp | 15.98× |
| Opus 12k | 3.49% | +0.32 pp | 20.93× |
| Opus 8k | 4.32% | +1.15 pp | 31.41× |
| Opus 6k | 5.45% | +2.28 pp | 40.28× |

### test-other

| Condition | WER | ΔWER | Compression Ratio |
|---|---:|---:|---:|
| WAV | 8.26% | 0.00 pp | 1.00× |
| MP3 128k | 8.50% | +0.24 pp | 1.95× |
| MP3 64k | 8.54% | +0.27 pp | 3.89× |
| MP3 32k | 8.95% | +0.68 pp | 7.75× |
| MP3 24k | 9.69% | +1.42 pp | 10.30× |
| MP3 16k | 12.00% | +3.74 pp | 15.33× |
| Opus 64k | 8.33% | +0.07 pp | 3.64× |
| Opus 32k | 8.36% | +0.10 pp | 8.34× |
| Opus 16k | 8.90% | +0.64 pp | 16.08× |
| Opus 12k | 9.60% | +1.33 pp | 20.92× |
| Opus 8k | 14.89% | +6.63 pp | 31.52× |
| Opus 6k | 19.79% | +11.53 pp | 40.23× |

---

## Signal, Representation and Recognition Results

All values are means over the same 500 utterances. Signal and drift columns are `test-clean`; ΔWER is shown for both subsets.

| Condition | Measured kbps | LSD (dB) | Retained BW (kHz) | Drift conv | Drift L12 | ΔWER clean | ΔWER other |
|---|---:|---:|---:|---:|---:|---:|---:|
| WAV | 256.1 | 0.0 | 8.0 | 0.000 | 0.000 | 0.00 | 0.00 |
| MP3 128k | 130.2 | 3.5 | 7.3 | 0.010 | 0.004 | −0.02 | +0.24 |
| MP3 64k | 65.1 | 4.1 | 7.3 | 0.017 | 0.005 | −0.07 | +0.27 |
| MP3 32k | 32.6 | 6.3 | 7.3 | 0.075 | 0.021 | +0.11 | +0.68 |
| MP3 24k | 24.5 | 9.2 | 5.8 | 0.144 | 0.041 | +0.02 | +1.42 |
| MP3 16k | 16.4 | 11.7 | 5.6 | 0.228 | 0.071 | +0.91 | +3.74 |
| Opus 64k | 71.5 | 1.7 | 8.0 | 0.007 | 0.003 | +0.02 | +0.07 |
| Opus 32k | 31.6 | 4.0 | 8.0 | 0.041 | 0.009 | −0.04 | +0.10 |
| Opus 16k | 16.1 | 5.4 | 8.0 | 0.095 | 0.022 | +0.01 | +0.64 |
| Opus 12k | 12.2 | 5.9 | 8.0 | 0.138 | 0.035 | +0.32 | +1.33 |
| Opus 8k | 8.1 | 11.4 | 4.6 | 0.335 | 0.086 | +1.15 | +6.63 |
| Opus 6k | 6.3 | 11.6 | 4.7 | 0.408 | 0.130 | +2.28 | +11.53 |

### Which measurement predicts ASR degradation?

Correlation with ΔWER, 95% speaker-level block-bootstrap CI (`results/predictor_analysis/`):

| Subset | Predictor | Condition level (Pearson) | Utterance level, pooled (Spearman) | Within condition (mean Spearman) |
|---|---|---:|---:|---:|
| test-clean | LSD | 0.75 [0.66, 0.81] | 0.15 [0.12, 0.19] | −0.02 [−0.06, 0.02] |
| test-clean | Bandwidth loss | 0.81 [0.74, 0.85] | 0.17 [0.13, 0.20] | −0.01 [−0.05, 0.04] |
| test-clean | Drift, conv | 0.93 [0.88, 0.96] | 0.19 [0.15, 0.22] | 0.01 [−0.03, 0.05] |
| test-clean | Drift, layer 12 | **0.96 [0.91, 0.98]** | **0.21 [0.17, 0.25]** | **0.09 [0.05, 0.13]** |
| test-other | LSD | 0.79 [0.76, 0.81] | 0.32 [0.27, 0.37] | 0.03 [−0.03, 0.10] |
| test-other | Bandwidth loss | 0.85 [0.82, 0.87] | 0.33 [0.29, 0.37] | −0.01 [−0.06, 0.04] |
| test-other | Drift, conv | 0.96 [0.94, 0.97] | 0.36 [0.31, 0.41] | 0.01 [−0.04, 0.06] |
| test-other | Drift, layer 12 | **0.97 [0.96, 0.98]** | **0.42 [0.37, 0.46]** | **0.23 [0.18, 0.27]** |

For condition-level Pearson, pooled Spearman and within-condition Spearman, the paired bootstrap CI of (layer-12 drift − LSD) excludes zero on both subsets (`predictor_differences_vs_lsd.csv`). Condition-level Spearman (rank order) does not separate the two measures: 0.77 vs 0.78 on test-clean and 0.99 vs 0.97 on test-other.

## Current Findings

1. **Moderate lossy compression has little practical effect on clean speech.** Up to Opus 16 kbps and MP3 24 kbps, test-clean ΔWER stays within ±0.11 pp and no bootstrap CI excludes zero.
2. **Substantial signal distortion is absorbed by the recogniser.** MP3 24 kbps has an LSD of 9.2 dB and removes everything above 5.8 kHz, yet test-clean WER is unchanged (+0.02 pp).
3. **Wav2Vec2 attenuates codec perturbations with depth.** Standardised drift is highest at the conv output or in layers 1–3 and falls 2.7–4.4× from the conv output to layer 12 on test-clean (e.g. Opus 16 kbps: 0.095 → 0.022). On test-other the attenuation is only 1.5–2.7×, consistent with its larger WER degradation.
4. **Bandwidth is the clearest physical marker of the transition.** The largest WER jumps coincide with the codec low-pass cutoff moving into the speech band: MP3 at 24/16 kbps (cutoff ~5.8/5.6 kHz) and Opus at 8/6 kbps, where libopus switches to narrowband (~4.6 kHz). Opus 12 kbps keeps the full 8 kHz band and degrades only slightly.
5. **Bandwidth alone is not sufficient.** Opus 8 and 6 kbps have the same cutoff and almost the same LSD (11.4 vs 11.6 dB), but ΔWER roughly doubles (+1.15 → +2.28 pp clean; +6.63 → +11.53 pp other). Layer-12 drift separates them (0.086 vs 0.130).
6. **Late-layer drift tracks the size of ΔWER more linearly than any signal measure across conditions** (Pearson r = 0.96–0.97 vs 0.75–0.79 for LSD). Rank order is similar for both (Spearman 0.77–0.99), because LSD saturates rather than misorders. **Within a fixed condition**, signal measures have essentially zero correlation with which utterances degrade, while layer-12 drift shows a modest association (mean ρ = 0.09 test-clean, 0.23 test-other).
7. **The harder `test-other` subset is substantially more vulnerable to aggressive compression.**
8. **Most additional recognition errors under severe compression are substitutions.**

**Caveat on late-layer drift.** Layer 12 sits directly below the CTC output layer, so a change in the transcript necessarily implies some change in layer 12. Its predictive power is therefore partly expected. The informative results are (i) how much of the early-layer perturbation disappears before layer 12, and (ii) that signal-level measures, which are available without running the model, fail to predict which utterances break.

Selected bootstrap results:

| Dataset | Condition | ΔWER | 95% CI |
|---|---|---:|---:|
| test-clean | MP3 16k | +0.91 pp | [+0.59, +1.29] |
| test-clean | Opus 12k | +0.32 pp | [+0.09, +0.57] |
| test-clean | Opus 8k | +1.15 pp | [+0.81, +1.51] |
| test-other | MP3 16k | +3.74 pp | [+3.07, +4.43] |
| test-other | Opus 8k | +6.63 pp | [+5.71, +7.62] |
| test-clean | Opus 6k | +2.28 pp | [+1.86, +2.76] |
| test-other | Opus 6k | +11.53 pp | [+10.30, +12.83] |

Selected error-type increases (JiWER alignment; `results/final_report/table_main_results.csv`):

| Condition | ΔS | ΔD | ΔI |
|---|---:|---:|---:|
| test-clean MP3 16k | +85 | +15 | -7 |
| test-clean Opus 8k | +104 | +14 | -1 |
| test-other MP3 16k | +282 | +41 | +5 |
| test-other Opus 8k | +492 | +61 | +29 |
| test-clean Opus 6k | +199 | +27 | +6 |
| test-other Opus 6k | +858 | +116 | +38 |

The spectrogram analysis shows substantial attenuation and modification of high-frequency spectral content under very low bitrate compression. These observations are treated as supporting evidence rather than proof of direct causation.

## Methodological Corrections

Three measurement issues were found and corrected after the proposal v2 results. All tables and figures in this README use the corrected versions.

1. **Log-spectral floor.** Spectral distortion was originally computed as `20·log10(|X| + 1e-8)` with no floor. MP3 quantises some bins to exactly zero, which maps them to −160 dB and lets a few bins dominate the average. As a result, MP3 128 kbps appeared about four times more distorted than Opus 16 kbps (317 vs 72 dB²). Log spectra are now clipped 80 dB below the reference peak, giving 23.6 vs 32.3 dB², and LSD in dB is reported as the main signal measure.
2. **Bandwidth measure.** The original "effective bandwidth" was the 95% cumulative-energy frequency. Speech energy lies mostly below 3 kHz, so this measure (~2.9 kHz for every condition) did not detect codec low-pass filtering. It was replaced by the retained bandwidth defined above.
3. **Anisotropy of Wav2Vec2 layers.** Raw cosine similarity is not comparable across layers. In layer 11, frames of two *unrelated* utterances already have a cosine similarity of about 0.94, so raw drift in that layer is close to zero whatever the input. Hidden states are therefore standardised per dimension, using mean and standard deviation estimated from the WAV frames of 100 fixed calibration utterances (Timkey & van Schijndel, 2021), before computing cosine similarity. Over 50 pairs of unrelated test-clean utterances (`src/anisotropy_check.py`), the raw cosine similarity is 0.96 in layer 11; after standardisation it is between 0.02 and 0.08 in every layer. Raw drift is still stored (`drift_*` columns) for comparison.

In addition, the signal, representation and EnCodec analyses now use the same 500 utterances per subset as the ASR experiment, instead of a separate 100-utterance sample, so every measurement is paired with the WER of the same utterance.

---

## Requirements

### System Requirements

- Python 3.12
- Linux or WSL recommended
- FFmpeg with:
  - `libmp3lame`
  - `libopus`

Check FFmpeg:

```bash
ffmpeg -version
```
### FFmpeg and Codec Configuration

The experiments were run using FFmpeg 6.1.1-3ubuntu5 on Ubuntu.

The following encoders were used:

- MP3: `libmp3lame`
- Opus: `libopus`

All LibriSpeech inputs are mono 16 kHz speech signals. Target bitrates are explicitly specified using FFmpeg's `-b:a` option.

The MP3 encoding command follows the form:

```bash
ffmpeg -y -loglevel error -i input.wav \
-codec:a libmp3lame \
-b:a BITRATE \
compressed.mp3
```

The Opus encoding command follows the form:

```bash
ffmpeg -y -loglevel error -i input.wav \
-codec:a libopus \
-b:a BITRATE \
compressed.opus
```

No additional VBR or CBR mode option was explicitly specified. The encoder defaults were retained consistently across all compression conditions.

Actual effective bitrate is measured from the encoded file size and audio duration rather than assuming that the requested bitrate is achieved exactly.

Codec comparisons are based on explicit encoder configurations rather than file extensions alone. The encoder implementation, target bitrate, sample rate and channel configuration are kept consistent and documented for every condition so that differences can be attributed to the codec settings rather than uncontrolled encoding choices.

### Python Environment

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

PyTorch can run on CPU or CUDA. CUDA is optional but significantly speeds up Wav2Vec2 inference.

---

## Dataset Setup

The scripts use LibriSpeech through TorchAudio.

The first time a subset is used, set:

```python
download=True
```

After the dataset has been downloaded locally, use:

```python
download=False
```

The current experiments use:

```python
NUM_SAMPLES = 500
RANDOM_SEED = 5305
```

Keep the same random seed when reproducing the reported results.

---

## How to Run

Run all commands from the repository root.

### Recommended Execution Order

For the main analysis, run the following scripts from the repository root:

```bash
python src/run_all_experiments.py
python src/bootstrap_analysis.py
python src/error_analysis.py
python src/signal_distortion_analysis.py
python src/representation_analysis.py
python src/predictor_analysis.py
python src/integrated_analysis.py
python src/failure_case_analysis.py
python src/encodec_extension.py   # optional; needs representation_analysis.py output
```

Final report tables, checks and figures (no model training; see `FINAL_REPORT_CHECKLIST.md`):

```bash
python src/anisotropy_check.py        # unrelated-utterance cosine per layer
python src/opus_mode_check.py         # Opus coding mode / bandwidth from the bitstream
python src/final_report_tables.py     # authoritative tables + validation checks
python src/final_case_studies.py      # rule-based case selection + drift quartiles
python src/final_report_figures.py    # report Figures 1-3
bash report/build_report.sh           # FINAL_REPORT.md + FINAL_REPORT.pdf (needs pandoc + xelatex)
```

The main experiment generates the ASR results first. The later analysis scripts read the saved outputs from `results/` and generate additional statistical summaries, signal-level analysis, representation-level analysis, integrated figures, and representative failure cases.

### Additional / Development Scripts

### 1. Baseline ASR test

```bash
python src/baseline_asr.py
```

Checks the Wav2Vec2 ASR pipeline on LibriSpeech audio.

### 2. Early MP3 experiment

```bash
python src/experiment_mp3.py
```

Runs the original MP3-focused pilot experiment. This script is retained as part of the project development history.

### 3. Main compression experiment

```bash
python src/run_all_experiments.py
```

Runs WAV, MP3 and Opus conditions for the selected LibriSpeech subset and saves:

- per-utterance results
- WER summaries
- compression ratios

The dataset is selected inside the script using:

```python
DATASET_NAME = "test-clean"
```

or:

```python
DATASET_NAME = "test-other"
```

Run the script once for each subset.

### 4. Generate result figures

```bash
python src/analyse_results.py
```

Reads the saved summary CSV files and generates plots including:

- WER vs bitrate
- ΔWER vs bitrate
- WER vs compression ratio
- test-clean vs test-other comparison

### 5. Bootstrap confidence intervals

```bash
python src/bootstrap_analysis.py
```

Runs paired bootstrap resampling using the existing per-utterance results and saves:

```text
results/bootstrap_results.csv
```

### 6. Error analysis

```bash
python src/error_analysis.py
```

Calculates:

- new recognition errors
- recovered errors
- worsened / improved utterances
- substitution, deletion and insertion changes

Outputs are saved under:

```text
results/error_analysis/
```

### 7. Sentence-level spectrogram analysis

```bash
python src/spectrogram_analysis.py
```

Generates WAV, compressed, difference and combined spectrogram figures for representative cases.

### 8. Local word-level spectrogram analysis

```bash
python src/local_spectrogram_analysis.py
```

Generates local spectrogram comparisons around selected substitution-error regions.

### 9. Codec and dataset comparison analysis

```bash
python src/comparison_analysis.py
```

### 10. Predictor analysis

```bash
python src/predictor_analysis.py
```

Tests which signal-level or representation-level measurement best predicts ΔWER and saves:

```text
results/predictor_analysis/
results/figures/predictor_condition_scatter.png
results/figures/predictor_correlation_by_layer.png
```

---

## Repository Structure

```text
elec5305-project-540077463/
├── src/
│   ├── baseline_asr.py
│   ├── experiment_mp3.py
│   ├── run_all_experiments.py
│   ├── analyse_results.py
│   ├── bootstrap_analysis.py
│   ├── error_analysis.py
│   ├── spectrogram_analysis.py
│   ├── local_spectrogram_analysis.py
│   ├── signal_distortion_analysis.py
│   ├── representation_analysis.py
│   ├── predictor_analysis.py
│   ├── integrated_analysis.py
│   ├── failure_case_analysis.py
│   ├── comparison_analysis.py
│   ├── encodec_extension.py
├── results/
│   ├── baseline_results.csv
│   ├── mp3_32k_results.csv
│   ├── test-clean_experiment_details.csv
│   ├── test-other_experiment_details.csv
│   ├── test-clean_summary_results.csv
│   ├── test-other_summary_results.csv
│   ├── bootstrap_results.csv
│   ├── signal_distortion_summary.csv
│   ├── representation_similarity_summary.csv
│   ├── representation_standardisation.pt
│   ├── integrated_analysis_summary.csv
│   ├── encodec_extension_results.csv
│   ├── encodec_extension_summary.csv
│   ├── error_analysis/
│   ├── comparison_analysis/
│   ├── frequency_distortion/
│   ├── predictor_analysis/
│   └── figures/
├── requirements.txt
├── README.md
├── CODE_GUIDE.md
├── index.md
├── PROJECT_RATIONALE.md
└── ELEC5305 Project Proposal v1.pdf
```

The repository includes the experiment code, summary results, per-utterance outputs, error-analysis results, comparison-analysis outputs, and generated figures used in the current analysis. Downloaded LibriSpeech data, temporary compressed audio, virtual environments, and cache files are excluded from version control.

---

## Reproducibility Notes

- Random seed: `5305`
- Same utterances are used across all codec conditions within each dataset.
- The same Wav2Vec2 model is used for every condition.
- FFmpeg is used for both MP3 and Opus encoding.
- Bootstrap comparisons are paired by LibriSpeech dataset index.
- Summary and per-utterance experimental results used in the current analysis are included in the repository.
- Downloaded LibriSpeech data and temporary codec files are not stored in Git.

For an exact snapshot of the current Python environment, an optional lock file can be generated with:

```bash
pip freeze > requirements-lock.txt
```

---

## Core Project Scope Status

The core project scope is now complete. The current implementation includes:

- a fixed Wav2Vec2 ASR model;
- paired WAV, MP3 and Opus evaluation;
- multiple bitrate conditions covering robust, transition and degraded regions;
- `test-clean` and `test-other` evaluation;
- actual bitrate measurement;
- WER and ΔWER analysis;
- paired bootstrap confidence intervals;
- substitution, deletion and insertion analysis;
- signal-level spectral distortion and retained-bandwidth analysis;
- Wav2Vec2 representation drift at the conv output and all 12 transformer layers;
- a predictor analysis comparing signal and representation measures as predictors of ΔWER;
- an integrated signal → representation → recognition analysis;
- representative individual failure cases.

Any additional neural-codec experiment is treated as an extension rather than a requirement for the core project.

## Project Contribution

The main contribution of this project is not simply a comparison of MP3 and Opus recognition accuracy.

Instead, the project investigates why Wav2Vec2 remains robust under lossy compression by linking three levels of analysis:

1. **Signal level** — how compression changes the spectral characteristics of the speech signal;
2. **Representation level** — how these changes propagate through early, middle and late Wav2Vec2 hidden representations;
3. **Task level** — when these changes become large enough to produce measurable ASR degradation.

The results show that substantial signal distortion (up to ~9 dB LSD and a 5.8 kHz low-pass cutoff) can occur without measurable WER change, that Wav2Vec2 attenuates codec perturbations by 2.7–4.4× (test-clean) and 1.5–2.7× (test-other) between its conv output and final layer, and that late-layer drift tracks task degradation more closely than signal distortion. This association is not causal, and within a condition it is modest.

This provides a more informative explanation of compression robustness than WER-only codec comparison.

## Next Steps

- write the final research report around the predictor analysis and the corrected signal / representation results;
- update the failure-case discussion using utterances with high layer-12 drift;
- prepare the project demonstration video.

### Optional EnCodec Extension

EnCodec was evaluated as an optional extension after the MP3/Opus analysis was completed. Its purpose was to test whether the observed signal → representation → WER relationship also generalises to a neural codec architecture.

Because the 24 kHz mono EnCodec model operates at a different sample rate from LibriSpeech and Wav2Vec2, the experiment included an uncompressed 16 → 24 → 16 kHz resampling control.

The extension uses the same 500 `test-clean` utterances as the main experiment, so its WAV baseline (3.17%) is identical to the main WAV result.

| Condition | LSD (dB) | Drift L1 | Drift L12 | WER | ΔWER |
|---|---:|---:|---:|---:|---:|
| WAV | 0.0 | 0.000 | 0.000 | 3.17% | 0.00 pp |
| 16 → 24 → 16 kHz control | 1.4 | 0.004 | 0.002 | 3.19% | +0.02 pp |
| EnCodec 24 kbps | 7.0 | 0.179 | 0.038 | 3.28% | +0.12 pp |
| EnCodec 6 kbps | 7.5 | 0.265 | 0.062 | 3.77% | +0.60 pp |
| EnCodec 1.5 kbps | 8.5 | 0.458 | 0.198 | 9.95% | +6.79 pp |

The resampling control produced only a negligible WER change. EnCodec supports the main conclusion: at 24 kbps its LSD (7.0 dB) is larger than that of Opus 16 kbps (5.4 dB), yet WER is almost unchanged. Its layer-12 drift (0.038) is close to that of MP3 24 kbps (0.041), which also leaves WER unchanged. The relationship between late-layer drift and WER therefore carries over to a neural codec, while the relationship between spectral distortion and WER does not.

This extension is treated as supporting evidence for the main MP3/Opus analysis rather than as a separate codec-ranking experiment.

---

## References

1. A. Baevski, Y. Zhou, A. Mohamed and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020.
2. W.-N. Hsu et al., "Robust wav2vec 2.0: Analyzing Domain Shift in Self-Supervised Pre-Training," Interspeech, 2021.
3. A. Pasad, J.-C. Chou and K. Livescu, "Layer-wise Analysis of a Self-supervised Speech Representation Model," IEEE ASRU, 2021.
4. V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," ICASSP, 2015.
5. J.-M. Valin, K. Vos and T. Terriberry, "Definition of the Opus Audio Codec," IETF RFC 6716, 2012.
6. K. Brandenburg, "MP3 and AAC Explained," AES 17th International Conference on High-Quality Audio Coding, 1999.
7. L. Besacier, C. Bergamini, D. Vaufreydaz and E. Castelli, "The Effect of Speech and Audio Compression on Speech Recognition Performance," IEEE Workshop on Multimedia Signal Processing, 2001.
8. H.-G. Hirsch and D. Pearce, "The AURORA Experimental Framework for the Performance Evaluation of Speech Recognition Systems under Noisy Conditions," ISCA ITRW ASR2000, 2000.
9. A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML, 2023.
10. A. Défossez, J. Copet, G. Synnaeve and Y. Adi, "High Fidelity Neural Audio Compression," Transactions on Machine Learning Research, 2023.
11. N. Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec," IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2022.
12. H. Wu et al., "Codec-SUPERB: An In-Depth Analysis of Sound Codec Models," Findings of ACL, 2024.
13. W. Timkey and M. van Schijndel, "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality," EMNLP, 2021.
14. K. Ethayarajh, "How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings," EMNLP-IJCNLP, 2019.
15. S. Kornblith, M. Norouzi, H. Lee and G. Hinton, "Similarity of Neural Network Representations Revisited," ICML, 2019.
16. R. M. Gray, A. Buzo, A. H. Gray and Y. Matsuyama, "Distortion Measures for Speech Processing," IEEE Transactions on Acoustics, Speech, and Signal Processing, 1980.
17. M. Bisani and H. Ney, "Bootstrap Estimates for Confidence Intervals in ASR Performance Evaluation," ICASSP, 2004.
18. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

Software and data resources: [LibriSpeech (OpenSLR 12)](https://www.openslr.org/12/), [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (the torchaudio `WAV2VEC2_ASR_BASE_960H` bundle is a port of the same fairseq checkpoint), [JiWER](https://github.com/jitsi/jiwer), [FFmpeg codecs](https://ffmpeg.org/ffmpeg-codecs.html), [Opus](https://www.opus-codec.org/), [EnCodec](https://github.com/facebookresearch/encodec), [Codec-Evaluation benchmark](https://github.com/wuzhiyue111/Codec-Evaluation).
