# Why Is Wav2Vec2 Robust to Lossy Audio Compression? A Signal and Representation-Level Study of MP3 and Opus

ELEC5305 project investigating why Wav2Vec2 remains robust under MP3 and Opus lossy audio compression, and how signal distortion and learned representation changes relate to the onset of ASR errors.

### Overall Objective

The overall objective is to explain the robustness of Wav2Vec2 to lossy audio compression by connecting codec-induced signal distortion, changes in internal learned representations, and the onset of recognition errors.

Rather than treating WER as the only outcome, the project investigates whether measurable acoustic degradation can occur while higher-level representations and recognition performance remain relatively stable, and how this relationship changes under severe compression.

**Project Site:**  
https://surnamemei.github.io/elec5305-project-540077463/

**Proposal:**  
[ELEC5305 Project Proposal v1.pdf](ELEC5305%20Project%20Proposal%20v1.pdf)

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

Codec-induced changes are measured using:

- log-spectral distortion;
- frequency-dependent spectral distortion;
- effective retained bandwidth.

### Level 2 – Learned Representation

Wav2Vec2 hidden representations are compared between the original WAV signal and compressed versions using cosine similarity and representation drift across selected transformer layers.

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

---

## Current Findings

1. **Moderate lossy compression has little practical effect on clean speech.**
2. **Severe low-bitrate compression causes clear ASR degradation.**
3. **The harder `test-other` subset is substantially more vulnerable to aggressive compression.**
4. **Most additional recognition errors under severe compression are substitutions.**

Selected bootstrap results:

| Dataset | Condition | ΔWER | 95% CI |
|---|---|---:|---:|
| test-clean | MP3 16k | +0.91 pp | [+0.59, +1.29] |
| test-clean | Opus 12k | +0.32 pp | [+0.09, +0.57] |
| test-clean | Opus 8k | +1.15 pp | [+0.81, +1.51] |
| test-other | MP3 16k | +3.74 pp | [+3.07, +4.43] |
| test-other | Opus 8k | +6.63 pp | [+5.71, +7.62] |

Selected error-type increases:

| Condition | ΔS | ΔD | ΔI |
|---|---:|---:|---:|
| test-clean MP3 16k | +85 | +15 | -7 |
| test-clean Opus 8k | +104 | +14 | -1 |
| test-other MP3 16k | +280 | +42 | +6 |
| test-other Opus 8k | +490 | +62 | +30 |

The spectrogram analysis shows substantial attenuation and modification of high-frequency spectral content under very low bitrate compression. These observations are treated as supporting evidence rather than proof of direct causation.

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
python src/integrated_analysis.py
python src/failure_case_analysis.py
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
│   └── comparison_analysis.py
├── results/
│   ├── baseline_results.csv
│   ├── mp3_32k_results.csv
│   ├── test-clean_experiment_details.csv
│   ├── test-other_experiment_details.csv
│   ├── test-clean_summary_results.csv
│   ├── test-other_summary_results.csv
│   ├── bootstrap_results.csv
│   ├── error_analysis/
│   ├── comparison_analysis/
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
- signal-level spectral distortion and effective-bandwidth analysis;
- Wav2Vec2 representation drift at early, middle and late layers;
- an integrated signal → representation → recognition analysis;
- representative individual failure cases.

Any additional neural-codec experiment is treated as an extension rather than a requirement for the core project.

## Project Contribution

The main contribution of this project is not simply a comparison of MP3 and Opus recognition accuracy.

Instead, the project investigates why Wav2Vec2 remains robust under lossy compression by linking three levels of analysis:

1. **Signal level** — how compression changes the spectral characteristics of the speech signal;
2. **Representation level** — how these changes propagate through early, middle and late Wav2Vec2 hidden representations;
3. **Task level** — when these changes become large enough to produce measurable ASR degradation.

The results suggest that substantial signal distortion can occur before recognition performance degrades strongly. Under more severe compression, representation drift increases, particularly in deeper layers, together with larger WER increases.

This provides a more informative explanation of compression robustness than WER-only codec comparison.

## Next Steps

- review the matched-bitrate codec comparison and speech-difficulty comparison;
- collect and organise literature relevant to ASR robustness and lossy compression;
- incorporate project feedback;
- determine whether any additional robustness experiment is necessary;
- prepare the final research report;
- prepare the project demonstration video.

A neural audio codec, EnCodec, was evaluated as an optional extension after the MP3/Opus analysis was completed. The purpose of this extension was to test whether the observed signal → representation → WER relationship also generalises to a neural codec architecture.

The EnCodec experiment used a separate 16 → 24 → 16 kHz resampling control to distinguish codec effects from sample-rate conversion effects. The control produced only a negligible WER change, while EnCodec showed progressively larger representation drift and recognition degradation as bitrate was reduced from 24 kbps to 6 kbps and finally to 1.5 kbps.

If included, EnCodec will not be treated as a separate codec-ranking experiment. It will be used only to test whether the signal → representation → WER relationship observed for MP3 and Opus also appears under a neural codec architecture.

Because the standard 24 kHz mono EnCodec model operates at a different sample rate from LibriSpeech and Wav2Vec2, any EnCodec experiment will include an uncompressed 16 → 24 → 16 kHz resampling control. This prevents resampling effects from being incorrectly attributed to neural codec compression.