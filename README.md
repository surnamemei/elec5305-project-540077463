# Evaluating the Impact of Lossy Audio Compression on Automatic Speech Recognition Performance

ELEC5305 project investigating the robustness of automatic speech recognition (ASR) under MP3 and Opus lossy audio compression.

**Project Site:**  
https://surnamemei.github.io/elec5305-project-540077463/

**Proposal:**  
[ELEC5305 Project Proposal v1.pdf](ELEC5305%20Project%20Proposal%20v1.pdf)

---

## Research Question

> **How robust is a modern automatic speech recognition system to lossy audio compression, and at what bitrate or compression ratio does recognition performance begin to degrade significantly?**

Secondary question:

> **Does lossy compression have a greater impact when the underlying speech is already more difficult for the ASR system to recognise?**

---

## Experimental Setup

The project uses two LibriSpeech evaluation subsets:

- `test-clean`
- `test-other`

For each subset, **500 utterances** are selected using the fixed random seed `5305` for reproducibility.

The same pretrained **Wav2Vec2** ASR system is used for every condition.

### Compression Conditions

| Codec | Bitrates |
|---|---|
| WAV | Uncompressed baseline |
| MP3 | 128, 64, 32, 24, 16 kbps |
| Opus | 64, 32, 16, 12, 8 kbps |

### Main Metrics

- Word Error Rate (WER)
- WER change relative to WAV baseline (ΔWER)
- Average compression ratio
- 95% paired bootstrap confidence interval
- Substitution, deletion and insertion error counts
- Sentence-level and local spectrogram comparison

---

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

## Next Steps

- review the matched-bitrate codec comparison and speech-difficulty comparison;
- collect and organise literature relevant to ASR robustness and lossy compression;
- incorporate project feedback;
- determine whether any additional robustness experiment is necessary;
- prepare the final research report;
- prepare the project demonstration video.

