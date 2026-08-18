# Evaluating the Impact of Lossy Audio Compression on Automatic Speech Recognition Performance

## Project Overview

This ELEC5305 project investigates how lossy audio compression affects the robustness of automatic speech recognition (ASR).

Speech recordings from LibriSpeech are compressed using MP3 and Opus at a range of bitrates and then decoded and processed using the same pretrained Wav2Vec2 ASR model. Recognition performance is evaluated using Word Error Rate (WER), while compression efficiency is measured using the average compression ratio relative to uncompressed WAV audio.

The main research question is:

> **How robust is a modern automatic speech recognition system to lossy audio compression, and at what bitrate or compression ratio does recognition performance begin to degrade significantly?**

A secondary question is:

> **Does lossy compression have a greater impact when the underlying speech is already more difficult for the ASR system to recognise?**

## Experimental Setup

Two LibriSpeech evaluation subsets are used:

- `test-clean`
- `test-other`

For each subset, 500 utterances are selected using a fixed random seed (`5305`) to ensure reproducibility.

### ASR System

- Pretrained Wav2Vec2 model
- PyTorch / TorchAudio
- 16 kHz speech input
- Word Error Rate calculated using JiWER

### Compression Conditions

#### WAV
- Uncompressed baseline

#### MP3
- 128 kbps
- 64 kbps
- 32 kbps
- 24 kbps
- 16 kbps

#### Opus
- 64 kbps
- 32 kbps
- 16 kbps
- 12 kbps
- 8 kbps

FFmpeg is used for encoding and decoding the compressed files.

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

## Statistical Analysis

A paired bootstrap analysis with 2000 resamples is used to estimate 95% confidence intervals for WER change relative to the WAV baseline.

Examples of compression conditions with confidence intervals entirely above zero include:

- `test-clean`, MP3 16 kbps: ΔWER ≈ +0.91 pp, 95% CI [+0.59, +1.29]
- `test-clean`, Opus 12 kbps: ΔWER ≈ +0.32 pp, 95% CI [+0.09, +0.57]
- `test-clean`, Opus 8 kbps: ΔWER ≈ +1.15 pp, 95% CI [+0.81, +1.51]
- `test-other`, MP3 16 kbps: ΔWER ≈ +3.74 pp, 95% CI [+3.07, +4.43]
- `test-other`, Opus 8 kbps: ΔWER ≈ +6.63 pp, 95% CI [+5.71, +7.62]

The results indicate that severe lossy compression produces consistent ASR degradation, while moderate compression has a much smaller practical effect.

## Error Analysis

Per-utterance analysis shows that severe compression increases the number of utterances whose recognition becomes worse.

### Selected severe conditions

- `test-clean`, MP3 16 kbps:
  - 65 new errors
  - 101 worsened samples
  - 16 recovered samples

- `test-clean`, Opus 8 kbps:
  - 64 new errors
  - 114 worsened samples
  - 15 recovered samples

- `test-other`, MP3 16 kbps:
  - 88 new errors
  - 215 worsened samples
  - 8 recovered samples

- `test-other`, Opus 8 kbps:
  - 98 new errors
  - 268 worsened samples
  - 11 recovered samples

Additional word-level analysis shows that most of the extra recognition errors are substitutions.

| Condition | Δ Substitutions | Δ Deletions | Δ Insertions |
|---|---:|---:|---:|
| test-clean MP3 16k | +85 | +15 | -7 |
| test-clean Opus 8k | +104 | +14 | -1 |
| test-other MP3 16k | +280 | +42 | +6 |
| test-other Opus 8k | +490 | +62 | +30 |

This suggests that aggressive compression primarily causes the ASR system to confuse one word with another, rather than simply deleting or inserting words.

## Signal-Level Analysis

Sentence-level and local spectrogram comparisons have been produced for representative cases where the WAV baseline was recognised correctly but the compressed version introduced a new recognition error.

The low-bitrate MP3 and Opus cases show substantial attenuation and modification of high-frequency spectral content. These signal-level changes are consistent with the observed increase in substitution errors, although they do not by themselves establish a direct causal relationship.

Representative examples include:

- `test-other`, MP3 16 kbps
- `test-other`, Opus 8 kbps
- `test-clean`, MP3 16 kbps

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
│   └── local_spectrogram_analysis.py
├── results/
│   ├── test-clean_summary_results.csv
│   ├── test-other_summary_results.csv
│   ├── bootstrap_results.csv
│   ├── error_analysis/
│   └── figures/
├── README.md
├── index.md
└── proposal.pdf
```

## Current Findings

The current results support three main observations:

1. Moderate lossy compression has little practical effect on ASR performance for clean speech.
2. Severe compression introduces consistent degradation, with clearer thresholds at low bitrates.
3. More challenging speech is substantially more vulnerable to aggressive lossy compression than cleaner speech.

## Next Steps

The next stage will focus on:

- refining plots and statistical visualisation
- selecting the strongest case-study figures
- comparing MP3 and Opus robustness more systematically
- documenting limitations and experimental assumptions
- preparing the final report and demonstration

## Project Site

https://surnamemei.github.io/elec5305-project-540077463/
