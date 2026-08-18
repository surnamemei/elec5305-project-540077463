---
layout: default
title: ELEC5305 Project
---

# Evaluating the Impact of Lossy Audio Compression on Automatic Speech Recognition Performance

## Overview

This project investigates how lossy audio compression affects the robustness of automatic speech recognition (ASR).

The same LibriSpeech speech samples are encoded using MP3 and Opus at different bitrates, decoded, and then processed using a fixed pretrained Wav2Vec2 ASR model. Recognition performance is measured using Word Error Rate (WER), while compression efficiency is measured using the compression ratio relative to uncompressed WAV audio.

### Research Question

> **How robust is a modern automatic speech recognition system to lossy audio compression, and at what bitrate or compression ratio does recognition performance begin to degrade significantly?**

A secondary question is whether compression has a greater impact on speech that is already more difficult for the ASR system to recognise.

## Experimental Design

The current evaluation uses:

- LibriSpeech `test-clean`
- LibriSpeech `test-other`
- 500 fixed random utterances from each subset
- fixed random seed `5305`
- pretrained Wav2Vec2 ASR
- FFmpeg MP3 and Opus encoding
- WER evaluation using JiWER
- compression ratio measurement
- paired bootstrap analysis
- per-utterance error analysis
- spectrogram comparison

### Compression Conditions

**MP3:** 128, 64, 32, 24, and 16 kbps

**Opus:** 64, 32, 16, 12, and 8 kbps

**Baseline:** uncompressed WAV

## Main Results

### test-clean

The uncompressed WAV baseline achieved approximately **3.17% WER**.

Recognition performance remained close to the baseline under most moderate compression settings. More severe degradation appeared at low bitrates:

- MP3 16 kbps: **4.08% WER**
- Opus 12 kbps: **3.49% WER**
- Opus 8 kbps: **4.32% WER**

### test-other

The uncompressed WAV baseline achieved approximately **8.26% WER**.

The harder `test-other` subset was substantially more sensitive to compression:

- MP3 24 kbps: **9.69% WER**
- MP3 16 kbps: **12.00% WER**
- Opus 12 kbps: **9.60% WER**
- Opus 8 kbps: **14.89% WER**

The strongest degradation observed so far is Opus at 8 kbps on `test-other`, where WER increases by approximately **6.63 percentage points** relative to the WAV baseline.

## Bootstrap Analysis

A paired bootstrap analysis with 2000 resamples was used to estimate 95% confidence intervals for the change in WER relative to WAV.

Selected results include:

| Dataset | Condition | ΔWER | 95% CI |
|---|---|---:|---:|
| test-clean | MP3 16k | +0.91 pp | [+0.59, +1.29] |
| test-clean | Opus 12k | +0.32 pp | [+0.09, +0.57] |
| test-clean | Opus 8k | +1.15 pp | [+0.81, +1.51] |
| test-other | MP3 16k | +3.74 pp | [+3.07, +4.43] |
| test-other | Opus 8k | +6.63 pp | [+5.71, +7.62] |

These intervals show that severe compression produces consistent recognition degradation.

## Error-Type Analysis

The error analysis indicates that most of the additional recognition errors produced by aggressive compression are substitutions.

| Condition | ΔS | ΔD | ΔI |
|---|---:|---:|---:|
| test-clean MP3 16k | +85 | +15 | -7 |
| test-clean Opus 8k | +104 | +14 | -1 |
| test-other MP3 16k | +280 | +42 | +6 |
| test-other Opus 8k | +490 | +62 | +30 |

This suggests that low-bitrate codec distortion most often causes the ASR model to confuse one word with another rather than simply inserting or deleting words.

## Spectrogram Case Studies

Representative sentence-level and word-level spectrogram comparisons show substantial attenuation and modification of high-frequency spectral content under very low bitrate compression.

The examples are consistent with the observed increase in substitution errors, but the spectral differences are treated as supporting evidence rather than proof of direct causation.

Current case studies include:

- `test-other` + MP3 16 kbps
- `test-other` + Opus 8 kbps
- `test-clean` + MP3 16 kbps

## Current Interpretation

The project currently supports three main findings:

1. **Moderate compression is relatively safe for clean speech.**
2. **Severe low-bitrate compression produces measurable ASR degradation.**
3. **More challenging speech is more vulnerable to compression-induced distortion.**

The results also suggest that Opus can maintain strong ASR performance at relatively high compression ratios, but very aggressive settings such as 8 kbps produce substantial degradation.

## Next Steps

The next stage will focus on:

- refining final figures and tables
- selecting representative case-study plots
- comparing codec efficiency more systematically
- documenting limitations
- preparing the final report and project demonstration

## Repository

GitHub repository:

https://github.com/surnamemei/elec5305-project-540077463
