---
title: "ELEC5305 Project Proposal — Version 3"
---

## 1. Project Title

**Why Is Wav2Vec2 Robust to Lossy Audio Compression? A Signal- and Representation-Level Study of MP3, Opus and EnCodec**

## 2. Student Information

| | |
|---|---|
| **Full Name** | Jinghang Mei |
| **Student ID (SID)** | 540077463 |
| **GitHub Username** | surnamemei |
| **GitHub Project Link** | <https://github.com/surnamemei/elec5305-project-540077463> |
| **GitHub Pages Site** | <https://surnamemei.github.io/elec5305-project-540077463/> |

## 3. Project Overview

**Problem.** Speech is routinely compressed with lossy codecs (MP3, Opus, and increasingly neural codecs) before it reaches an automatic speech recognition (ASR) system. Codec robustness is usually reported only as a change in word error rate (ΔWER). This shows *when* recognition degrades but not *why*.

**Importance.** Our pilot found that Wav2Vec2 was almost unaffected by MP3 down to 32 kbps and by Opus down to 16 kbps, even though the audio was visibly altered. Understanding this robustness, and where it breaks, matters for choosing bitrates in voice and ASR pipelines.

**Proposed solution.** We follow the codec perturbation through three levels on the same utterances:

1. signal distortion;
2. hidden-representation drift at every layer of a fixed Wav2Vec2 model;
3. recognition errors.

**Research question:** *How do MP3 and Opus compression alter the acoustic signal and the internal representations of Wav2Vec2, and which codec-induced distortions are associated with the onset of ASR errors?* A secondary question asks whether Wav2Vec2 stays robust because codec changes are attenuated before they reach its upper layers.

## 4. Background and Motivation

**Codecs and ASR.** Earlier work showed that low-bitrate MPEG coding degraded HMM recognisers [5]. For a neural recogniser, MP3 64 kbps and Opus 24 kbps caused no loss, while MP3 at 23 kbps did [4]. Codec benchmarks such as Codec-SUPERB rank codecs by WER after resynthesis [7]. All of these report *whether* recognition degrades, not *where* inside the model a perturbation is absorbed.

**What the layers encode.** Layer-wise analyses show that Wav2Vec2 encodes acoustic information in early layers and phonetic and word information in higher layers. Fine-tuning for ASR changes the top layers most [3].

**Internal measures.** In speech enhancement, a distance inside the acoustic model tracks WER better than perceptual metrics [8].

**Why this topic.** Together, this literature motivated comparing signal-level and representation-level measures as indicators of ASR failure. It also connects ELEC5305 signal-processing concepts (spectral analysis, bandwidth, perceptual coding) with modern learned representations.

**Changes since v2.** Following teaching-staff feedback and a review of the analysis, v3 changes the measurement methods:

- spectral distortion uses a dynamic-range floor;
- codec bandwidth is measured directly;
- hidden states are standardised before comparison, because Wav2Vec2 layers are strongly anisotropic [6];
- confidence intervals resample speakers rather than utterances [9].

## 5. Proposed Methodology

**Tools.** Python 3.12, PyTorch/torchaudio, FFmpeg 6.1.1 (libmp3lame 3.100, libopus 1.4), JiWER and EnCodec, run on a local GPU.

**Data.** LibriSpeech test-clean and test-other [2]: 500 fixed utterances per subset (seed 5305), reused across all conditions, so every comparison is paired.

**Codecs.**

- MP3: 128, 64, 32, 24 and 16 kbps, constant bitrate.
- Opus: 64, 32, 16, 12, 8 and 6 kbps, default variable bitrate [10]. The coding mode libopus actually chose is read from the packet headers.
- Bitrate is measured from the encoded file size.

**Recogniser.** Frozen `WAV2VEC2_ASR_BASE_960H` with greedy CTC decoding [1]. It is never retrained.

**Signal processing.**

- Alignment by normalised cross-correlation.
- STFT with a 25 ms Hann window and 10 ms hop.
- Log-spectral distance (LSD), with log spectra floored 80 dB below the reference peak.
- Frequency-dependent distortion.
- Retained bandwidth: the highest frequency at which codec power stays within 20 dB of the original.

**Representation analysis.** Hidden states are taken from the convolutional encoder and all 12 transformer layers. Each dimension is standardised using statistics from uncompressed calibration speech [6]. Drift is 1 − frame-wise cosine similarity.

**Statistics.**

- WER and ΔWER in percentage points, with a substitution/deletion/insertion breakdown.
- 95% confidence intervals from a speaker-level paired bootstrap [9].
- A predictor analysis comparing signal and representation measures against ΔWER, both across conditions and within a single condition.
- Failure cases selected by a fixed rule.

**Extension.** EnCodec at 24, 6 and 1.5 kbps [11], with an uncompressed 16→24→16 kHz resampling control.

## 6. Expected Outcomes

- A reproducible pipeline covering 11 codec conditions × 2 subsets, including validation scripts that re-check every reported number.
- Performance metrics: WER and ΔWER with confidence intervals; LSD and retained bandwidth; layer-wise drift; and the correlation of each measure with ΔWER.
- A final research report (about 10 pages) with figures linking signal distortion, representation drift and recognition errors.
- Honest limits: the analysis is correlational and uses a single recogniser.
- GitHub documentation, a GitHub Pages site, and a demonstration video.

Preliminary results (Appendix) indicate the following:

- Large spectral changes can leave WER unchanged.
- The largest degradations occur when the codec cutoff enters the speech band, especially on test-other.
- Late-layer drift tracks ΔWER more closely than signal distortion.

## 7. Timeline (Weeks 1–13)

| Week | Task | Status |
|------|--------------------------------------------------------------------|------------|
| 1–2 | Topic selection and research question | Completed |
| 3–5 | Literature review, dataset setup, baseline ASR | Completed |
| 6–9 | MP3/Opus pipeline, bitrate sweep, bootstrap, error analysis | Completed |
| 10–11 | Signal and representation analysis, measurement corrections, predictor analysis, EnCodec extension | Completed |
| 12–13 | Final report, GitHub documentation, demonstration video | In progress |

## 8. References

[1] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 12449–12460.

[2] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "Librispeech: An ASR corpus based on public domain audio books," in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210, doi: 10.1109/ICASSP.2015.7178964.

[3] A. Pasad, J.-C. Chou, and K. Livescu, "Layer-wise analysis of a self-supervised speech representation model," in *Proc. IEEE ASRU*, 2021, pp. 914–921, doi: 10.1109/ASRU51503.2021.9688093.

[4] A. Narayanan *et al.*, "Toward domain-invariant speech recognition via large scale training," in *Proc. IEEE SLT*, 2018, pp. 441–447, doi: 10.1109/SLT.2018.8639610.

[5] L. Besacier, C. Bergamini, D. Vaufreydaz, and E. Castelli, "The effect of speech and audio compression on speech recognition performance," in *Proc. IEEE MMSP*, 2001, pp. 301–306, doi: 10.1109/MMSP.2001.962750.

[6] W. Timkey and M. van Schijndel, "All bark and no bite: Rogue dimensions in transformer language models obscure representational quality," in *Proc. EMNLP*, 2021, pp. 4527–4546, doi: 10.18653/v1/2021.emnlp-main.372.

[7] H. Wu *et al.*, "Codec-SUPERB: An in-depth analysis of sound codec models," in *Findings of ACL 2024*, pp. 10330–10348, doi: 10.18653/v1/2024.findings-acl.616.

[8] L. Chai, J. Du, Q.-F. Liu, and C.-H. Lee, "A cross-entropy-guided measure (CEGM) for assessing speech recognition performance and optimizing DNN-based speech enhancement," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 29, pp. 106–117, 2021, doi: 10.1109/TASLP.2020.3036783.

[9] M. Bisani and H. Ney, "Bootstrap estimates for confidence intervals in ASR performance evaluation," in *Proc. IEEE ICASSP*, 2004, pp. I-409–I-412, doi: 10.1109/ICASSP.2004.1326009.

[10] J.-M. Valin, K. Vos, and T. Terriberry, "Definition of the Opus audio codec," IETF RFC 6716, 2012, doi: 10.17487/RFC6716.

[11] A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, "High fidelity neural audio compression," *Transactions on Machine Learning Research*, 2023.

## 9. Appendix: Preliminary Results

**Table A1.** Selected results on 500 utterances per subset. ΔWER is in percentage points (pp) with 95% speaker-level bootstrap confidence intervals. LSD, retained bandwidth and standardised layer-12 drift are test-clean means. WAV WER: 3.17% (test-clean), 8.26% (test-other).

| Condition | LSD (dB) | Band (kHz) | L12 drift | ΔWER test-clean (pp) | ΔWER test-other (pp) |
|------------|------:|------:|------:|--------------------:|----------------------:|
| MP3 24 kbps | 9.2 | 5.8 | 0.041 | +0.02 [−0.26, +0.31] | +1.42 [+0.89, +1.99] |
| Opus 16 kbps | 5.4 | 8.0 | 0.022 | +0.01 [−0.21, +0.24] | +0.64 [+0.30, +0.97] |
| Opus 12 kbps | 5.9 | 8.0 | 0.035 | +0.32 [+0.05, +0.60] | +1.33 [+0.85, +1.82] |
| Opus 8 kbps | 11.4 | 4.6 | 0.086 | +1.15 [+0.82, +1.48] | +6.63 [+4.92, +8.60] |
| Opus 6 kbps | 11.6 | 4.7 | 0.130 | +2.28 [+1.82, +2.80] | +11.53 [+9.07, +14.18] |

**Across conditions:** the Pearson correlation with ΔWER is 0.96 (test-clean) and 0.97 (test-other) for layer-12 drift, against 0.75 and 0.79 for LSD.

**Within a fixed condition:** the association of layer-12 drift with which utterances fail is modest (mean Spearman ρ = 0.09 and 0.23). It is near zero for all signal measures.

**AI assistance.** Generative AI tools were used for brainstorming, writing and debugging analysis code, literature-search assistance, and drafting and editing text. All experiments were run on the author's machine, numerical results were checked against output files, references were verified against original sources, and the author takes responsibility for the work.
