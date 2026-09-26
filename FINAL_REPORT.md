# Why Is Wav2Vec2 Robust to Lossy Audio Compression? A Signal- and Representation-Level Study of MP3, Opus and EnCodec

**Jinghang Mei (SID 540077463)** — ELEC5305, The University of Sydney  
Code and results: https://github.com/surnamemei/elec5305-project-540077463

*Generated from `report/FINAL_REPORT.pandoc.md` by `report/build_report.sh`; a typeset version is in `FINAL_REPORT.pdf`.*

## Abstract

Speech is increasingly stored and transmitted with lossy codecs before it reaches an automatic speech recognition (ASR) system, yet codec robustness is usually reported only as a word error rate (WER). This study asks *why* a fixed Wav2Vec2 recogniser tolerates compression and what changes when it stops doing so. On 500 paired utterances from each of LibriSpeech test-clean and test-other, eleven MP3 and Opus settings (6–128 kbps) were analysed at three levels: signal distortion (log-spectral distance, LSD, and retained bandwidth), standardised hidden-representation drift at every Wav2Vec2 layer, and WER with speaker-level bootstrap confidence intervals. MP3 at 24 kbps changed the spectrum by 9.2 dB LSD and removed content above 5.8 kHz without a measurable test-clean WER change (+0.02 pp, 95% CI \[−0.26, +0.31\]). The largest degradations coincided with the codec cutoff entering the speech band, reaching +11.53 pp on test-other at Opus 6 kbps. Across conditions, layer-12 drift tracked ΔWER more linearly than LSD (Pearson r = 0.96–0.97 versus 0.75–0.79), although both ranked conditions similarly. Within a fixed condition, only late-layer drift showed a modest association with which utterances failed (mean Spearman ρ = 0.09 and 0.23). A neural-codec extension (EnCodec) followed the same pattern. Because late layers feed the CTC output directly, these associations are not causal evidence.

## 1. Introduction

Voice messages, conferencing platforms, archives and edge devices commonly store or transmit speech with perceptual codecs such as MP3 [\[1\]](#ref-brandenburg1999mp3) and Opus [\[2\]](#ref-rfc6716), and increasingly with neural codecs such as EnCodec [\[3\]](#ref-defossez2023encodec). These codecs are designed around human perception: they discard or coarsely quantise spectral detail that a listener is unlikely to notice. An ASR model is not a human listener, so it is not obvious in advance whether the discarded information matters to it.

Codec effects on ASR are normally summarised by a single number, the change in word error rate (ΔWER) [\[4\]](#ref-besacier2001effect), [\[5\]](#ref-narayanan2018domain), [\[6\]](#ref-wu2024codecsuperb). A pilot version of this project did the same and found that Wav2Vec2 [\[7\]](#ref-baevski2020wav2vec) was almost unaffected by MP3 down to 32 kbps and by Opus down to 16 kbps. A flat WER curve is useful for deployment decisions, but it cannot distinguish two explanations:

- the codec barely changes the signal; or
- the signal changes substantially and the recogniser absorbs the change.

These explanations predict different failure points, and WER alone cannot separate them.

Layer-wise studies show that Wav2Vec2 layers encode different kinds of information [\[8\]](#ref-pasad2021layerwise). In speech enhancement, distances measured inside an acoustic model track WER better than perceptual quality metrics [\[9\]](#ref-chai2021cegm). What is missing for codecs is a controlled analysis that follows the perturbation through the whole chain: from the signal, through every layer of a fixed recogniser, to the recognition errors.

This report addresses the following question:

> *How do MP3 and Opus compression alter the acoustic signal and the internal representations of Wav2Vec2, and which codec-induced distortions are associated with the onset of ASR errors?*

A secondary question is whether Wav2Vec2 is robust because the codecs preserve recogniser-relevant information even when spectral distortion is already substantial.

The contributions are:

1.  A controlled multi-level analysis of codec-induced robustness in a fixed Wav2Vec2 ASR system. It links signal distortion, standardised layer-wise representation drift and recognition errors on the same 500 utterances per subset for eleven MP3/Opus settings, with speaker-level bootstrap confidence intervals.
2.  A predictor analysis that separates two questions: which measure tracks degradation *across* compression conditions, and which identifies *which utterances* fail *within* a condition.
3.  Two measurement choices this kind of analysis requires, each shown to change the conclusions: a dynamic-range floor for LSD, and per-dimension standardisation of hidden states before cosine comparison.
4.  A neural-codec extension (EnCodec, with a resampling control) testing whether the pattern depends on the codec family.

## 2. Related Work

**Layer-wise structure of self-supervised speech models.** Wav2Vec2 maps raw audio through a convolutional feature encoder and a transformer, and is fine-tuned for ASR with a connectionist temporal classification (CTC) loss [\[7\]](#ref-baevski2020wav2vec), [\[10\]](#ref-graves2006ctc). Using canonical correlation and mutual-information probes, Pasad et al. found an acoustic-to-linguistic hierarchy in the pre-trained model: the final convolutional and first transformer layers correlate most with mel-spectrogram features, phonetic and word information peak in intermediate layers, and the deepest layers revert towards the input (“autoencoder-style” behaviour) [\[8\]](#ref-pasad2021layerwise). Fine-tuning for ASR changes the top layers most and removes this reversion. A comparison across self-supervised models found that these trends depend on the pre-training objective [\[11\]](#ref-pasad2023comparative). Such probing studies characterise what each layer encodes for clean input. They do not examine how an input *perturbation* propagates through the layers. Two decisions in this project follow from them: drift is measured at every layer rather than only the output, and the final layers of a CTC-fine-tuned model are treated as close to the output by construction.

**ASR robustness to degradation and compression.** Robustness has traditionally been benchmarked with additive noise, e.g. Aurora [\[12\]](#ref-hirsch2000aurora). For HMM recognisers, Besacier et al. found that MPEG audio coding degraded recognition at low bitrates whereas telephony speech codecs (GSM, G.711) did not [\[4\]](#ref-besacier2001effect). For a neural recogniser, Narayanan et al. reported no loss at MP3 64 kbps and Opus 24 kbps but a clear loss at MP3 23 kbps (10.5% → 13.6% WER), largely recovered by codec-augmented training [\[5\]](#ref-narayanan2018domain). Drude et al. showed that Opus bitrate trades off directly against far-field WER [\[13\]](#ref-drude2021opus). For self-supervised models, Wav2Vec2 degrades under domain mismatch between pre-training, fine-tuning and test data [\[14\]](#ref-hsu2021robust), and SUPERB-style models degrade under noise and reverberation, although codecs were not tested [\[15\]](#ref-huang2022distortion). Training on very large, diverse data improves robustness overall [\[16\]](#ref-radford2023whisper). Across this literature, robustness is measured almost entirely through WER. The studies establish *when* recognition degrades (Narayanan et al.’s MP3 onset is consistent with the results below), but not *where* in the model the perturbation is absorbed. This project therefore keeps WER as the outcome and adds signal-level and representation-level measurements on the same utterances.

**Perceptual codecs and bandwidth.** MP3 quantises filter-bank coefficients so that quantisation noise stays below a psychoacoustic masking threshold. When the bit budget is insufficient, high-frequency bandwidth is sacrificed [\[1\]](#ref-brandenburg1999mp3). Opus combines a linear-prediction coder (SILK) with a transform coder (CELT). It operates at defined audio bandwidths (narrowband 4 kHz, wideband 8 kHz and wider), and for 20 ms frames its specification gives 8–12 kbit/s as the operating range for narrowband speech and 16–20 kbit/s for wideband speech [\[2\]](#ref-rfc6716). Both codecs therefore change the signal’s *bandwidth*, not only its fine structure. Spectral distortion measures have a long history in speech processing [\[17\]](#ref-gray1980distortion). A global log-spectral distance, however, cannot distinguish removed bandwidth from in-band quantisation noise. This motivated measuring the effective cutoff alongside LSD, and reading the coding mode Opus actually used from its bitstream.

**Neural codecs and downstream evaluation.** SoundStream [\[18\]](#ref-zeghidour2022soundstream) and EnCodec [\[3\]](#ref-defossez2023encodec) encode audio with a learned encoder and residual vector quantisation and reconstruct it with a learned decoder. The output can be perceptually good while differing from the input waveform. Codec benchmarks assess content preservation by the WER of a fixed recogniser on resynthesised speech. Codec-SUPERB (Whisper WER on LibriSpeech) finds that WER falls as bitrate rises [\[6\]](#ref-wu2024codecsuperb), and ESPnet-Codec and AudioCodecBench (a preprint) report similar WER-based evaluations [\[19\]](#ref-shi2024espnetcodec), [\[20\]](#ref-wang2025audiocodecbench). These benchmarks *rank* codecs but do not explain why a codec harms recognition. Here EnCodec is used only to test whether the relationships found for MP3 and Opus hold for a codec with a different kind of distortion.

**Representation change and its link to the task.** Comparing hidden states requires a similarity measure. Transformer representations are often anisotropic: vectors occupy a narrow cone, so unrelated inputs have high cosine similarity, generally more so in upper layers [\[21\]](#ref-ethayarajh2019contextual). Timkey and van Schijndel showed that one to three “rogue” dimensions can dominate cosine similarity in transformer language models, and that per-dimension standardisation corrects this [\[22\]](#ref-timkey2021bark). Centred kernel alignment instead compares whole representation spaces [\[23\]](#ref-kornblith2019similarity), but it does not give the frame-paired, per-utterance measure needed here.

In speech, Zhu et al. found that the cosine similarity of Wav2Vec2 outputs between noisy and clean input increases with signal-to-noise ratio [\[24\]](#ref-zhu2022noiserobust). Wav2vec-Switch improves noise robustness by training clean and noisy representations to agree [\[25\]](#ref-wang2022wav2vecswitch). Chai et al. showed that a distance between an acoustic model’s internal posteriors correlates with WER better than PESQ or STOI [\[9\]](#ref-chai2021cegm). Iwamoto et al. found that processing artefacts, not residual noise, drive ASR errors after speech enhancement, so the *type* of distortion matters more than its amount [\[26\]](#ref-iwamoto2022artifacts). These studies concern noise and enhancement, use a single layer or the output, and apply raw cosine similarity. This project applies the idea to codecs at every layer, uses standardised cosine similarity after verifying that Wav2Vec2 shows the anisotropy that motivates it, and separates across-condition tracking from within-condition prediction.

**Statistical evaluation.** WER differences on shared test data are best compared with paired resampling [\[27\]](#ref-efron1993bootstrap). For speaker-independent ASR, Bisani and Ney recommend resampling whole *speakers*, because utterance-level intervals are too optimistic [\[28\]](#ref-bisani2004bootstrap). All intervals below follow this recommendation.

## 3. Methodology

### 3.1 Data and design

LibriSpeech [\[29\]](#ref-panayotov2015librispeech) provides read English speech at 16 kHz. Test-clean (2,620 utterances) and test-other (2,939) contain lower- and higher-difficulty speakers respectively. From each subset, 500 utterances were drawn once with Python’s `random.sample` and seed 5305. The draw contains 40 speakers and 10,168 reference words (test-clean), and 33 speakers and 8,775 words (test-other).

Every codec condition, signal measurement and representation measurement uses exactly these utterances, so all comparisons are paired. This was verified programmatically for every analysis. The LibriSpeech source recordings were themselves distributed as MP3 [\[29\]](#ref-panayotov2015librispeech), so the “WAV” reference is the uncompressed decoding of already-coded audio.

### 3.2 Codecs

Each utterance was written as 16-bit PCM WAV (256 kbps) and encoded with FFmpeg 6.1.1.

- **MP3:** libmp3lame (LAME 3.100) at 128, 64, 32, 24 and 16 kbps. With a 16 kHz mono input this gives constant-bitrate MPEG-2 Layer III.
- **Opus:** libopus 1.4 at 64, 32, 16, 12, 8 and 6 kbps, with FFmpeg defaults (variable bitrate, `application=audio`, 20 ms frames). The output was decoded at 48 kHz and resampled to 16 kHz.

The Opus coding mode was read from the table-of-contents byte of every packet [\[2\]](#ref-rfc6716) for 50 utterances. It was CELT wideband at 64 kbps, SILK wideband at 32–12 kbps (≥ 99.97% of packets) and SILK narrowband at 8 and 6 kbps (100%). Measured bitrate is 8 × encoded bytes / duration, including container overhead.

### 3.3 Recogniser and task-level metrics

The recogniser was torchaudio’s `WAV2VEC2_ASR_BASE_960H`: Wav2Vec2 Base fine-tuned with CTC on 960 h of LibriSpeech [\[7\]](#ref-baevski2020wav2vec). It used greedy CTC decoding without a language model and was never retrained. Corpus WER was computed with JiWER 4.0 against the upper-case LibriSpeech transcripts, which match the model’s output alphabet:

``` math
\mathrm{WER}=\frac{S+D+I}{N},\qquad \Delta\mathrm{WER}=\mathrm{WER}_{\text{codec}}-\mathrm{WER}_{\text{WAV}},
```

where $`S`$, $`D`$, $`I`$ are substitutions, deletions and insertions summed over utterances and $`N`$ is the number of reference words. ΔWER is in percentage points (pp).

95% confidence intervals come from a paired percentile bootstrap with 2,000 resamples of *speakers*, each resample keeping all of a speaker’s utterances. The same resample was applied to WAV and codec outputs [\[28\]](#ref-bisani2004bootstrap). An utterance-level bootstrap gave narrower intervals but the same conclusion about whether zero was excluded, in all 22 comparisons.

### 3.4 Signal-level measures

Decoded signals were aligned to the reference by normalised cross-correlation. The lag was 0 samples for MP3 and at most 2 samples (0.125 ms) for Opus. Short-time Fourier transforms used a 25 ms Hann window, 10 ms hop and 512-point FFT. Log magnitudes were floored at 80 dB below the reference spectrogram peak:

``` math
L(k,m)=\max\!\big(20\log_{10}|X(k,m)|,\;L^{\text{ref}}_{\max}-80\,\text{dB}\big),\qquad
\mathrm{LSD}=\frac{1}{M}\sum_{m=1}^{M}\sqrt{\frac{1}{K}\sum_{k=1}^{K}\big(L_{\text{codec}}(k,m)-L_{\text{WAV}}(k,m)\big)^{2}}.
```

Without the floor, bins that MP3 sets exactly to zero map to $`20\log_{10}\varepsilon`$ and dominate the average. A pilot version without it ranked MP3 128 kbps as more distorted than Opus 16 kbps.

The **retained bandwidth** is the highest frequency at which the decoded signal’s long-term power stays within 20 dB of the reference:

``` math
f_{\text{ret}}=\max\{f_k : 10\log_{10}(\bar P_{\text{codec}}(k)/\bar P_{\text{WAV}}(k))>-20\ \text{dB}\}.
```

It replaces a cumulative-energy roll-off used in the pilot, which did not detect low-pass filtering because speech energy is concentrated below about 3 kHz. Frequency-resolved distortion is kept as supporting material in the repository.

### 3.5 Representation-level measures

Hidden states were extracted from the convolutional feature encoder (512 dimensions, 20 ms frames) and all twelve transformer layers (768 dimensions), and truncated to the common frame count.

Raw cosine similarity is not comparable across layers. Over 50 random pairs of *different* test-clean utterances, the mean frame-wise cosine between unrelated speech was 0.04 at the convolutional output, 0.30 at layer 10 and **0.96 at layer 11**, so raw drift in layer 11 is near zero whatever the input. Following Timkey and van Schijndel [\[22\]](#ref-timkey2021bark), each dimension $`d`$ of layer $`\ell`$ was therefore standardised with a mean $`\mu_{\ell,d}`$ and standard deviation $`\sigma_{\ell,d}`$, estimated once from all WAV frames of a fixed calibration set (the first 100 selected test-clean utterances). After standardisation, unrelated utterances had a cosine of 0.02–0.08 in every layer. Drift is

``` math
D_\ell=1-\frac{1}{T}\sum_{t=1}^{T}\cos\!\Big(\tfrac{h^{\text{WAV}}_\ell(t)-\mu_\ell}{\sigma_\ell},\;\tfrac{h^{\text{codec}}_\ell(t)-\mu_\ell}{\sigma_\ell}\Big).
```

Raw-cosine drift is retained in the result files for reference only.

### 3.6 Predictor analysis

Four signal measures (LSD, the mean squared log-spectral difference, bandwidth loss $`8\,\text{kHz}-f_{\text{ret}}`$, and 4–8 kHz power loss) were compared with the drift at each of the 13 representation levels. Three analyses are kept separate:

1.  **Across conditions:** Pearson and Spearman correlation between the condition-mean measure and corpus ΔWER, over 11 codec conditions plus WAV (12 points).
2.  **Pooled:** Spearman correlation with utterance-level ΔWER over all 5,500 (utterance, condition) pairs.
3.  **Within condition:** Spearman correlation across the 500 utterances of one condition, averaged over the 11 conditions. Codec and bitrate are fixed, so this asks only which *utterances* degrade.

Confidence intervals use the same speaker-level bootstrap. Because every measure is evaluated on the same resamples, each measure’s correlation minus the LSD correlation also has an interval. The intervals do not reflect the choice of the 11 codec settings.

### 3.7 Case selection and EnCodec extension

Illustrative cases were chosen by a fixed rule from Opus 8 kbps on test-clean, among WAV-correct utterances of 4–10 s. The **failure case** is the utterance with the *median* layer-12 drift among those whose Opus transcript contains an error and whose drift lies in the condition’s top quartile (10 eligible). The **robust case** is the error-free utterance in the bottom drift quartile (43 eligible) whose LSD is closest to that of the failure case. As population context, new-error rates were compared between the top and bottom quartiles of layer-12 drift and of LSD in every condition.

For the EnCodec extension, the 24 kHz EnCodec model [\[3\]](#ref-defossez2023encodec) was applied at 24, 6 and 1.5 kbps to the same 500 test-clean utterances. The audio was resampled 16 → 24 kHz, coded, and resampled back to 16 kHz. A **resampling control** (16 → 24 → 16 kHz without coding) isolates the resampling path. EnCodec output showed zero lag against its input on the utterances checked. LSD and standardised drift were computed as above (layers 1, 6, 12).

## 4. Results

### 4.1 Recognition performance

The WAV baselines were 3.17% (test-clean) and 8.26% (test-other). This is consistent with the 3.4% and 8.5% reported for the full test sets with the same model and no language model [\[7\]](#ref-baevski2020wav2vec). The EnCodec pipeline reproduced the test-clean baseline exactly.

**Table 1.** Signal distortion, layer-12 drift and recognition results. Bitrate is measured, including container overhead. LSD, retained bandwidth (Band) and standardised layer-12 drift (L12) are test-clean means; test-other values differ by at most 1.4 dB, 0.06 kHz and 0.09. ΔWER is in percentage points with 95% speaker-level bootstrap confidence intervals (40 / 33 speakers).

| Condition | Bitrate (kbps) | LSD (dB) | Band (kHz) | L12 drift | Clean WER (%) | Clean ΔWER (pp) \[95% CI\] | Other WER (%) | Other ΔWER (pp) \[95% CI\] |
|----|---:|---:|---:|---:|---:|---:|---:|---:|
| WAV | 256.1 | 0 | 8.0 | 0 | 3.17 | — | 8.26 | — |
| MP3 128k | 130.2 | 3.5 | 7.3 | 0.004 | 3.15 | −0.02 \[−0.16, +0.11\] | 8.50 | +0.24 \[+0.02, +0.47\] |
| MP3 64k | 65.1 | 4.1 | 7.3 | 0.005 | 3.10 | −0.07 \[−0.21, +0.07\] | 8.54 | +0.27 \[+0.06, +0.50\] |
| MP3 32k | 32.6 | 6.3 | 7.3 | 0.021 | 3.27 | +0.11 \[−0.13, +0.35\] | 8.95 | +0.68 \[+0.22, +1.15\] |
| MP3 24k | 24.5 | 9.2 | 5.8 | 0.041 | 3.19 | +0.02 \[−0.26, +0.31\] | 9.69 | +1.42 \[+0.89, +1.99\] |
| MP3 16k | 16.4 | 11.7 | 5.6 | 0.071 | 4.08 | +0.91 \[+0.61, +1.26\] | 12.00 | +3.74 \[+2.79, +4.71\] |
| Opus 64k | 71.5 | 1.7 | 8.0 | 0.003 | 3.19 | +0.02 \[−0.06, +0.11\] | 8.33 | +0.07 \[−0.09, +0.24\] |
| Opus 32k | 31.6 | 4.0 | 8.0 | 0.009 | 3.13 | −0.04 \[−0.20, +0.12\] | 8.36 | +0.10 \[−0.14, +0.32\] |
| Opus 16k | 16.1 | 5.4 | 8.0 | 0.022 | 3.18 | +0.01 \[−0.21, +0.24\] | 8.90 | +0.64 \[+0.30, +0.97\] |
| Opus 12k | 12.2 | 5.9 | 8.0 | 0.035 | 3.49 | +0.32 \[+0.05, +0.60\] | 9.60 | +1.33 \[+0.85, +1.82\] |
| Opus 8k | 8.1 | 11.4 | 4.6 | 0.086 | 4.32 | +1.15 \[+0.82, +1.48\] | 14.89 | +6.63 \[+4.92, +8.60\] |
| Opus 6k | 6.3 | 11.6 | 4.7 | 0.130 | 5.45 | +2.28 \[+1.82, +2.80\] | 19.79 | +11.53 \[+9.07, +14.18\] |

**Test-clean.** No condition down to MP3 24 kbps or Opus 16 kbps changed WER measurably: all seven intervals include zero, with point estimates between −0.07 and +0.11 pp. Degradation became measurable at Opus 12 kbps (+0.32 pp) and reached +2.28 pp at Opus 6 kbps.

**Test-other.** ΔWER was larger than on test-clean in every condition. All MP3 settings, and Opus at 16 kbps and below, had intervals excluding zero. At Opus 6 kbps the degradation was five times the test-clean value.

**Error types.** Substitutions made up 84–91% of the net additional errors for MP3 16 kbps and Opus 8 and 6 kbps on both subsets. For example, test-other Opus 6 kbps added 858 substitutions, 116 deletions and 38 insertions.

### 4.2 Signal distortion without task degradation

<figure>
<img src="results/final_report/fig1_integrated.png" alt="Figure 1. Signal, representation and recognition against measured bitrate (log scale) for MP3 (blue) and Opus (orange); top row test-clean, bottom row test-other. Columns: log-spectral distance (dB); retained bandwidth (kHz), the highest frequency at which the codec keeps power within 20 dB of WAV; standardised drift at layers 1 and 12 (1 − cosine similarity, dimensionless); ΔWER relative to WAV (percentage points; shaded: 95% speaker-level bootstrap CI). Each point is a mean over 500 utterances." />
<figcaption aria-hidden="true"><strong>Figure 1.</strong> Signal, representation and recognition against measured bitrate (log scale) for MP3 (blue) and Opus (orange); top row test-clean, bottom row test-other. Columns: log-spectral distance (dB); retained bandwidth (kHz), the highest frequency at which the codec keeps power within 20 dB of WAV; standardised drift at layers 1 and 12 (1 − cosine similarity, dimensionless); ΔWER relative to WAV (percentage points; shaded: 95% speaker-level bootstrap CI). Each point is a mean over 500 utterances.</figcaption>
</figure>

LSD rose steadily as bitrate fell, but test-clean ΔWER stayed flat until much lower bitrates (Figure 1).

MP3 24 kbps is the clearest case. Its LSD (9.2 dB) exceeded that of every Opus setting down to 12 kbps, and it removed content above 5.8 kHz, yet test-clean ΔWER was +0.02 pp \[−0.26, +0.31\].

Retained bandwidth changed in steps. MP3 kept about 7.3 kHz from 128 to 32 kbps, then 5.8 kHz at 24 kbps and 5.6 kHz at 16 kbps. Opus kept the full 8 kHz down to 12 kbps, then 4.6–4.7 kHz at 8 and 6 kbps, where the bitstream switches to SILK narrowband.

On test-other, the four largest ΔWER values all occurred in the four conditions with a cutoff below 6 kHz. On test-clean, three of the four largest did. The exception is MP3 24 kbps, which had no measurable test-clean effect despite its 5.8 kHz cutoff.

Opus 12 kbps kept the full band and degraded far less than Opus 8 kbps (+0.32 vs +1.15 pp test-clean; +1.33 vs +6.63 pp test-other). Opus 8 and 6 kbps had nearly identical LSD (11.4, 11.6 dB) and cutoff (4.6, 4.7 kHz), yet test-other ΔWER nearly doubled between them.

### 4.3 Representation drift across layers

<figure>
<img src="results/final_report/fig2_drift_by_layer.png" style="width:70.0%" alt="Figure 2. Mean standardised representation drift (1 − cosine similarity of per-dimension standardised hidden states; dimensionless) at the convolutional feature-encoder output (“conv”) and each transformer layer. Columns: MP3, Opus; rows: test-clean, test-other. Darker lines denote lower target bitrates. Means over 500 utterances." />
<figcaption aria-hidden="true"><strong>Figure 2.</strong> Mean standardised representation drift (1 − cosine similarity of per-dimension standardised hidden states; dimensionless) at the convolutional feature-encoder output (“conv”) and each transformer layer. Columns: MP3, Opus; rows: test-clean, test-other. Darker lines denote lower target bitrates. Means over 500 utterances.</figcaption>
</figure>

Standardised drift increased monotonically with compression severity at every layer (Figure 2). It was highest at the convolutional output or in layers 1–3, and declined towards layers 10–12.

From the convolutional output to layer 12, drift fell by a factor of 2.7–4.4 on test-clean (e.g. Opus 16 kbps: 0.095 → 0.022) but only 1.5–2.7 on test-other (e.g. Opus 6 kbps: 0.425 → 0.214).

Raw cosine drift instead showed a 5–64-fold drop between layers 10 and 11. This is the anisotropy artefact described in Section 3.5, and it disappears after standardisation.

### 4.4 Which measure tracks degradation?

<figure>
<img src="results/final_report/fig3_predictors.png" alt="Figure 3. Predictor analysis. (a, b) Across conditions: corpus ΔWER (percentage points) against condition-mean LSD (dB) and standardised layer-12 drift (dimensionless) for the 12 conditions (WAV grey, MP3 blue, Opus orange; circles test-clean, triangles test-other), with Pearson r and 95% speaker-level bootstrap CI. (c) Within a condition: mean Spearman correlation between each layer’s drift and utterance-level ΔWER over the 11 codec conditions (blue; shaded 95% CI) and the same statistic for LSD (green); solid test-clean, dashed test-other." />
<figcaption aria-hidden="true"><strong>Figure 3.</strong> Predictor analysis. (a, b) Across conditions: corpus ΔWER (percentage points) against condition-mean LSD (dB) and standardised layer-12 drift (dimensionless) for the 12 conditions (WAV grey, MP3 blue, Opus orange; circles test-clean, triangles test-other), with Pearson r and 95% speaker-level bootstrap CI. (c) Within a condition: mean Spearman correlation between each layer’s drift and utterance-level ΔWER over the 11 codec conditions (blue; shaded 95% CI) and the same statistic for LSD (green); solid test-clean, dashed test-other.</figcaption>
</figure>

**Across conditions** (Figure 3a–b), layer-12 drift was nearly linear in ΔWER:

| Measure | Pearson r, test-clean | Pearson r, test-other |
|----|----|----|
| Layer-12 drift | 0.96 \[0.91, 0.98\] | 0.97 \[0.96, 0.98\] |
| LSD | 0.75 \[0.66, 0.81\] | 0.79 \[0.76, 0.81\] |
| Paired difference (drift − LSD) | +0.21 \[+0.16, +0.25\] | +0.18 \[+0.17, +0.20\] |

Rank correlations did not separate the two measures. Spearman ρ was 0.77 (LSD) vs 0.78 (drift) on test-clean, and 0.99 vs 0.97 on test-other. The difference is therefore one of proportionality: LSD saturates near 11–12 dB while ΔWER keeps rising, whereas drift continues to separate Opus 8 kbps (0.086 / 0.153 on test-clean / test-other) from Opus 6 kbps (0.130 / 0.214).

**Pooled over utterances**, all measures correlated weakly with utterance-level ΔWER. Layer-12 drift was highest: ρ = 0.21 \[0.17, 0.25\] on test-clean and 0.42 \[0.37, 0.46\] on test-other, against 0.15 and 0.32 for LSD.

**Within a condition** (Figure 3c), signal measures showed no association with which utterances degraded. For LSD, mean ρ = −0.02 \[−0.06, +0.02\] on test-clean and +0.03 \[−0.03, +0.10\] on test-other; bandwidth loss and convolutional-output drift were also near zero.

The association increased broadly with depth, reaching mean ρ = 0.09 \[0.05, 0.13\] (test-clean) and 0.23 \[0.18, 0.27\] (test-other) at layer 12. It was larger in the severe conditions: point estimates of 0.53 and 0.64 for test-other Opus 8 and 6 kbps, and 0.24 and 0.40 on test-clean.

In test-other Opus 8 kbps, 86% of WAV-correct utterances in the top layer-12 drift quartile acquired an error, against 23% in the bottom quartile. The LSD quartiles gave 53% and 45%. On test-clean the drift quartiles gave 30% and 7%, and the LSD quartiles 15% and 25%.

### 4.5 Illustrative cases

<figure>
<img src="results/final_report/fig4_case_study.png" style="width:86.0%" alt="Figure 4. Robust versus failure case at Opus 8 kbps (test-clean), selected by the rule in Section 3.7. Left, centre: spectrograms (dB relative to the WAV peak; frequency in kHz, time in s) of the WAV and decoded signals; both lose content above about 4.5 kHz. Right: the utterance’s standardised drift profile (orange) against the condition mean and interquartile range (grey). Robust case 7729-102255-0025: LSD 10.8 dB, layer-12 drift 0.042, transcript unchanged. Failure case 672-122797-0073: LSD 10.5 dB, layer-12 drift 0.129, “FLAMED UP” recognised as “FLAME OT”." />
<figcaption aria-hidden="true"><strong>Figure 4.</strong> Robust versus failure case at Opus 8 kbps (test-clean), selected by the rule in Section 3.7. Left, centre: spectrograms (dB relative to the WAV peak; frequency in kHz, time in s) of the WAV and decoded signals; both lose content above about 4.5 kHz. Right: the utterance’s standardised drift profile (orange) against the condition mean and interquartile range (grey). Robust case 7729-102255-0025: LSD 10.8 dB, layer-12 drift 0.042, transcript unchanged. Failure case 672-122797-0073: LSD 10.5 dB, layer-12 drift 0.129, “FLAMED UP” recognised as “FLAME OT”.</figcaption>
</figure>

The two selected utterances (Figure 4) have almost the same LSD (10.8 and 10.5 dB), the same narrowband cutoff, and similar convolutional-output drift (0.29 and 0.32). They diverge in the upper half of the network. The robust case falls below the condition mean from layer 4 and ends at 0.042, below the condition’s lower quartile (0.062). The failure case rises above the mean from layer 6 and ends at 0.129, above the upper quartile (0.100).

The error is a substitution at a word boundary (“FLAMED UP” → “FLAME OT”). A single example supports no phonetic generalisation.

### 4.6 EnCodec extension

**Table 2.** EnCodec extension on the same 500 test-clean utterances (WAV WER 3.17%).

| Condition            | LSD (dB) | Layer-1 drift | Layer-12 drift | WER (%) | ΔWER (pp) |
|----------------------|---------:|--------------:|---------------:|--------:|----------:|
| 16→24→16 kHz control |      1.4 |         0.004 |          0.002 |    3.19 |     +0.02 |
| EnCodec 24 kbps      |      7.0 |         0.179 |          0.038 |    3.28 |     +0.12 |
| EnCodec 6 kbps       |      7.5 |         0.265 |          0.062 |    3.77 |     +0.60 |
| EnCodec 1.5 kbps     |      8.5 |         0.458 |          0.198 |    9.95 |     +6.79 |

The resampling path alone had a negligible effect.

EnCodec 24 kbps retained the full 8 kHz band (checked on six utterances) but had a higher LSD than Opus at 16 or 12 kbps (7.0 vs 5.4 and 5.9 dB), with little WER change. Its layer-12 drift (0.038) was close to that of MP3 24 kbps (0.041) and Opus 12 kbps (0.035).

At 1.5 kbps, layer-12 drift (0.198) and ΔWER (+6.79 pp) rose sharply, while LSD rose only from 7.0 to 8.5 dB.

## 5. Discussion

### 5.1 Distortion without task failure

Spectral distortion and recognition degradation are only loosely coupled. MP3 24 kbps and EnCodec 24 kbps change the log spectrum by 7–9 dB, more than Opus 16 kbps, yet neither changes test-clean WER measurably.

This is consistent with how the codecs work. A perceptual coder hides its quantisation noise below the masking threshold and gives up high-frequency bandwidth first [\[1\]](#ref-brandenburg1999mp3). A neural codec reconstructs a plausible signal rather than the waveform [\[3\]](#ref-defossez2023encodec). Much of the measured LSD therefore probably comes from low-energy bins, fine spectral structure and the upper band, which carry little of the information the recogniser uses.

The pattern matches findings from speech enhancement, where the *type* of processing error matters more than its magnitude [\[26\]](#ref-iwamoto2022artifacts). It is also consistent with the robustness of neural ASR at moderate MP3 and Opus bitrates reported by Narayanan et al. [\[5\]](#ref-narayanan2018domain).

The representation results suggest where the absorption happens. Drift at the convolutional output grows with LSD but is reduced 2.7–4.4-fold by layer 12 on test-clean. This fits the account in which the upper layers of a CTC-fine-tuned model encode task categories rather than acoustic detail [\[8\]](#ref-pasad2021layerwise).

LSD is not useless: it ranks conditions almost as well as late-layer drift (Spearman 0.77–0.99). Its *magnitude*, however, is a poor guide to the size of the task effect.

### 5.2 Transition into the speech band

On test-other, the four largest degradations occur where the codec cutoff falls below about 6 kHz; on test-clean, three of the four do. For Opus the mechanism is visible in the bitstream. Between 12 and 8 kbps, libopus switches every packet from SILK wideband to SILK narrowband, which is the operating range RFC 6716 gives for 8–12 kbit/s speech [\[2\]](#ref-rfc6716). The 4–8 kHz band is then no longer transmitted at all. Opus 12 kbps shows that bitrate reduction as such is not the problem: with the full band retained, degradation is small.

MP3 24 kbps, with a 5.8 kHz cutoff, has no measurable test-clean effect but a clear test-other effect (+1.42 pp). This suggests that band-limitation interacts with speech that is already difficult. Narayanan et al.’s loss at MP3 23 kbps [\[5\]](#ref-narayanan2018domain) was measured on different data and a different model, but falls at the same point.

Two cautions apply:

- bandwidth is confounded with coarser in-band quantisation at the same bitrates, so these are associations across a small number of settings (Section 6);
- bandwidth is not sufficient on its own: Opus 8 and 6 kbps share a cutoff and nearly the same LSD, yet ΔWER nearly doubles, so in-band coding precision also matters, and neither global signal measure captures it.

### 5.3 Representation drift and recognition failure

**Across conditions,** layer-12 drift tracks the *size* of ΔWER nearly linearly, while LSD saturates. This extends the benchmark practice of judging codecs by downstream WER [\[6\]](#ref-wu2024codecsuperb), [\[19\]](#ref-shi2024espnetcodec) with an internal measure that scales with the task effect across three codec families. It parallels Chai et al.’s finding that distances inside an acoustic model track WER better than perceptual metrics [\[9\]](#ref-chai2021cegm), and Zhu et al.’s observation that Wav2Vec2 output similarity follows noise level [\[24\]](#ref-zhu2022noiserobust).

**Within a condition,** every signal measure is near zero, including the bandwidth measure that explained the across-condition transition. Only late-layer drift shows an association, and it is modest: 0.09 and 0.23 on average, up to 0.64 in the severe conditions. Most of the variation in *which* utterances fail is therefore unexplained by any measure studied. The case study illustrates the pattern: two utterances with similar global signal distortion and similar early-layer drift show substantially different late-layer drift and recognition outcomes.

These associations are not causal. Layer 12 feeds the CTC output projection [\[10\]](#ref-graves2006ctc). Any change in the transcript requires some change in the final hidden states, so part of the layer-12 correlation is expected by construction. Two results are informative despite this: within-condition association grows steadily with depth from about zero at the convolutional output, and signal measures, which can be computed without the model, have no within-condition association. Together they suggest that whether a codec perturbation matters is decided inside the network, not visible in the signal. Whether this reflects the speech content or the model’s processing cannot be determined from correlations.

The subset difference fits this reading. Test-other shows weaker attenuation (1.5–2.7×), larger late-layer drift for the same codec setting, and five times larger ΔWER at Opus 6 kbps. When speech is already difficult, the recogniser appears to have less margin to absorb further acoustic change, which is consistent with the domain-shift sensitivity of Wav2Vec2 [\[14\]](#ref-hsu2021robust).

### 5.4 Generality and the EnCodec extension

EnCodec keeps the full band but reconstructs rather than transmits spectral detail, so its LSD is high even at 24 kbps. It nonetheless follows the MP3/Opus pattern:

- settings with similar layer-12 drift (EnCodec 24 kbps 0.038, MP3 24 kbps 0.041, Opus 12 kbps 0.035) have similarly small test-clean WER changes;
- the degradation at 1.5 kbps coincides with a large rise in late-layer drift rather than in LSD.

The resampling control indicates that the 24 kHz resampling path contributes negligibly to the observed degradation. This suggests that the drift–WER relationship is not specific to one codec family. The extension does not show that it holds for other recognisers, languages or recording conditions.

### 5.5 Answer to the secondary question

The results are consistent with Wav2Vec2 remaining robust while codec-induced changes are largely attenuated before they reach the upper representations. This holds on clean speech down to about 24 kbps for MP3 and 16 kbps for Opus. Preservation of bandwidth and coarse spectral structure appears to contribute to this robustness, but the present analysis does not isolate the specific acoustic cues responsible. Degradation became pronounced when the codec removed part of the speech band or coded the retained band more coarsely, and in these conditions more of the perturbation remained in the final layers.

## 6. Limitations

- **One recogniser.** Only Wav2Vec2 Base with greedy CTC decoding was tested; models such as Whisper [\[16\]](#ref-radford2023whisper) may absorb distortion differently. *Future work:* repeat the analysis for an encoder–decoder model and for a codec-augmented model [\[5\]](#ref-narayanan2018domain).
- **Late layers are coupled to the output.** The layer-12 correlation with WER is partly structural. *Future work:* intervene, e.g. replace layer-*k* states of the compressed input with WAV states and measure how much WER recovers.
- **Bandwidth is confounded with quantisation.** *Future work:* apply matched low-pass filters to uncompressed audio, and force Opus to wideband at 8 kbps.
- **Read speech from MP3-coded sources.** LibriSpeech is read audiobook speech, and its source audio was already MP3-coded, so all conditions are tandem coding. *Future work:* conversational or noisy corpora recorded without lossy coding.
- **Encoder defaults.** LAME CBR and libopus VBR (`application=audio`) as configured by FFmpeg. Other modes would change bandwidth decisions. *Future work:* compare alternative modes at the transition bitrates.
- **Bitrate is not perceptual quality.** No listening test or perceptual metric was used. *Future work:* add STOI or ViSQOL and compare them with ASR degradation.
- **Modest and coarse evidence.** Within-condition correlations average 0.09–0.23. Condition-level correlations rest on 12 settings per subset, and their intervals ignore setting selection. *Future work:* a denser bitrate grid around the transition, and a multivariate utterance-level model.
- **EnCodec scope.** The extension covers test-clean and one neural codec, and requires 16 → 24 → 16 kHz resampling (controlled).
- **Illustrative cases.** The case study is rule-selected but illustrative. Population claims rest on the correlation and quartile analyses.

## 7. Conclusion

MP3 and Opus change the acoustic signal well before they change the recognition performance of a fixed Wav2Vec2 model.

- Down to MP3 24 kbps and Opus 16 kbps, up to 9.2 dB LSD and a 5.8 kHz cutoff produced no measurable WER change on test-clean.
- The largest degradations were concentrated in conditions where the codec cutoff entered the speech band (MP3 ≤ 24 kbps; Opus narrowband at 8 and 6 kbps), particularly on test-other, where ΔWER reached +11.53 pp. However, bandwidth loss alone was not sufficient to produce measurable degradation on test-clean (MP3 24 kbps: +0.02 pp, CI including zero). In-band quantisation also mattered: Opus 8 and 6 kbps share the same cutoff but differ substantially in ΔWER.
- Inside the model, codec perturbations were largest in the early layers and were reduced 2.7–4.4-fold by the final layer on clean speech, less on harder speech.
- Across conditions, final-layer drift tracked the size of the WER change far more linearly than spectral distortion, although both ranked conditions similarly.
- Within a condition, final-layer drift was the only measure with any association with which utterances failed, and that association was modest.
- EnCodec showed the same pattern.

Wav2Vec2’s robustness to lossy compression is therefore better described as attenuation of codec-induced change before it reaches the upper representations than as insensitivity to signal change. Because the late-layer association is partly structural and entirely correlational, locating where robustness is decided will require interventional follow-up work.

## Acknowledgement of AI Assistance

Generative AI tools were used during project development for brainstorming, writing and debugging analysis code, literature-search assistance, and drafting and editing report text. All experiments were run on the author’s local machine, all numerical results were independently checked against the generated output files, all references were verified against original sources, and the author takes responsibility for the methodology, interpretation and final report.

## References

<div id="refs" class="references csl-bib-body" entry-spacing="0">

<div id="ref-brandenburg1999mp3" class="csl-entry">

<span class="csl-left-margin">\[1\] </span><span class="csl-right-inline">K. Brandenburg, “MP3 and AAC explained,” in *<span class="nocase">Proc. AES 17th International Conference on High-Quality Audio Coding</span>*, Florence, Italy, 1999. Available: <https://www.aes.org/e-lib/browse.cfm?elib=8079></span>

</div>

<div id="ref-rfc6716" class="csl-entry">

<span class="csl-left-margin">\[2\] </span><span class="csl-right-inline">J.-M. Valin, K. Vos, and T. B. Terriberry, “Definition of the Opus audio codec,” IETF, RFC 6716, Sep. 2012. doi: [10.17487/RFC6716](https://doi.org/10.17487/RFC6716).</span>

</div>

<div id="ref-defossez2023encodec" class="csl-entry">

<span class="csl-left-margin">\[3\] </span><span class="csl-right-inline">A. Défossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural audio compression,” *Transactions on Machine Learning Research*, 2023, Available: <https://openreview.net/forum?id=ivCd8z8zR2></span>

</div>

<div id="ref-besacier2001effect" class="csl-entry">

<span class="csl-left-margin">\[4\] </span><span class="csl-right-inline">L. Besacier, C. Bergamini, D. Vaufreydaz, and E. Castelli, “The effect of speech and audio compression on speech recognition performance,” in *<span class="nocase">Proc. IEEE Fourth Workshop on Multimedia Signal Processing (MMSP)</span>*, 2001, pp. 301–306. doi: [10.1109/MMSP.2001.962750](https://doi.org/10.1109/MMSP.2001.962750).</span>

</div>

<div id="ref-narayanan2018domain" class="csl-entry">

<span class="csl-left-margin">\[5\] </span><span class="csl-right-inline">A. Narayanan *et al.*, “Toward domain-invariant speech recognition via large scale training,” in *Proc. IEEE Spoken Language Technology Workshop (SLT)*, 2018, pp. 441–447. doi: [10.1109/SLT.2018.8639610](https://doi.org/10.1109/SLT.2018.8639610).</span>

</div>

<div id="ref-wu2024codecsuperb" class="csl-entry">

<span class="csl-left-margin">\[6\] </span><span class="csl-right-inline">H. Wu *et al.*, “Codec-SUPERB: An in-depth analysis of sound codec models,” in *<span class="nocase">Findings of the Association for Computational Linguistics: ACL 2024</span>*, 2024, pp. 10330–10348. doi: [10.18653/v1/2024.findings-acl.616](https://doi.org/10.18653/v1/2024.findings-acl.616).</span>

</div>

<div id="ref-baevski2020wav2vec" class="csl-entry">

<span class="csl-left-margin">\[7\] </span><span class="csl-right-inline">A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in *<span class="nocase">Advances in Neural Information Processing Systems</span>*, 2020, pp. 12449–12460. Available: <https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html></span>

</div>

<div id="ref-pasad2021layerwise" class="csl-entry">

<span class="csl-left-margin">\[8\] </span><span class="csl-right-inline">A. Pasad, J.-C. Chou, and K. Livescu, “Layer-wise analysis of a self-supervised speech representation model,” in *<span class="nocase">Proc. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)</span>*, 2021, pp. 914–921. doi: [10.1109/ASRU51503.2021.9688093](https://doi.org/10.1109/ASRU51503.2021.9688093).</span>

</div>

<div id="ref-chai2021cegm" class="csl-entry">

<span class="csl-left-margin">\[9\] </span><span class="csl-right-inline">L. Chai, J. Du, Q.-F. Liu, and C.-H. Lee, “A cross-entropy-guided measure (CEGM) for assessing speech recognition performance and optimizing DNN-based speech enhancement,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 29, pp. 106–117, 2021, doi: [10.1109/TASLP.2020.3036783](https://doi.org/10.1109/TASLP.2020.3036783).</span>

</div>

<div id="ref-graves2006ctc" class="csl-entry">

<span class="csl-left-margin">\[10\] </span><span class="csl-right-inline">A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, “Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks,” in *<span class="nocase">Proc. 23rd International Conference on Machine Learning (ICML)</span>*, 2006, pp. 369–376. doi: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891).</span>

</div>

<div id="ref-pasad2023comparative" class="csl-entry">

<span class="csl-left-margin">\[11\] </span><span class="csl-right-inline">A. Pasad, B. Shi, and K. Livescu, “Comparative layer-wise analysis of self-supervised speech models,” in *Proc. IEEE ICASSP*, 2023. doi: [10.1109/ICASSP49357.2023.10096149](https://doi.org/10.1109/ICASSP49357.2023.10096149).</span>

</div>

<div id="ref-hirsch2000aurora" class="csl-entry">

<span class="csl-left-margin">\[12\] </span><span class="csl-right-inline">H.-G. Hirsch and D. Pearce, “The AURORA experimental framework for the performance evaluation of speech recognition systems under noisy conditions,” in *<span class="nocase">Proc. ASR2000 – Automatic Speech Recognition: Challenges for the New Millenium (ISCA ITRW)</span>*, 2000, pp. 181–188. Available: <https://www.isca-archive.org/asr_2000/hirsch00_asr.html></span>

</div>

<div id="ref-drude2021opus" class="csl-entry">

<span class="csl-left-margin">\[13\] </span><span class="csl-right-inline">L. Drude, J. Heymann, A. Schwarz, and J.-M. Valin, “Multi-channel Opus compression for far-field automatic speech recognition with a fixed bitrate budget,” in *Proc. Interspeech*, 2021, pp. 1669–1673. doi: [10.21437/Interspeech.2021-1214](https://doi.org/10.21437/Interspeech.2021-1214).</span>

</div>

<div id="ref-hsu2021robust" class="csl-entry">

<span class="csl-left-margin">\[14\] </span><span class="csl-right-inline">W.-N. Hsu *et al.*, “Robust wav2vec 2.0: Analyzing domain shift in self-supervised pre-training,” in *Proc. Interspeech*, 2021, pp. 721–725. doi: [10.21437/Interspeech.2021-236](https://doi.org/10.21437/Interspeech.2021-236).</span>

</div>

<div id="ref-huang2022distortion" class="csl-entry">

<span class="csl-left-margin">\[15\] </span><span class="csl-right-inline">K. P. Huang, Y.-K. Fu, Y. Zhang, and H. Lee, “Improving distortion robustness of self-supervised speech processing tasks with domain adaptation,” in *Proc. Interspeech*, 2022, pp. 2193–2197. doi: [10.21437/Interspeech.2022-519](https://doi.org/10.21437/Interspeech.2022-519).</span>

</div>

<div id="ref-radford2023whisper" class="csl-entry">

<span class="csl-left-margin">\[16\] </span><span class="csl-right-inline">A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in *<span class="nocase">Proc. 40th International Conference on Machine Learning (ICML)</span>*, in PMLR, vol. 202. 2023, pp. 28492–28518. Available: <https://proceedings.mlr.press/v202/radford23a.html></span>

</div>

<div id="ref-gray1980distortion" class="csl-entry">

<span class="csl-left-margin">\[17\] </span><span class="csl-right-inline">R. M. Gray, A. Buzo, A. H. Gray Jr., and Y. Matsuyama, “Distortion measures for speech processing,” *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 28, no. 4, pp. 367–376, 1980, doi: [10.1109/TASSP.1980.1163421](https://doi.org/10.1109/TASSP.1980.1163421).</span>

</div>

<div id="ref-zeghidour2022soundstream" class="csl-entry">

<span class="csl-left-margin">\[18\] </span><span class="csl-right-inline">N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “SoundStream: An end-to-end neural audio codec,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 30, pp. 495–507, 2022, doi: [10.1109/TASLP.2021.3129994](https://doi.org/10.1109/TASLP.2021.3129994).</span>

</div>

<div id="ref-shi2024espnetcodec" class="csl-entry">

<span class="csl-left-margin">\[19\] </span><span class="csl-right-inline">J. Shi *et al.*, “ESPnet-Codec: Comprehensive training and evaluation of neural codecs for audio, music, and speech,” in *Proc. IEEE Spoken Language Technology Workshop (SLT)*, 2024, pp. 562–569. doi: [10.1109/SLT61566.2024.10832289](https://doi.org/10.1109/SLT61566.2024.10832289).</span>

</div>

<div id="ref-wang2025audiocodecbench" class="csl-entry">

<span class="csl-left-margin">\[20\] </span><span class="csl-right-inline">L. Wang *et al.*, “AudioCodecBench: A comprehensive benchmark for audio codec evaluation.” 2025. Available: <https://arxiv.org/abs/2509.02349></span>

</div>

<div id="ref-ethayarajh2019contextual" class="csl-entry">

<span class="csl-left-margin">\[21\] </span><span class="csl-right-inline">K. Ethayarajh, “How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 55–65. doi: [10.18653/v1/D19-1006](https://doi.org/10.18653/v1/D19-1006).</span>

</div>

<div id="ref-timkey2021bark" class="csl-entry">

<span class="csl-left-margin">\[22\] </span><span class="csl-right-inline">W. Timkey and M. van Schijndel, “All bark and no bite: Rogue dimensions in transformer language models obscure representational quality,” in *<span class="nocase">Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)</span>*, 2021, pp. 4527–4546. doi: [10.18653/v1/2021.emnlp-main.372](https://doi.org/10.18653/v1/2021.emnlp-main.372).</span>

</div>

<div id="ref-kornblith2019similarity" class="csl-entry">

<span class="csl-left-margin">\[23\] </span><span class="csl-right-inline">S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in *<span class="nocase">Proc. 36th International Conference on Machine Learning (ICML)</span>*, in PMLR, vol. 97. 2019, pp. 3519–3529. Available: <https://proceedings.mlr.press/v97/kornblith19a.html></span>

</div>

<div id="ref-zhu2022noiserobust" class="csl-entry">

<span class="csl-left-margin">\[24\] </span><span class="csl-right-inline">Q.-S. Zhu, J. Zhang, Z.-Q. Zhang, M.-H. Wu, X. Fang, and L.-R. Dai, “A noise-robust self-supervised pre-training model based speech representation learning for automatic speech recognition,” in *Proc. IEEE ICASSP*, 2022, pp. 3174–3178. doi: [10.1109/ICASSP43922.2022.9747379](https://doi.org/10.1109/ICASSP43922.2022.9747379).</span>

</div>

<div id="ref-wang2022wav2vecswitch" class="csl-entry">

<span class="csl-left-margin">\[25\] </span><span class="csl-right-inline">Y. Wang, J. Li, H. Wang, Y. Qian, C. Wang, and Y. Wu, “Wav2vec-switch: Contrastive learning from original-noisy speech pairs for robust speech recognition,” in *Proc. IEEE ICASSP*, 2022, pp. 7097–7101. doi: [10.1109/ICASSP43922.2022.9746929](https://doi.org/10.1109/ICASSP43922.2022.9746929).</span>

</div>

<div id="ref-iwamoto2022artifacts" class="csl-entry">

<span class="csl-left-margin">\[26\] </span><span class="csl-right-inline">K. Iwamoto *et al.*, “How bad are artifacts?: Analyzing the impact of speech enhancement errors on ASR,” in *Proc. Interspeech*, 2022, pp. 5418–5422. doi: [10.21437/Interspeech.2022-318](https://doi.org/10.21437/Interspeech.2022-318).</span>

</div>

<div id="ref-efron1993bootstrap" class="csl-entry">

<span class="csl-left-margin">\[27\] </span><span class="csl-right-inline">B. Efron and R. J. Tibshirani, *An introduction to the bootstrap*. New York: Chapman & Hall, 1993. doi: [10.1007/978-1-4899-4541-9](https://doi.org/10.1007/978-1-4899-4541-9).</span>

</div>

<div id="ref-bisani2004bootstrap" class="csl-entry">

<span class="csl-left-margin">\[28\] </span><span class="csl-right-inline">M. Bisani and H. Ney, “Bootstrap estimates for confidence intervals in ASR performance evaluation,” in *Proc. IEEE ICASSP*, 2004, pp. I-409-I-412. doi: [10.1109/ICASSP.2004.1326009](https://doi.org/10.1109/ICASSP.2004.1326009).</span>

</div>

<div id="ref-panayotov2015librispeech" class="csl-entry">

<span class="csl-left-margin">\[29\] </span><span class="csl-right-inline">V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: An ASR corpus based on public domain audio books,” in *Proc. IEEE ICASSP*, 2015, pp. 5206–5210. doi: [10.1109/ICASSP.2015.7178964](https://doi.org/10.1109/ICASSP.2015.7178964).</span>

</div>

</div>
