# Project Rationale and Development Process

## 1. From Topic Selection to a Focused Research Question

At the beginning of the ELEC5305 project, the main goal was to choose a topic that was clearly connected to speech and audio processing, technically meaningful, experimentally feasible, and capable of producing measurable results within the semester.

AI assistance was used at this stage as a discussion and planning tool. Several possible directions were considered, but a recurring concern was that many speech-processing topics could easily become too broad. For example, combining noise, reverberation, multiple codecs, multiple ASR systems, different datasets, and perceptual evaluation would make it difficult to identify which variable was actually responsible for any observed change.

The project was therefore narrowed to a controlled question:

> **How robust is a modern automatic speech recognition system to lossy audio compression, and at what bitrate or compression ratio does recognition performance begin to degrade significantly?**

A secondary question was then added:

> **Does lossy compression have a greater impact when the underlying speech is already more difficult for the ASR system to recognise?**

This gave the project a clear experimental structure. The main independent variables would be:

- audio codec,
- bitrate,
- and speech difficulty.

The ASR system itself would remain fixed.

The topic was selected because it links several areas relevant to ELEC5305:

- speech and audio processing,
- lossy compression,
- signal representation,
- automatic speech recognition,
- quantitative evaluation,
- and signal-level interpretation.

The central idea became:

```text
lossy compression
    ↓
changes to the speech signal
    ↓
possible changes in ASR recognition
    ↓
measurable WER degradation
```

---

## 2. Why the Scope Was Limited to MP3 and Opus

After the topic was selected, the next decision was which codecs to study.

AI-assisted discussion was used to compare possible choices and to keep the project manageable.

Two codecs were selected:

### MP3

MP3 provides a conventional and widely recognised lossy audio reference. It is useful for examining how ASR behaves as bitrate is gradually reduced from moderate to very aggressive compression.

### Opus

Opus is a modern codec designed for speech and general audio communication and remains practical at much lower bitrates than MP3.

Using both codecs allows the project to investigate:

1. how ASR changes as bitrate decreases within the same codec;
2. whether two different codecs produce different ASR robustness at similar compression levels.

The project deliberately stopped at two codecs. Adding AAC or other codecs was considered possible future work, but would increase the number of experimental variables without necessarily improving the clarity of the main research question.

---

## 3. Choosing the Bitrate Range

The bitrate settings were selected to cover a wide range from moderate compression to deliberately aggressive compression.

### MP3

- 128 kbps
- 64 kbps
- 32 kbps
- 24 kbps
- 16 kbps

### Opus

- 64 kbps
- 32 kbps
- 16 kbps
- 12 kbps
- 8 kbps

An uncompressed WAV condition is used as the baseline.

The aim was not to assume in advance where ASR would begin to fail. Instead, the range was chosen so that the experiment would include:

- conditions expected to remain close to the WAV baseline;
- intermediate conditions where degradation might begin;
- severe low-bitrate conditions where a clear failure region could be observed.

This makes it possible to identify a practical robustness boundary rather than only comparing one compressed setting against WAV.

---

## 4. Why LibriSpeech Was Selected

The dataset needed to provide reliable reference transcripts, enough speech samples for repeated experiments, and a meaningful distinction between easier and harder recognition conditions.

LibriSpeech was selected because it provides:

- standard ASR evaluation data;
- accurate reference transcriptions;
- public availability;
- and multiple evaluation subsets.

Two subsets were used:

### `test-clean`

This represents relatively clean and easier speech and provides a strong baseline for testing whether moderate compression has little practical effect.

### `test-other`

This is more challenging for the ASR system and produces a higher baseline error rate.

The use of both subsets directly supports the secondary research question:

> Is speech that is already more difficult for ASR more vulnerable to compression?

This was a more controlled way to test speech difficulty than introducing additional noise or reverberation, which would create another independent variable.

---

## 5. Why a Fixed Wav2Vec2 Model Was Used

The project is designed to evaluate the effect of compression, not to train or improve an ASR model.

For that reason, the pretrained TorchAudio:

```text
WAV2VEC2_ASR_BASE_960H
```

model was selected and kept fixed throughout the experiment.

The following are held constant:

- ASR architecture;
- pretrained model weights;
- decoding method;
- expected sample rate;
- evaluation metric.

Only the audio compression condition changes.

This is an important experimental control. If the ASR model changed between conditions, it would be difficult to determine whether a WER difference came from the codec or from the recogniser.

The ASR system therefore acts as a fixed measurement instrument.

---

# Development and Testing Process

## 6. Stage One — Build the WAV Baseline First

AI guidance was used to break the project into small, testable stages rather than attempting the full experiment immediately.

The first goal was simply to verify:

```text
LibriSpeech
    ↓
Wav2Vec2
    ↓
Predicted transcript
    ↓
WER
```

This stage was necessary because compression should not be introduced until the basic ASR pipeline is known to work.

The initial script:

```text
src/baseline_asr.py
```

was used to verify:

- LibriSpeech loading;
- waveform shape;
- sample-rate handling;
- model loading;
- GPU/CPU execution;
- Wav2Vec2 inference;
- greedy CTC decoding;
- transcript generation;
- Word Error Rate calculation.

I ran this code locally and inspected the predictions and WER output.

This established the first important checkpoint:

> **The uncompressed speech → ASR → WER pipeline worked independently of any codec.**

---

## 7. Stage Two — Test a Single MP3 Condition

Once the baseline worked, the next step was to introduce compression in the smallest possible way.

Rather than immediately implementing all codecs and bitrates, AI guidance suggested testing one MP3 condition first.

The pilot pipeline was:

```text
LibriSpeech WAV
    ↓
save temporary WAV
    ↓
FFmpeg MP3 encoding
    ↓
decode MP3 back to waveform
    ↓
Wav2Vec2
    ↓
WER
```

This became:

```text
src/experiment_mp3.py
```

The pilot was used to verify several new parts separately:

- FFmpeg was installed correctly;
- `libmp3lame` encoding worked;
- the compressed audio could be decoded;
- TorchAudio could load the decoded waveform;
- Wav2Vec2 still accepted the signal;
- compressed and original file sizes could be measured;
- compression ratio could be calculated.

This was an important development decision.

Instead of debugging ASR, FFmpeg, file I/O, codec handling, and result analysis simultaneously, the project verified one new layer at a time.

---

## 8. Stage Three — Generalise to the Full Experiment

After the MP3 pilot worked, the next step was to generalise the pipeline.

AI assistance was used to help restructure the experiment into reusable stages:

```text
load utterance
    ↓
choose condition
    ↓
WAV baseline / MP3 / Opus
    ↓
decode waveform
    ↓
same Wav2Vec2 model
    ↓
prediction
    ↓
WER + file size
```

The full experiment became:

```text
src/run_all_experiments.py
```

The script was expanded to support:

```text
WAV baseline

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
```

AI assistance during this stage mainly involved:

- suggesting code structure;
- reducing duplicated code;
- defining condition tables;
- helping resolve Python typing and Pylance issues;
- helping diagnose TorchAudio / TorchCodec / FFmpeg issues;
- and organising output files.

I ran the code locally, checked terminal output, and inspected the generated CSV files after the experiment completed.

---

## 9. Stage Four — Make the Sample Selection Reproducible

Running every LibriSpeech utterance at every codec condition would significantly increase runtime.

A fixed sample of:

```text
500 utterances per subset
```

was selected.

The experiment uses:

```python
RANDOM_SEED = 5305
```

AI guidance was useful here in identifying that reproducibility alone was not the only reason for fixing the random seed.

The more important experimental reason is pairing:

> Every codec and bitrate condition must be evaluated using exactly the same source utterances.

This provides fair comparisons and later makes paired statistical analysis possible.

The main experiment therefore holds constant:

- source utterance;
- ASR model;
- decoding;
- metric;
- sample selection.

Only compression changes.

---

## 10. Stage Five — Run `test-clean`

The first full experiment was run on:

```text
LibriSpeech test-clean
```

This produced a complete set of results for WAV, MP3, and Opus.

The first main result pattern was that many moderate bitrate conditions remained very close to the WAV baseline, while aggressive low-bitrate conditions produced larger degradation.

A particularly important interpretation issue appeared at this stage:

some compressed conditions had WER values slightly lower than WAV.

AI-assisted discussion helped avoid an incorrect conclusion.

These small negative ΔWER values were **not** interpreted as evidence that compression improves ASR.

Instead, the working interpretation became:

> Small changes around the baseline can move in either direction because individual recognition decisions can change. Strong conclusions should rely on the size and consistency of degradation, not on tiny point-estimate differences.

This observation directly motivated the next stage.

---

## 11. Stage Six — Add `test-other`

After obtaining the `test-clean` results, the project returned to the secondary research question:

> Does compression affect already difficult speech more strongly?

Rather than introducing artificial noise or reverberation, the same experiment was repeated on:

```text
LibriSpeech test-other
```

The methodology remained unchanged.

This is important because the comparison now isolates speech difficulty while keeping the experimental pipeline consistent.

The resulting pattern was much clearer:

- `test-other` had a higher WAV baseline WER;
- aggressive MP3 and Opus compression produced substantially larger WER increases;
- degradation appeared earlier and more strongly than for `test-clean`.

This became one of the most useful findings of the project.

---

# Analysis Development

## 12. Stage Seven — Plot WER and Compression Efficiency

Once both datasets had been tested, the raw CSV results needed to be converted into interpretable figures.

This led to:

```text
src/analyse_results.py
```

The analysis visualises:

- WER vs bitrate;
- ΔWER vs bitrate;
- WER vs compression ratio;
- comparison between `test-clean` and `test-other`.

AI guidance helped refine the interpretation around two complementary questions:

### Recognition performance

How much does WER change?

### Compression benefit

How much file-size reduction is achieved?

This is why compression ratio is reported in addition to bitrate.

For each utterance:

\[
\text{Compression Ratio}
=
\frac{\text{Original WAV Size}}
{\text{Compressed File Size}}
\]

This allows the project to discuss a practical trade-off:

> How much storage reduction can be achieved before ASR performance begins to degrade noticeably?

---

## 13. Why ΔWER Was Added

Absolute WER is useful, but the baseline difficulty differs greatly between `test-clean` and `test-other`.

The project therefore uses:

\[
\Delta WER
=
WER_{\text{compressed}}
-
WER_{\text{WAV}}
\]

This directly measures the additional recognition error associated with compression.

It makes comparisons easier because:

- `test-clean` is evaluated relative to its own WAV baseline;
- `test-other` is evaluated relative to its own WAV baseline.

The project therefore uses both:

- absolute WER;
- baseline-relative ΔWER.

---

## 14. Stage Eight — Add Paired Bootstrap Analysis

The main results showed that some conditions had very small ΔWER values.

Point estimates alone could not show how stable those differences were.

AI guidance therefore suggested adding uncertainty analysis using a **paired bootstrap**.

This became:

```text
src/bootstrap_analysis.py
```

The procedure is:

1. match WAV and compressed predictions using the same `dataset_index`;
2. sample utterances with replacement;
3. use the same sampled utterances for both WAV and compressed conditions;
4. recalculate corpus WER;
5. calculate ΔWER;
6. repeat 2000 times;
7. obtain an approximate 95% confidence interval.

The analysis is paired because each compressed utterance is derived from the exact same speech sample as its WAV baseline.

This allowed the project to distinguish between:

### Near-baseline variation

The confidence interval includes zero.

### Consistent degradation

The confidence interval is above zero.

AI-assisted interpretation also helped separate:

> **statistical detectability**

from:

> **practical effect size**

A very small ΔWER can be detectable without necessarily being practically important.

---

## 15. Stage Nine — Analyse What Type of ASR Errors Increased

WER answers:

> How much did recognition performance change?

It does not explain:

> What kind of recognition failure became more common?

This motivated:

```text
src/error_analysis.py
```

The analysis uses word-level edit distance to count:

- substitutions;
- deletions;
- insertions.

It also compares each compressed utterance against its WAV counterpart and classifies cases as:

- new error;
- recovered;
- worsened;
- improved;
- unchanged.

AI assistance was used to reason through the dynamic-programming edit-distance implementation and to determine which comparisons would be useful.

The severe low-bitrate conditions showed that most of the additional recognition errors were substitutions.

This gave the project a stronger interpretation:

> Aggressive compression tends to cause the recogniser to confuse one word with another more often than simply deleting or inserting words.

---

# Returning to Signal Processing

## 16. Stage Ten — Add Spectrogram Case Studies

At this stage, the project already had a strong ASR evaluation chain:

```text
codec / bitrate
    ↓
WER
    ↓
ΔWER
    ↓
bootstrap CI
    ↓
error type
```

However, this still did not directly show how the speech signal itself changed.

Because ELEC5305 focuses on speech and audio processing, AI guidance suggested adding a signal-level component.

Representative cases were selected where:

- WAV recognition was correct;
- the compressed version introduced a new recognition error.

This led to:

```text
src/spectrogram_analysis.py
```

For each selected case, the project compares:

- WAV spectrogram;
- compressed spectrogram;
- compressed-minus-WAV spectral difference.

The severe low-bitrate examples show substantial modification or attenuation of spectral energy, especially at higher frequencies.

The interpretation is deliberately cautious:

> The observed signal changes are consistent with the measured ASR degradation and provide supporting evidence, but the spectrogram alone does not prove that a specific spectral difference directly caused a specific word error.

This wording was maintained throughout the analysis to avoid over-claiming causality.

---

## 17. Stage Eleven — Add Local Spectrogram Analysis

A sentence-level spectrogram can make a short recognition error difficult to inspect visually.

For selected substitution errors, a shorter local region was analysed.

This became:

```text
src/local_spectrogram_analysis.py
```

A key methodological decision was:

> Compress the full sentence first, then crop the same absolute time interval from WAV and compressed audio.

This preserves the actual codec behaviour.

Cropping first and then encoding a short segment could cause the codec to behave differently from the main experiment.

The selected local windows were manually estimated around representative errors, so they are treated as approximate case-study regions rather than exact forced-alignment word boundaries.

---

## 18. Stage Twelve — Compare Codecs and Speech Difficulty Directly

After the main WER, bootstrap, error-type and spectrogram analyses were complete, one remaining issue was that two important comparisons were present in the data but were not yet summarised directly.

The first was the comparison between MP3 and Opus at the same nominal bitrate.

Both codecs were tested at:

```text
64 kbps
32 kbps
16 kbps
```

---

# How AI Was Used

## 19. AI as an Iterative Development Assistant

AI was used extensively during this project, but the project did not appear as one complete generated solution.

The workflow was iterative.

A typical cycle was:

```text
identify the next problem
    ↓
discuss possible approach with AI
    ↓
implement or revise code
    ↓
run locally
    ↓
inspect errors / outputs / figures
    ↓
discuss interpretation
    ↓
decide the next experiment
```

This is how the project expanded from a simple baseline into the current analysis pipeline.

AI assistance included:

- brainstorming and comparing project topics;
- narrowing the scope;
- refining the research question;
- explaining technical concepts;
- suggesting experimental controls;
- drafting or restructuring code;
- helping diagnose environment and library errors;
- helping resolve Pylance/type issues;
- suggesting statistical analysis;
- helping interpret WER changes cautiously;
- suggesting error-type analysis;
- suggesting useful spectrogram comparisons;
- helping organise GitHub and documentation.

---

## 20. My Role in the AI-Assisted Workflow

My role was not limited to accepting generated code.

I was responsible for:

- selecting the final project topic;
- deciding which proposed directions to keep;
- configuring the local environment;
- installing and testing dependencies;
- downloading and managing LibriSpeech;
- running all experiments locally;
- checking whether code actually executed correctly;
- inspecting terminal output;
- checking CSV files;
- comparing results between conditions;
- inspecting generated plots;
- selecting useful case studies;
- deciding whether an interpretation was justified;
- managing Git and GitHub;
- and taking responsibility for the final project conclusions.

The project therefore developed through repeated interaction between:

> **AI-assisted technical guidance**

and:

> **local execution, checking, judgement, and experimental decisions**

---

## 21. Development Timeline in One View

The project can be summarised as:

```text
Explore project ideas
    ↓
Select lossy compression + ASR
    ↓
Refine research question
    ↓
Limit scope to MP3 + Opus
    ↓
Choose LibriSpeech
    ↓
Choose fixed Wav2Vec2 model
    ↓
Build WAV baseline
    ↓
Test one MP3 pilot
    ↓
Generalise full compression pipeline
    ↓
Fix 500-sample reproducible selection
    ↓
Run test-clean
    ↓
Interpret first WER pattern
    ↓
Run test-other
    ↓
Compare easy vs difficult speech
    ↓
Plot WER / ΔWER / compression ratio
    ↓
Add paired bootstrap confidence intervals
    ↓
Add substitution / deletion / insertion analysis
    ↓
Identify representative recognition failures
    ↓
Add sentence-level spectrogram analysis
    ↓
Add local spectrogram case studies
    ↓
Compare MP3 vs Opus at matched bitrates
    ↓
Quantify test-other vs test-clean degradation gap
    ↓
Organise README, results, GitHub Pages and code documentation
```

The key point is that each new stage was added because the previous stage raised a new question.

For example:

```text
WER changed
    ↓
but is the change stable?
    ↓
bootstrap analysis
```

then:

```text
WER became worse
    ↓
but what type of errors increased?
    ↓
error-type analysis
```

then:

```text
more word substitutions appeared
    ↓
what happened to the signal?
    ↓
spectrogram analysis
```

The project therefore grew logically rather than by simply adding unrelated features.

---

# Current Interpretation

## 22. Main Pattern Observed So Far

The current results support three main conclusions.

### 1. Moderate compression is relatively robust

For `test-clean`, many moderate MP3 and Opus settings remain close to the WAV baseline.

### 2. Severe low-bitrate compression causes clear degradation

At the lowest bitrates, WER increases consistently.

### 3. Difficult speech is more vulnerable

The same aggressive compression produces substantially larger degradation on `test-other`.

The current analysis also suggests that:

- additional errors are dominated by substitutions;
- Opus can maintain strong ASR performance at relatively high compression ratios;
- but extremely aggressive Opus compression still causes major degradation.

### 4. Codec choice matters at low bitrate

The matched-bitrate results suggest that MP3 and Opus do not produce the same ASR degradation at equivalent nominal bitrates.

The difference is especially clear under difficult speech conditions. For example, at 16 kbps on `test-other`, MP3 produces substantially greater ΔWER than Opus.

This does not establish that Opus is universally superior for ASR, because the experiment uses one ASR model, one corpus and a specific set of codec configurations.

Instead, the result is treated as evidence that codec design is an important part of the compression-robustness relationship.

---

## 23. Why the Current Scope Is Considered Sufficient

Possible future extensions include:

- AAC;
- additional ASR models;
- larger datasets;
- background noise;
- reverberation;
- perceptual quality scores;
- forced alignment;
- other objective signal metrics.

However, adding these would introduce additional variables.

The current project already contains a complete experimental chain:

```text
controlled compression
    ↓
ASR performance
    ↓
compression efficiency
    ↓
statistical confidence
    ↓
error mechanism
    ↓
signal-level evidence
```

For this reason, the current priority is not to continuously expand the experiment.

The remaining work is mainly to:

- refine the explanation;
- incorporate professor feedback;
- finalise figures;
- document limitations;
- and write the final report.

---

# Discussion Points for the Professor

## 24. Questions I Would Like Feedback On

1. **Is MP3 + Opus a sufficient codec scope for the final project?**

2. **Is using 500 fixed utterances from each LibriSpeech subset a reasonable balance between computational cost and reliability?**

3. **Is the paired bootstrap analysis sufficient for quantifying uncertainty in ΔWER?**

4. **Is the `test-clean` vs `test-other` comparison a suitable way to support the secondary question about difficult speech?**

5. **Is the current spectrogram analysis sufficient as the signal-processing component, or would one additional objective signal metric improve the project?**

6. **For the final report, should the main emphasis be placed on the compression-efficiency/ASR trade-off, or on explaining the observed recognition failures?**

---

# Short Explanation for Discussion

## 25. One-Minute Project Summary

> I wanted to investigate whether lossy compression removes information that matters to an ASR model even when the speech may still remain understandable to a human listener. With AI assistance, I first narrowed the project so that codec and bitrate were the main variables while the ASR model remained fixed. I then built the project in stages rather than attempting everything at once. I first verified a WAV-to-Wav2Vec2 baseline, then tested one MP3 condition, then generalised the code to MP3 and Opus across several bitrates. I used a fixed set of 500 LibriSpeech utterances for reproducibility and first tested `test-clean`. After seeing that moderate compression produced only small changes, I added `test-other` to test whether already difficult speech was more vulnerable. I then added paired bootstrap confidence intervals to distinguish stable degradation from small fluctuations, followed by substitution/deletion/insertion analysis to understand how recognition failed. Finally, I added spectrogram case studies to connect the ASR results back to signal-level changes. The current pattern is that moderate compression is relatively robust, while very low bitrates cause clear degradation, especially for the more difficult `test-other` speech.
