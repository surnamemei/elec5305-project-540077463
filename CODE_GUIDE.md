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
WER + compression ratio CSV files
    ↓
analyse_results.py
bootstrap_analysis.py
error_analysis.py
spectrogram_analysis.py
local_spectrogram_analysis.py
```

## `run_all_experiments.py`

**Purpose:** Main experiment script.

This script:
1. Loads a fixed set of LibriSpeech utterances.
2. Uses uncompressed WAV as the baseline.
3. Encodes the same utterance using MP3 or Opus at different bitrates.
4. Decodes the compressed audio back to a waveform.
5. Sends every waveform through the same pretrained Wav2Vec2 ASR model.
6. Calculates per-utterance and corpus-level Word Error Rate (WER).
7. Measures the file-size compression ratio.
8. Saves detailed and summary CSV files.

Important settings:
- `DATASET_NAME`: choose `test-clean` or `test-other`.
- `NUM_SAMPLES = 500`
- `RANDOM_SEED = 5305`
- `CONDITIONS`: defines the codec/bitrate combinations.

The fixed random seed ensures that every codec condition is evaluated using the same utterances.

## `analyse_results.py`

**Purpose:** Turn summary CSV data into figures.

This script:
1. Reads the summary results for `test-clean` and `test-other`.
2. Extracts the WAV baseline for each dataset.
3. Calculates ΔWER relative to the corresponding WAV baseline.
4. Converts WER values into percentages.
5. Creates figures showing:
   - WER vs bitrate
   - ΔWER vs bitrate
   - WER vs compression ratio
   - `test-clean` vs `test-other`

This script does **not** rerun the ASR model. It only analyses saved results.

## `bootstrap_analysis.py`

**Purpose:** Estimate uncertainty in the measured WER degradation.

This script performs a **paired bootstrap**:
1. WAV and compressed predictions are paired using `dataset_index`.
2. The same utterance indices are resampled with replacement.
3. WER is recalculated for WAV and the compressed condition.
4. Their difference is stored.
5. Repeating this process 2000 times gives a distribution of ΔWER.
6. The 2.5th and 97.5th percentiles form the approximate 95% confidence interval.

Pairing is important because the WAV and compressed conditions contain the same source utterances.

## `error_analysis.py`

**Purpose:** Explain what kind of ASR errors increase after compression.

The script uses word-level edit distance to count:
- Substitutions (S)
- Deletions (D)
- Insertions (I)

For each compression condition it compares the compressed prediction with the WAV baseline and identifies:
- `new_error`: WAV was correct but compressed audio is wrong.
- `recovered`: WAV was wrong but compressed audio becomes correct.
- `worsened`: utterance-level WER increases.
- `improved`: utterance-level WER decreases.
- `unchanged`: utterance-level WER does not change.

The script also calculates:
- Δ substitutions
- Δ deletions
- Δ insertions

This helps explain *how* recognition performance changes, rather than only reporting overall WER.

## `spectrogram_analysis.py`

**Purpose:** Compare signal-level changes caused by aggressive compression.

For selected utterances the script:
1. Loads the original LibriSpeech waveform.
2. Compresses the full utterance using the same FFmpeg settings as the main experiment.
3. Decodes the compressed audio.
4. Calculates spectrograms for WAV and compressed audio.
5. Creates a difference spectrogram.
6. Saves individual and combined comparison figures.

The difference spectrogram is used as supporting evidence of codec-induced spectral changes. It does not by itself prove that a particular spectral change caused a specific ASR error.

## `local_spectrogram_analysis.py`

**Purpose:** Zoom into short regions around representative recognition errors.

The script:
1. Compresses the entire utterance first.
2. Crops the same absolute time range from WAV and compressed audio.
3. Generates local spectrograms.
4. Creates a local difference plot.

Compressing before cropping keeps the analysis consistent with the main experiment.

The current time windows are manually selected case-study regions rather than forced-alignment word timestamps.

## Pilot Scripts

### `baseline_asr.py`

Early development script used to confirm:
- LibriSpeech loading
- Wav2Vec2 inference
- CTC decoding
- baseline WER calculation

It is retained as project development history.

### `experiment_mp3.py`

Early pilot used to test:
- FFmpeg MP3 encoding
- MP3 decoding
- compression ratio calculation
- compressed-audio ASR

The final multi-codec experiment is now implemented in `run_all_experiments.py`.

## Recommended Execution Order

```bash
python src/run_all_experiments.py
python src/analyse_results.py
python src/bootstrap_analysis.py
python src/error_analysis.py
python src/spectrogram_analysis.py
python src/local_spectrogram_analysis.py
```

`baseline_asr.py` and `experiment_mp3.py` are optional pilot scripts and are not required for the final analysis.

## Main Experimental Principle

Only the audio compression condition changes between experiments.

The following remain fixed:
- source utterance
- ASR model
- decoding method
- evaluation metric
- random sample selection

This controlled design allows changes in WER to be compared against the uncompressed WAV baseline.
