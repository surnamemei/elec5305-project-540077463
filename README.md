# ELEC5305 Project

## Evaluating the Impact of Lossy Audio Compression on Automatic Speech Recognition Performance

This project investigates the robustness of automatic speech recognition (ASR) to lossy audio compression.

Speech from the LibriSpeech dataset is compressed using MP3 and Opus at different bitrates and evaluated using a pretrained Wav2Vec2 ASR model.

The project focuses on the trade-off between compression efficiency and recognition performance.

## Research Question

How robust is automatic speech recognition to lossy audio compression at different bitrates, and at what compression level does recognition performance begin to degrade significantly?

## Current Experimental Conditions

### Baseline
- WAV — uncompressed

### MP3
- 128 kbps
- 64 kbps
- 32 kbps

### Opus
- 64 kbps
- 32 kbps
- 16 kbps

## Evaluation Metrics

- Word Error Rate (WER)
- Compression ratio
- File size / bitrate
- Spectrogram and signal-level analysis

## Current Status

A working experimental pipeline has been implemented using:

- Python
- PyTorch
- TorchAudio
- Wav2Vec2
- FFmpeg
- JiWER

Preliminary evaluation has been completed using 100 randomly selected LibriSpeech test-clean utterances.

## Repository Structure

```text
src/
├── baseline_asr.py
├── experiment_mp3.py
└── run_all_experiments.py