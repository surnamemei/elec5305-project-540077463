# ELEC5305 Project

## Evaluating the Impact of Lossy Audio Compression on Automatic Speech Recognition Performance

### Student
**SID:** 540077463  
**GitHub:** surnamemei

## Project Overview

This project investigates how lossy audio compression affects automatic speech recognition performance.

Speech from the LibriSpeech dataset is compressed using MP3 and Opus at different bitrates. Each compressed signal is decoded and processed using the same pretrained Wav2Vec2 automatic speech recognition model.

Recognition performance is evaluated using Word Error Rate (WER), while compression ratio is used to measure storage efficiency.

The objective is to identify how much speech audio can be compressed before recognition performance begins to degrade significantly.

## Method

```text
LibriSpeech Speech
       ↓
Original WAV
       ↓
MP3 / Opus Compression
       ↓
Different Bitrates
       ↓
Wav2Vec2 ASR
       ↓
Predicted Transcript
       ↓
Word Error Rate