import csv
import os
import subprocess
import tempfile

import torch
import torchaudio
from jiwer import wer


# --------------------------------------------------
# Settings
# --------------------------------------------------
NUM_SAMPLES = 10
BITRATE = "32k"

DATA_ROOT = "data"
RESULT_PATH = "results/mp3_32k_results.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("MP3 bitrate:", BITRATE)


# --------------------------------------------------
# Load pretrained Wav2Vec2
# --------------------------------------------------
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

target_sample_rate = int(bundle.sample_rate)
labels = bundle.get_labels()

model = bundle.get_model().to(device)
model.eval()

print("ASR model loaded.")


# --------------------------------------------------
# CTC decoder
# --------------------------------------------------
def decode(emissions):
    token_ids = torch.argmax(emissions, dim=-1)[0]

    blank_id = 0
    collapsed_ids = []
    previous_id = None

    for token_id in token_ids.tolist():
        if token_id != previous_id:
            if token_id != blank_id:
                collapsed_ids.append(token_id)

        previous_id = token_id

    transcript = "".join(labels[i] for i in collapsed_ids)

    return transcript.replace("|", " ").strip()


# --------------------------------------------------
# ASR function
# --------------------------------------------------
def recognise(waveform, sample_rate):
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sample_rate
        )

    waveform = waveform.to(device)

    with torch.inference_mode():
        emissions, _ = model(waveform)

    return decode(emissions)


# --------------------------------------------------
# MP3 compression
# --------------------------------------------------
def compress_to_mp3(waveform, sample_rate):
    """
    Save waveform to temporary WAV,
    compress it to MP3 using FFmpeg,
    then load the decoded MP3 back.
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        wav_path = os.path.join(temp_dir, "input.wav")
        mp3_path = os.path.join(temp_dir, "compressed.mp3")

        # Save original waveform as temporary WAV
        torchaudio.save(
            wav_path,
            waveform,
            sample_rate
        )

        # Compress WAV -> MP3
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            wav_path,
            "-codec:a",
            "libmp3lame",
            "-b:a",
            BITRATE,
            mp3_path,
        ]

        subprocess.run(
            command,
            check=True
        )

        # File size for compression analysis
        mp3_size = os.path.getsize(mp3_path)
        wav_size = os.path.getsize(wav_path)

        # Decode compressed MP3
        compressed_waveform, compressed_sample_rate = torchaudio.load(
            mp3_path
        )

        return (
            compressed_waveform,
            compressed_sample_rate,
            wav_size,
            mp3_size,
        )


# --------------------------------------------------
# Load LibriSpeech
# --------------------------------------------------
print("\nLoading LibriSpeech test-clean...")

dataset = torchaudio.datasets.LIBRISPEECH(
    root=DATA_ROOT,
    url="test-clean",
    download=False
)

print("Dataset loaded.")
print("Total samples:", len(dataset))


# --------------------------------------------------
# Evaluate
# --------------------------------------------------
os.makedirs("results", exist_ok=True)

results = []

all_references = []
all_predictions = []

for index in range(min(NUM_SAMPLES, len(dataset))):

    (
        waveform,
        sample_rate,
        reference,
        speaker_id,
        chapter_id,
        utterance_id,
    ) = dataset[index]

    (
        compressed_waveform,
        compressed_sample_rate,
        wav_size,
        mp3_size,
    ) = compress_to_mp3(
        waveform,
        sample_rate
    )

    prediction = recognise(
        compressed_waveform,
        compressed_sample_rate
    )

    sample_wer = wer(
        reference,
        prediction
    )

    compression_ratio = wav_size / mp3_size

    all_references.append(reference)
    all_predictions.append(prediction)

    results.append({
        "sample": index + 1,
        "speaker_id": speaker_id,
        "chapter_id": chapter_id,
        "utterance_id": utterance_id,
        "bitrate": BITRATE,
        "reference": reference,
        "prediction": prediction,
        "wer": sample_wer,
        "wav_size_bytes": wav_size,
        "mp3_size_bytes": mp3_size,
        "compression_ratio": compression_ratio,
    })

    print("\n" + "=" * 70)
    print(f"Sample {index + 1}")

    print("Reference:")
    print(reference)

    print("\nPrediction:")
    print(prediction)

    print(f"\nWER: {sample_wer:.4f}")
    print(f"Compression ratio: {compression_ratio:.2f}x")


# --------------------------------------------------
# Overall WER
# --------------------------------------------------
overall_wer = wer(
    all_references,
    all_predictions
)

print("\n" + "=" * 70)
print(f"Overall WER ({NUM_SAMPLES} samples): {overall_wer:.4f}")


# --------------------------------------------------
# Save CSV
# --------------------------------------------------
with open(
    RESULT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "sample",
            "speaker_id",
            "chapter_id",
            "utterance_id",
            "bitrate",
            "reference",
            "prediction",
            "wer",
            "wav_size_bytes",
            "mp3_size_bytes",
            "compression_ratio",
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print(f"\nResults saved to: {RESULT_PATH}")