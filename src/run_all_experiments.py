import csv
import os
import subprocess
import tempfile
import random
import torch
import torchaudio
from jiwer import wer
from tqdm import tqdm

# --------------------------------------------------
# Settings
# --------------------------------------------------
NUM_SAMPLES = 500
RANDOM_SEED = 5305
DATA_ROOT = "data"

DETAIL_RESULT_PATH = "results/all_experiment_details.csv"
SUMMARY_RESULT_PATH = "results/summary_results.csv"

CONDITIONS = [
    {"codec": "wav",  "bitrate": "uncompressed"},

    {"codec": "mp3",  "bitrate": "128k"},
    {"codec": "mp3",  "bitrate": "64k"},
    {"codec": "mp3",  "bitrate": "32k"},
    {"codec": "mp3",  "bitrate": "24k"},
    {"codec": "mp3",  "bitrate": "16k"},

    {"codec": "opus", "bitrate": "64k"},
    {"codec": "opus", "bitrate": "32k"},
    {"codec": "opus", "bitrate": "16k"},
    {"codec": "opus", "bitrate": "12k"},
    {"codec": "opus", "bitrate": "8k"},
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# Load Wav2Vec2
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
# ASR
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
# Compression
# --------------------------------------------------
def process_audio(waveform, sample_rate, codec, bitrate):
    """
    Returns:
        processed_waveform
        processed_sample_rate
        original_size
        compressed_size
        compression_ratio
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        wav_path = os.path.join(temp_dir, "input.wav")

        torchaudio.save(
            wav_path,
            waveform,
            sample_rate
        )

        original_size = os.path.getsize(wav_path)

        # ------------------------------
        # WAV baseline
        # ------------------------------
        if codec == "wav":
            processed_waveform = waveform
            processed_sample_rate = sample_rate
            compressed_size = original_size
            compression_ratio = 1.0

            return (
                processed_waveform,
                processed_sample_rate,
                original_size,
                compressed_size,
                compression_ratio,
            )

        # ------------------------------
        # MP3
        # ------------------------------
        if codec == "mp3":
            output_path = os.path.join(temp_dir, "compressed.mp3")

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
                bitrate,
                output_path,
            ]

        # ------------------------------
        # Opus
        # ------------------------------
        elif codec == "opus":
            output_path = os.path.join(temp_dir, "compressed.opus")

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                wav_path,
                "-codec:a",
                "libopus",
                "-b:a",
                bitrate,
                output_path,
            ]

        else:
            raise ValueError(f"Unsupported codec: {codec}")

        subprocess.run(
            command,
            check=True
        )

        compressed_size = os.path.getsize(output_path)

        processed_waveform, processed_sample_rate = torchaudio.load(
            output_path
        )

        compression_ratio = original_size / compressed_size

        return (
            processed_waveform,
            processed_sample_rate,
            original_size,
            compressed_size,
            compression_ratio,
        )


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("\nLoading LibriSpeech test-clean...")

dataset = torchaudio.datasets.LIBRISPEECH(
    root=DATA_ROOT,
    url="test-clean",
    download=False
)

print("Dataset loaded.")
print("Total samples:", len(dataset))
random.seed(RANDOM_SEED)

sample_indices = random.sample(
    range(len(dataset)),
    k=min(NUM_SAMPLES, len(dataset))
)

print("Selected samples:", len(sample_indices))

# --------------------------------------------------
# Run experiments
# --------------------------------------------------
os.makedirs("results", exist_ok=True)

detail_results = []
summary_results = []

for condition in CONDITIONS:

    codec = condition["codec"]
    bitrate = condition["bitrate"]

    print("\n" + "#" * 75)
    print(f"Running condition: {codec.upper()} {bitrate}")
    print("#" * 75)

    references = []
    predictions = []

    compression_ratios = []

    for sample_number, index in enumerate(
        tqdm(
            sample_indices,
            desc=f"{codec.upper()} {bitrate}",
            unit="sample"
        ),
        start=1
    ):

        (
            waveform,
            sample_rate,
            reference,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = dataset[index]

        (
            processed_waveform,
            processed_sample_rate,
            original_size,
            compressed_size,
            compression_ratio,
        ) = process_audio(
            waveform,
            sample_rate,
            codec,
            bitrate
        )

        prediction = recognise(
            processed_waveform,
            processed_sample_rate
        )

        sample_wer = wer(
            reference,
            prediction
        )

        references.append(reference)
        predictions.append(prediction)
        compression_ratios.append(compression_ratio)

        detail_results.append({
            "codec": codec,
            "bitrate": bitrate,
            "sample": sample_number,
            "dataset_index": index,
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "utterance_id": utterance_id,
            "reference": reference,
            "prediction": prediction,
            "wer": sample_wer,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": compression_ratio,
        })

        # print(
        #     f"{codec.upper():5s} "
        #     f"{bitrate:12s} "
        #     f"Sample {sample_number:3d} | "
        #     f"WER={sample_wer:.4f} | "
        #     f"Compression={compression_ratio:.2f}x"
        # )

    overall_wer = wer(
        references,
        predictions
    )

    average_compression_ratio = (
        sum(compression_ratios) / len(compression_ratios)
    )

    summary_results.append({
        "codec": codec,
        "bitrate": bitrate,
        "num_samples": len(references),
        "overall_wer": overall_wer,
        "average_compression_ratio": average_compression_ratio,
    })

    print("\nCondition result:")
    print(f"Overall WER: {overall_wer:.4f}")
    print(
        f"Average compression ratio: "
        f"{average_compression_ratio:.2f}x"
    )

# --------------------------------------------------
# Calculate WER difference from WAV baseline
# --------------------------------------------------
wav_wer = next(
    row["overall_wer"]
    for row in summary_results
    if row["codec"] == "wav"
)

for row in summary_results:
    row["delta_wer"] = row["overall_wer"] - wav_wer


# --------------------------------------------------
# Save detailed CSV
# --------------------------------------------------
with open(
    DETAIL_RESULT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "codec",
            "bitrate",
            "sample",
            "dataset_index",
            "speaker_id",
            "chapter_id",
            "utterance_id",
            "reference",
            "prediction",
            "wer",
            "original_size_bytes",
            "compressed_size_bytes",
            "compression_ratio",
        ]
    )

    writer.writeheader()
    writer.writerows(detail_results)


# --------------------------------------------------
# Save summary CSV
# --------------------------------------------------
with open(
    SUMMARY_RESULT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "codec",
            "bitrate",
            "num_samples",
            "overall_wer",
            "delta_wer",
            "average_compression_ratio",
        ]
    )

    writer.writeheader()
    writer.writerows(summary_results)


# --------------------------------------------------
# Final summary
# --------------------------------------------------
print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

for row in summary_results:
    print(
        f"{row['codec'].upper():5s} "
        f"{row['bitrate']:12s} | "
        f"WER={row['overall_wer']:.4f} | "
        f"ΔWER={row['delta_wer']:+.4f} | "
        f"Compression={row['average_compression_ratio']:.2f}x"
    )

print("\nSaved:")
print(DETAIL_RESULT_PATH)
print(SUMMARY_RESULT_PATH)