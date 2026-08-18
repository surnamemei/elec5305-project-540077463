import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import torch
import torchaudio


# --------------------------------------------------
# Settings
# --------------------------------------------------
DATA_ROOT = "data"

OUTPUT_DIR = "results/figures/case_studies"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CASES = [
    {
        "dataset": "test-other",
        "dataset_index": 78,
        "codec": "mp3",
        "bitrate": "16k",
        "label": "test-other_mp3_16k_case1",
    },
    {
        "dataset": "test-other",
        "dataset_index": 124,
        "codec": "opus",
        "bitrate": "8k",
        "label": "test-other_opus_8k_case2",
    },
    {
        "dataset": "test-clean",
        "dataset_index": 386,
        "codec": "mp3",
        "bitrate": "16k",
        "label": "test-clean_mp3_16k_case3",
    },
]


# --------------------------------------------------
# Compress and decode audio
# --------------------------------------------------
def compress_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    codec: str,
    bitrate: str,
) -> tuple[torch.Tensor, int]:

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(
            temp_dir,
            "input.wav"
        )

        torchaudio.save(
            input_path,
            waveform,
            sample_rate
        )

        if codec == "mp3":

            output_path = os.path.join(
                temp_dir,
                "compressed.mp3"
            )

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                output_path,
            ]

        elif codec == "opus":

            output_path = os.path.join(
                temp_dir,
                "compressed.opus"
            )

            command = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                input_path,
                "-codec:a",
                "libopus",
                "-b:a",
                bitrate,
                output_path,
            ]

        else:
            raise ValueError(
                f"Unsupported codec: {codec}"
            )

        subprocess.run(
            command,
            check=True
        )

        compressed_waveform, compressed_sr = (
            torchaudio.load(
                output_path
            )
        )

    return (
        compressed_waveform,
        compressed_sr
    )


# --------------------------------------------------
# Convert waveform to spectrogram
# --------------------------------------------------
def make_spectrogram(
    waveform: torch.Tensor,
) -> torch.Tensor:

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    transform = torchaudio.transforms.Spectrogram(
        n_fft=512,
        win_length=400,
        hop_length=160,
        power=2.0,
    )

    spectrogram = transform(
        waveform
    )

    spectrogram_db = (
        torchaudio.transforms.AmplitudeToDB(
            stype="power"
        )(spectrogram)
    )

    return spectrogram_db[0]


# --------------------------------------------------
# Plot one spectrogram
# --------------------------------------------------
def plot_spectrogram(
    spectrogram: torch.Tensor,
    sample_rate: int,
    title: str,
    output_path: str,
) -> None:

    spec = spectrogram.numpy()

    duration_frames = spec.shape[1]

    hop_length = 160
    duration = (
        duration_frames
        * hop_length
        / sample_rate
    )

    max_frequency = (
        sample_rate / 2
    )

    plt.figure(
        figsize=(10, 4)
    )

    extent: tuple[float, float, float, float] = (
        0.0,
        float(duration),
        0.0,
        float(max_frequency),
    )

    plt.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=extent,
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Frequency (Hz)"
    )

    plt.title(
        title
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# Run all case studies
# --------------------------------------------------
for case in CASES:

    dataset_name = case["dataset"]
    dataset_index = case["dataset_index"]
    codec = case["codec"]
    bitrate = case["bitrate"]
    label = case["label"]

    print(
        f"\nProcessing "
        f"{dataset_name} "
        f"index={dataset_index} "
        f"{codec.upper()} {bitrate}"
    )

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=DATA_ROOT,
        url=dataset_name,
        download=False
    )

    (
        waveform,
        sample_rate,
        reference,
        speaker_id,
        chapter_id,
        utterance_id,
    ) = dataset[
        dataset_index
    ]

    compressed_waveform, compressed_sr = (
        compress_audio(
            waveform,
            sample_rate,
            codec,
            bitrate
        )
    )

    original_spec = make_spectrogram(
        waveform
    )

    compressed_spec = make_spectrogram(
        compressed_waveform
    )

    original_path = os.path.join(
        OUTPUT_DIR,
        f"{label}_wav.png"
    )

    compressed_path = os.path.join(
        OUTPUT_DIR,
        f"{label}_{codec}_{bitrate}.png"
    )

    plot_spectrogram(
        original_spec,
        sample_rate,
        f"{label} - WAV",
        original_path
    )

    plot_spectrogram(
        compressed_spec,
        compressed_sr,
        f"{label} - {codec.upper()} {bitrate}",
        compressed_path
    )

    print(
        "Reference:",
        reference
    )

    print(
        "Saved:",
        original_path
    )

    print(
        "Saved:",
        compressed_path
    )


print(
    "\nSpectrogram case studies complete."
)