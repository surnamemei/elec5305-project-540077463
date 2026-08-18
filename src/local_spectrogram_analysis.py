import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


# --------------------------------------------------
# Settings
# --------------------------------------------------
DATA_ROOT = "data"
OUTPUT_DIR = "results/figures/local_case_studies"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160


# --------------------------------------------------
# Cases
#
# start_time / end_time are initial estimates.
# We can adjust them after seeing the plots.
# --------------------------------------------------
CASES: list[dict[str, str | int | float]] = [
    {
        "dataset": "test-other",
        "dataset_index": 78,
        "codec": "mp3",
        "bitrate": "16k",
        "start_time": 2.05,
        "end_time": 2.75,
        "word": "DEAR",
        "label": "case1_dear_mp3_16k",
    },
    {
        "dataset": "test-other",
        "dataset_index": 124,
        "codec": "opus",
        "bitrate": "8k",
        "start_time": 2.00,
        "end_time": 2.90,
        "word": "GRATEFUL",
        "label": "case2_grateful_opus_8k",
    },
]


# --------------------------------------------------
# Convert to mono
# --------------------------------------------------
def to_mono(
    waveform: torch.Tensor
) -> torch.Tensor:

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    return waveform


# --------------------------------------------------
# Crop waveform by time
# --------------------------------------------------
def crop_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> torch.Tensor:

    start_sample = int(
        start_time * sample_rate
    )

    end_sample = int(
        end_time * sample_rate
    )

    start_sample = max(
        0,
        start_sample
    )

    end_sample = min(
        int(waveform.shape[1]),
        end_sample
    )

    return waveform[
        :,
        start_sample:end_sample
    ]


# --------------------------------------------------
# Compress and decode
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
        int(compressed_sr)
    )


# --------------------------------------------------
# Spectrogram in dB
# --------------------------------------------------
def make_spectrogram(
    waveform: torch.Tensor
) -> torch.Tensor:

    waveform = to_mono(
        waveform
    )

    transform = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        power=2.0,
    )

    power_spec = transform(
        waveform
    )

    db_transform = (
        torchaudio.transforms.AmplitudeToDB(
            stype="power"
        )
    )

    db_spec = db_transform(
        power_spec
    )

    return db_spec[0]


# --------------------------------------------------
# Match spectrogram length
# --------------------------------------------------
def trim_spectrograms(
    original_spec: torch.Tensor,
    compressed_spec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    min_frames = min(
        int(original_spec.shape[1]),
        int(compressed_spec.shape[1])
    )

    return (
        original_spec[:, :min_frames],
        compressed_spec[:, :min_frames],
    )


# --------------------------------------------------
# Plot local comparison
# --------------------------------------------------
def plot_local_comparison(
    original_spec: torch.Tensor,
    compressed_spec: torch.Tensor,
    sample_rate: int,
    codec: str,
    bitrate: str,
    word: str,
    start_time: float,
    end_time: float,
    output_path: str,
) -> None:

    (
        original_spec,
        compressed_spec
    ) = trim_spectrograms(
        original_spec,
        compressed_spec
    )

    difference = (
        compressed_spec
        - original_spec
    )

    original_np = (
        original_spec
        .detach()
        .cpu()
        .numpy()
    )

    compressed_np = (
        compressed_spec
        .detach()
        .cpu()
        .numpy()
    )

    difference_np = (
        difference
        .detach()
        .cpu()
        .numpy()
    )

    shared_min = float(
        min(
            original_np.min(),
            compressed_np.min()
        )
    )

    shared_max = float(
        max(
            original_np.max(),
            compressed_np.max()
        )
    )

    diff_abs = float(
        np.max(
            np.abs(
                difference_np
            )
        )
    )

    if diff_abs == 0.0:
        diff_abs = 1.0

    duration = (
        original_np.shape[1]
        * HOP_LENGTH
        / float(sample_rate)
    )

    extent: tuple[
        float,
        float,
        float,
        float
    ] = (
        0.0,
        float(duration),
        0.0,
        float(sample_rate) / 2.0,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True
    )

    image1 = axes[0].imshow(
        original_np,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=shared_min,
        vmax=shared_max,
    )

    axes[0].set_title(
        f'Original WAV — target word "{word}"'
    )

    axes[0].set_ylabel(
        "Frequency (Hz)"
    )

    fig.colorbar(
        image1,
        ax=axes[0],
        label="Power (dB)"
    )

    image2 = axes[1].imshow(
        compressed_np,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=shared_min,
        vmax=shared_max,
    )

    axes[1].set_title(
        f"{codec.upper()} {bitrate}"
    )

    axes[1].set_ylabel(
        "Frequency (Hz)"
    )

    fig.colorbar(
        image2,
        ax=axes[1],
        label="Power (dB)"
    )

    image3 = axes[2].imshow(
        difference_np,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-diff_abs,
        vmax=diff_abs,
        cmap="coolwarm",
    )

    axes[2].set_title(
        "Difference: Compressed - WAV"
    )

    axes[2].set_xlabel(
        "Local Time (s)"
    )

    axes[2].set_ylabel(
        "Frequency (Hz)"
    )

    fig.colorbar(
        image3,
        ax=axes[2],
        label="Difference (dB)"
    )

    fig.suptitle(
        (
            f'Local Spectrogram Analysis: "{word}" '
            f"({start_time:.2f}s–{end_time:.2f}s)"
        ),
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:

    for case in CASES:

        dataset_name = str(
            case["dataset"]
        )

        dataset_index = int(
            case["dataset_index"]
        )

        codec = str(
            case["codec"]
        )

        bitrate = str(
            case["bitrate"]
        )

        start_time = float(
            case["start_time"]
        )

        end_time = float(
            case["end_time"]
        )

        word = str(
            case["word"]
        )

        label = str(
            case["label"]
        )

        print(
            f"\nProcessing "
            f"{dataset_name} "
            f"index={dataset_index} "
            f"{codec.upper()} {bitrate} "
            f'word="{word}"'
        )

        dataset = (
            torchaudio.datasets.LIBRISPEECH(
                root=DATA_ROOT,
                url=dataset_name,
                download=False
            )
        )

        (
            waveform,
            sample_rate,
            reference,
            _speaker_id,
            _chapter_id,
            _utterance_id,
        ) = dataset[
            dataset_index
        ]

        sample_rate = int(
            sample_rate
        )

        compressed_waveform, compressed_sr = (
            compress_audio(
                waveform,
                sample_rate,
                codec,
                bitrate
            )
        )

        if compressed_sr != sample_rate:

            compressed_waveform = (
                torchaudio.functional.resample(
                    compressed_waveform,
                    compressed_sr,
                    sample_rate
                )
            )

        # Crop AFTER compression so both correspond
        # to the same absolute time interval.
        original_local = crop_waveform(
            waveform,
            sample_rate,
            start_time,
            end_time
        )

        compressed_local = crop_waveform(
            compressed_waveform,
            sample_rate,
            start_time,
            end_time
        )

        original_spec = make_spectrogram(
            original_local
        )

        compressed_spec = make_spectrogram(
            compressed_local
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{label}_local_comparison.png"
        )

        plot_local_comparison(
            original_spec,
            compressed_spec,
            sample_rate,
            codec,
            bitrate,
            word,
            start_time,
            end_time,
            output_path
        )

        print(
            "Reference:",
            reference
        )

        print(
            "Saved:",
            output_path
        )

    print(
        "\nLocal spectrogram analysis complete."
    )


if __name__ == "__main__":
    main()