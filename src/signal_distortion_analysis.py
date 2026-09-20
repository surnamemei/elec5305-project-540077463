import os
import subprocess
import tempfile
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchaudio

from tqdm import tqdm


# ==================================================
# Settings
# ==================================================

DATA_ROOT = "data"
DATASET_NAME = "test-clean"

NUM_SAMPLES = 100
RANDOM_SEED = 5305

# STFT settings
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

EPS = 1e-8

# Effective bandwidth definition:
# frequency containing 95% of spectral energy
BANDWIDTH_ENERGY_RATIO = 0.95


# ==================================================
# Output paths
# ==================================================

RESULTS_DIR = "results"

DETAIL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "signal_distortion_results.csv"
)

SUMMARY_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "signal_distortion_summary.csv"
)

FREQ_OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "frequency_distortion"
)

FIGURE_OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FREQ_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# Compression conditions
# ==================================================

CONDITIONS = [

    # MP3
    {
        "codec": "mp3",
        "bitrate": "128k"
    },
    {
        "codec": "mp3",
        "bitrate": "64k"
    },
    {
        "codec": "mp3",
        "bitrate": "32k"
    },
    {
        "codec": "mp3",
        "bitrate": "24k"
    },
    {
        "codec": "mp3",
        "bitrate": "16k"
    },

    # Opus
    {
        "codec": "opus",
        "bitrate": "64k"
    },
    {
        "codec": "opus",
        "bitrate": "32k"
    },
    {
        "codec": "opus",
        "bitrate": "16k"
    },
    {
        "codec": "opus",
        "bitrate": "12k"
    },
    {
        "codec": "opus",
        "bitrate": "8k"
    },
    {
        "codec": "opus",
        "bitrate": "6k"
    },
]


# ==================================================
# Audio compression
# ==================================================

def compress_audio(
    waveform,
    sample_rate,
    codec,
    bitrate
):
    """
    Encode the source waveform using FFmpeg and
    decode it back to a waveform.

    Returns:
        compressed_waveform
        compressed_sample_rate
    """

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

        # ------------------------------------------
        # MP3
        # ------------------------------------------

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

        # ------------------------------------------
        # Opus
        # ------------------------------------------

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


# ==================================================
# Basic waveform alignment
# ==================================================

def align_waveforms(
    reference,
    processed
):
    """
    Temporary first-pass alignment.

    Both signals are trimmed to the same length.

    IMPORTANT:
    This does not yet compensate for codec delay
    or padding. Delay-aware alignment will be added
    later when addressing feedback point 23.
    """

    min_length = min(
        reference.shape[-1],
        processed.shape[-1]
    )

    reference = reference[
        ...,
        :min_length
    ]

    processed = processed[
        ...,
        :min_length
    ]

    return (
        reference,
        processed
    )


# ==================================================
# Log-magnitude spectrogram
# ==================================================

def log_spectrogram(
    waveform
):
    """
    Calculate STFT log-magnitude representation.
    """

    # Convert to mono if necessary
    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    window = torch.hann_window(
        WIN_LENGTH
    )

    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    magnitude = torch.abs(
        stft
    )

    log_mag = (
        20.0
        * torch.log10(
            magnitude + EPS
        )
    )

    return log_mag


# ==================================================
# Spectral distortion
# Feedback points 15 and 16
# ==================================================

def spectral_distortion(
    reference,
    processed,
    sample_rate
):
    """
    Returns:

    Dspec:
        overall mean squared log-spectral distortion

    frequencies:
        STFT frequency bins

    D(f):
        mean absolute log-spectral difference
        at each frequency bin
    """

    ref_log = log_spectrogram(
        reference
    )

    proc_log = log_spectrogram(
        processed
    )

    # Make frame counts equal
    min_frames = min(
        ref_log.shape[-1],
        proc_log.shape[-1]
    )

    ref_log = ref_log[
        :,
        :min_frames
    ]

    proc_log = proc_log[
        :,
        :min_frames
    ]

    difference = (
        proc_log
        -
        ref_log
    )

    # ----------------------------------------------
    # Feedback point 15:
    # overall spectral distortion
    # ----------------------------------------------

    dspec = torch.mean(
        difference ** 2
    )

    # ----------------------------------------------
    # Feedback point 16:
    # distortion as a function of frequency
    # ----------------------------------------------

    dfrequency = torch.mean(
        torch.abs(
            difference
        ),
        dim=1
    )

    frequencies = torch.linspace(
        0,
        sample_rate / 2,
        ref_log.shape[0]
    )

    return (
        float(
            dspec.item()
        ),
        frequencies.cpu().numpy(),
        dfrequency.cpu().numpy()
    )


# ==================================================
# Effective bandwidth
# Feedback point 17
# ==================================================

def effective_bandwidth(
    waveform,
    sample_rate,
    energy_ratio=BANDWIDTH_ENERGY_RATIO
):
    """
    Effective bandwidth is defined as the frequency
    below which the specified fraction of total
    spectral energy is contained.

    Default:
        95% spectral energy
    """

    if waveform.shape[0] > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    window = torch.hann_window(
        WIN_LENGTH
    )

    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True
    )

    power = (
        torch.abs(stft)
        ** 2
    )

    # Average energy across time
    mean_power = torch.mean(
        power,
        dim=1
    )

    cumulative_energy = torch.cumsum(
        mean_power,
        dim=0
    )

    total_energy = (
        cumulative_energy[-1]
    )

    if float(total_energy.item()) <= 0:

        return 0.0

    normalized_energy = (
        cumulative_energy
        /
        total_energy
    )

    indices = torch.where(
        normalized_energy
        >=
        energy_ratio
    )[0]

    if len(indices) == 0:

        index = (
            len(mean_power)
            - 1
        )

    else:

        index = int(
            indices[0].item()
        )

    frequencies = torch.linspace(
        0,
        sample_rate / 2,
        len(mean_power)
    )

    bandwidth_hz = frequencies[
        index
    ]

    return float(
        bandwidth_hz.item()
    )


# ==================================================
# Load LibriSpeech
# ==================================================

print(
    "\nLoading LibriSpeech..."
)

dataset = (
    torchaudio.datasets.LIBRISPEECH(
        root=DATA_ROOT,
        url=DATASET_NAME,
        download=False
    )
)

print(
    f"Dataset: {DATASET_NAME}"
)

print(
    f"Total utterances: {len(dataset)}"
)


# ==================================================
# Select fixed samples
# ==================================================

random.seed(
    RANDOM_SEED
)

sample_indices = random.sample(
    range(
        len(dataset)
    ),
    k=min(
        NUM_SAMPLES,
        len(dataset)
    )
)

print(
    f"Selected utterances: {len(sample_indices)}"
)


# ==================================================
# Result containers
# ==================================================

results = []

frequency_results = {}


# ==================================================
# Run distortion analysis
# ==================================================

for condition in CONDITIONS:

    codec = condition[
        "codec"
    ]

    bitrate = condition[
        "bitrate"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"Running {codec.upper()} {bitrate}"
    )

    print(
        "=" * 70
    )

    for index in tqdm(
        sample_indices,
        desc=(
            f"{codec.upper()} "
            f"{bitrate}"
        ),
        unit="sample"
    ):

        (
            waveform,
            sample_rate,
            reference_text,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = dataset[
            index
        ]

        # ------------------------------------------
        # Encode + decode
        # ------------------------------------------

        (
            compressed_waveform,
            compressed_sr
        ) = compress_audio(
            waveform,
            sample_rate,
            codec,
            bitrate
        )

        # ------------------------------------------
        # Match sample rates if required
        # ------------------------------------------

        if (
            compressed_sr
            !=
            sample_rate
        ):

            compressed_waveform = (
                torchaudio.functional.resample(
                    compressed_waveform,
                    compressed_sr,
                    sample_rate
                )
            )

        # ------------------------------------------
        # First-pass alignment
        # ------------------------------------------

        (
            aligned_ref,
            aligned_comp
        ) = align_waveforms(
            waveform,
            compressed_waveform
        )

        # ------------------------------------------
        # Dspec + D(f)
        # ------------------------------------------

        (
            dspec,
            frequencies,
            dfrequency
        ) = spectral_distortion(
            aligned_ref,
            aligned_comp,
            sample_rate
        )

        # ------------------------------------------
        # Effective bandwidth
        # ------------------------------------------

        reference_bandwidth = (
            effective_bandwidth(
                aligned_ref,
                sample_rate
            )
        )

        compressed_bandwidth = (
            effective_bandwidth(
                aligned_comp,
                sample_rate
            )
        )

        bandwidth_change = (
            compressed_bandwidth
            -
            reference_bandwidth
        )

        # ------------------------------------------
        # Save per-utterance results
        # ------------------------------------------

        results.append({

            "dataset":
                DATASET_NAME,

            "dataset_index":
                index,

            "speaker_id":
                speaker_id,

            "chapter_id":
                chapter_id,

            "utterance_id":
                utterance_id,

            "codec":
                codec,

            "bitrate":
                bitrate,

            "spectral_distortion":
                dspec,

            "reference_bandwidth_hz":
                reference_bandwidth,

            "compressed_bandwidth_hz":
                compressed_bandwidth,

            "bandwidth_change_hz":
                bandwidth_change,
        })

        # ------------------------------------------
        # Save D(f)
        # ------------------------------------------

        condition_key = (
            codec,
            bitrate
        )

        if (
            condition_key
            not in frequency_results
        ):

            frequency_results[
                condition_key
            ] = {

                "frequencies":
                    frequencies,

                "distortions":
                    [],
            }

        frequency_results[
            condition_key
        ][
            "distortions"
        ].append(
            dfrequency
        )


# ==================================================
# Save detailed results
# ==================================================

df = pd.DataFrame(
    results
)

df.to_csv(
    DETAIL_OUTPUT_PATH,
    index=False
)


# ==================================================
# Create summary
# ==================================================

summary = (

    df.groupby(
        [
            "codec",
            "bitrate"
        ]
    )

    .agg(

        spectral_distortion_mean=(
            "spectral_distortion",
            "mean"
        ),

        spectral_distortion_median=(
            "spectral_distortion",
            "median"
        ),

        spectral_distortion_std=(
            "spectral_distortion",
            "std"
        ),

        reference_bandwidth_mean_hz=(
            "reference_bandwidth_hz",
            "mean"
        ),

        compressed_bandwidth_mean_hz=(
            "compressed_bandwidth_hz",
            "mean"
        ),

        bandwidth_change_mean_hz=(
            "bandwidth_change_hz",
            "mean"
        ),
    )

    .reset_index()
)

summary.to_csv(
    SUMMARY_OUTPUT_PATH,
    index=False
)


# ==================================================
# Save frequency-dependent distortion curves
# ==================================================

for (
    codec,
    bitrate
), data in frequency_results.items():

    distortions = np.stack(
        data[
            "distortions"
        ],
        axis=0
    )

    mean_distortion = np.mean(
        distortions,
        axis=0
    )

    median_distortion = np.median(
        distortions,
        axis=0
    )

    frequency_df = (
        pd.DataFrame({

            "frequency_hz":
                data[
                    "frequencies"
                ],

            "mean_distortion_db":
                mean_distortion,

            "median_distortion_db":
                median_distortion,
        })
    )

    frequency_output_path = (
        os.path.join(
            FREQ_OUTPUT_DIR,
            (
                f"{codec}_"
                f"{bitrate}_"
                "frequency_distortion.csv"
            )
        )
    )

    frequency_df.to_csv(
        frequency_output_path,
        index=False
    )


# ==================================================
# Plot selected D(f) curves
# ==================================================

plt.figure(
    figsize=(
        10,
        6
    )
)

selected_conditions = [

    (
        "mp3",
        "64k"
    ),

    (
        "mp3",
        "16k"
    ),

    (
        "opus",
        "16k"
    ),

    (
        "opus",
        "12k"
    ),

    (
        "opus",
        "8k"
    ),

    (
        "opus",
        "6k"
    ),
]


for condition in selected_conditions:

    if (
        condition
        not in frequency_results
    ):
        continue

    data = frequency_results[
        condition
    ]

    distortions = np.stack(
        data[
            "distortions"
        ],
        axis=0
    )

    mean_distortion = np.mean(
        distortions,
        axis=0
    )

    codec, bitrate = (
        condition
    )

    plt.plot(
        data[
            "frequencies"
        ],
        mean_distortion,
        label=(
            f"{codec.upper()} "
            f"{bitrate}"
        )
    )


plt.xlabel(
    "Frequency (Hz)"
)

plt.ylabel(
    "Mean Absolute Log-Spectral Distortion (dB)"
)

plt.title(
    "Frequency-Dependent Codec Distortion"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

figure_path = os.path.join(
    FIGURE_OUTPUT_DIR,
    "frequency_distortion_comparison.png"
)

plt.savefig(
    figure_path,
    dpi=300
)

plt.close()


# ==================================================
# Print summary
# ==================================================

print(
    "\n"
    + "=" * 90
)

print(
    "SIGNAL DISTORTION SUMMARY"
)

print(
    "=" * 90
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    DETAIL_OUTPUT_PATH
)

print(
    SUMMARY_OUTPUT_PATH
)

print(
    FREQ_OUTPUT_DIR
)

print(
    figure_path
)