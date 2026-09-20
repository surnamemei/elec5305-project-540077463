import os
import random
import subprocess
import tempfile

from typing import cast

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchaudio

from tqdm import tqdm
from torchaudio.models import Wav2Vec2Model


# ==================================================
# Settings
# ==================================================

DATA_ROOT = "data"
DATASET_NAME = "test-clean"

NUM_SAMPLES = 100
RANDOM_SEED = 5305

RESULTS_DIR = "results"
FIGURE_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

DETAIL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "representation_similarity_results.csv"
)

SUMMARY_OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "representation_similarity_summary.csv"
)

FIGURE_OUTPUT_PATH = os.path.join(
    FIGURE_DIR,
    "representation_drift_by_layer.png"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ==================================================
# Representative compression conditions
# ==================================================

CONDITIONS = [
    {
        "codec": "mp3",
        "bitrate": "128k"
    },
    {
        "codec": "mp3",
        "bitrate": "16k"
    },
    {
        "codec": "opus",
        "bitrate": "16k"
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
# Selected Wav2Vec2 layers
# ==================================================

SELECTED_LAYERS = {
    "layer_1": 0,
    "layer_6": 5,
    "layer_12": 11,
}


# ==================================================
# Load fixed Wav2Vec2 model
# ==================================================

bundle = (
    torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
)

model = cast(
    Wav2Vec2Model,
    bundle.get_model()
)

model = model.to(
    device
)

model.eval()

TARGET_SAMPLE_RATE = int(
    bundle.sample_rate
)

print(
    "Wav2Vec2 model loaded."
)

print(
    "Target sample rate:",
    TARGET_SAMPLE_RATE
)


# ==================================================
# Load LibriSpeech
# ==================================================

dataset = torchaudio.datasets.LIBRISPEECH(
    root=DATA_ROOT,
    url=DATASET_NAME,
    download=False
)

random.seed(
    RANDOM_SEED
)

sample_indices = random.sample(
    range(len(dataset)),
    k=min(
        NUM_SAMPLES,
        len(dataset)
    )
)

print(
    "Dataset:",
    DATASET_NAME
)

print(
    "Selected utterances:",
    len(sample_indices)
)


# ==================================================
# Prepare waveform
# ==================================================

def prepare_waveform(
    input_waveform: torch.Tensor,
    input_sample_rate: int
) -> torch.Tensor:

    waveform_local = input_waveform

    if (
        input_sample_rate
        !=
        TARGET_SAMPLE_RATE
    ):

        waveform_local = (
            torchaudio.functional.resample(
                waveform_local,
                input_sample_rate,
                TARGET_SAMPLE_RATE
            )
        )

    return waveform_local


# ==================================================
# Compression helper
# ==================================================

def compress_audio(
    input_waveform: torch.Tensor,
    input_sample_rate: int,
    codec: str,
    bitrate: str
) -> tuple[torch.Tensor, int]:

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(
            temp_dir,
            "input.wav"
        )

        torchaudio.save(
            input_path,
            input_waveform,
            input_sample_rate
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


# ==================================================
# Extract selected hidden states
# ==================================================

def extract_hidden_states(
    input_waveform: torch.Tensor,
    input_sample_rate: int
) -> dict[str, torch.Tensor]:

    prepared = prepare_waveform(
        input_waveform,
        input_sample_rate
    )

    prepared = prepared.to(
        device
    )

    with torch.inference_mode():

        features, _ = model.extract_features(
            prepared
        )

    selected: dict[str, torch.Tensor] = {}

    for (
        layer_name,
        layer_index
    ) in SELECTED_LAYERS.items():

        selected[layer_name] = (
            features[layer_index]
            .detach()
            .cpu()
        )

    return selected


# ==================================================
# Representation similarity
# ==================================================

def representation_similarity(
    reference_representation: torch.Tensor,
    compressed_representation: torch.Tensor
) -> float:

    min_frames = min(
        reference_representation.shape[1],
        compressed_representation.shape[1]
    )

    reference_representation = (
        reference_representation[
            :,
            :min_frames,
            :
        ]
    )

    compressed_representation = (
        compressed_representation[
            :,
            :min_frames,
            :
        ]
    )

    frame_similarity = (
        F.cosine_similarity(
            reference_representation,
            compressed_representation,
            dim=-1
        )
    )

    mean_similarity = (
        frame_similarity.mean()
    )

    return float(
        mean_similarity.item()
    )


# ==================================================
# Run analysis
# ==================================================

results: list[dict[str, object]] = []

for dataset_index in tqdm(
    sample_indices,
    desc="Utterances",
    unit="sample"
):

    (
        waveform,
        sample_rate,
        transcript,
        speaker_id,
        chapter_id,
        utterance_id,
    ) = dataset[
        dataset_index
    ]

    # WAV baseline features
    wav_features = extract_hidden_states(
        waveform,
        sample_rate
    )

    for condition in CONDITIONS:

        codec = str(
            condition["codec"]
        )

        bitrate = str(
            condition["bitrate"]
        )

        (
            compressed_waveform,
            compressed_sr
        ) = compress_audio(
            waveform,
            sample_rate,
            codec,
            bitrate
        )

        compressed_features = (
            extract_hidden_states(
                compressed_waveform,
                compressed_sr
            )
        )

        for layer_name in SELECTED_LAYERS:

            cosine_similarity = (
                representation_similarity(
                    wav_features[layer_name],
                    compressed_features[layer_name]
                )
            )

            representation_drift = (
                1.0
                -
                cosine_similarity
            )

            results.append({

                "dataset":
                    DATASET_NAME,

                "dataset_index":
                    dataset_index,

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

                "layer":
                    layer_name,

                "cosine_similarity":
                    cosine_similarity,

                "representation_drift":
                    representation_drift,
            })


# ==================================================
# Save detailed results
# ==================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    DETAIL_OUTPUT_PATH,
    index=False
)


# ==================================================
# Summary statistics
# ==================================================

summary_df = (
    results_df
    .groupby(
        [
            "codec",
            "bitrate",
            "layer"
        ]
    )
    .agg(

        cosine_mean=(
            "cosine_similarity",
            "mean"
        ),

        cosine_median=(
            "cosine_similarity",
            "median"
        ),

        cosine_std=(
            "cosine_similarity",
            "std"
        ),

        drift_mean=(
            "representation_drift",
            "mean"
        ),

        drift_median=(
            "representation_drift",
            "median"
        ),

        drift_std=(
            "representation_drift",
            "std"
        ),
    )
    .reset_index()
)

summary_df.to_csv(
    SUMMARY_OUTPUT_PATH,
    index=False
)


# ==================================================
# Print summary
# ==================================================

print()
print(
    "=" * 100
)

print(
    "REPRESENTATION SIMILARITY SUMMARY"
)

print(
    "=" * 100
)

print(
    summary_df.to_string(
        index=False
    )
)


# ==================================================
# Plot representation drift by layer
# ==================================================

layer_order = [
    "layer_1",
    "layer_6",
    "layer_12"
]

layer_positions = [
    1,
    6,
    12
]

plt.figure(
    figsize=(
        10,
        6
    )
)

for condition in CONDITIONS:

    codec = str(
        condition["codec"]
    )

    bitrate = str(
        condition["bitrate"]
    )

    subset = summary_df[
        (
            summary_df["codec"]
            ==
            codec
        )
        &
        (
            summary_df["bitrate"]
            ==
            bitrate
        )
    ].copy()

    drift_values = []

    for layer_name in layer_order:

        row = subset[
            subset["layer"]
            ==
            layer_name
        ]

        if row.empty:

            drift_values.append(
                float("nan")
            )

        else:

            drift_values.append(
                float(
                    row[
                        "drift_mean"
                    ].iloc[0]
                )
            )

    plt.plot(
        layer_positions,
        drift_values,
        marker="o",
        label=(
            f"{codec.upper()} "
            f"{bitrate}"
        )
    )


plt.xlabel(
    "Wav2Vec2 Transformer Layer"
)

plt.ylabel(
    "Mean Representation Drift (1 - Cosine Similarity)"
)

plt.title(
    "Representation Drift Across Wav2Vec2 Layers"
)

plt.xticks(
    layer_positions
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_OUTPUT_PATH,
    dpi=300
)

plt.close()


# ==================================================
# Final output
# ==================================================

print()
print(
    "Saved:"
)

print(
    DETAIL_OUTPUT_PATH
)

print(
    SUMMARY_OUTPUT_PATH
)

print(
    FIGURE_OUTPUT_PATH
)