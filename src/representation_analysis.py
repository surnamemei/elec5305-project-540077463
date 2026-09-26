"""
Wav2Vec2 representation drift analysis (feedback points 18-22).

For every selected utterance and codec condition, hidden representations of
the compressed signal are compared with those of the WAV signal at:

- the convolutional feature-encoder output ("conv");
- every transformer layer ("layer_1" ... "layer_12").

Similarity is the frame-wise cosine similarity averaged over frames, and
representation drift is D = 1 - cosine similarity.

Two versions of the drift are stored:

- drift_<layer>:  cosine on the raw hidden states;
- sdrift_<layer>: cosine after standardising every hidden dimension with a
  per-layer mean and standard deviation estimated from WAV frames of a fixed
  calibration set (Timkey & van Schijndel, 2021).

Raw cosine is not comparable across layers because some Wav2Vec2 layers are
strongly anisotropic: in layer 11, frames of two unrelated utterances already
have a cosine similarity of about 0.94, so raw drift in that layer is close
to zero whatever the input. Standardisation removes the shared direction and
makes unrelated frames close to orthogonal in every layer, so sdrift is the
measure used for comparisons across layers.

The detailed output is stored in wide format (one row per utterance and
condition, one drift column per layer) and the summary in long format (one
row per condition and layer).
"""

import os
import random
import subprocess
import tempfile
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models import Wav2Vec2Model
from tqdm import tqdm


# ==================================================
# Settings
# ==================================================

DATA_ROOT = "data"
DATASET_NAMES = ["test-clean", "test-other"]

# Same utterances as run_all_experiments.py (same seed and sample size)
NUM_SAMPLES = 500
RANDOM_SEED = 5305

RESULTS_DIR = "results"
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")

DETAIL_OUTPUT_PATH = os.path.join(
    RESULTS_DIR, "representation_similarity_results.csv"
)
SUMMARY_OUTPUT_PATH = os.path.join(
    RESULTS_DIR, "representation_similarity_summary.csv"
)
FIGURE_OUTPUT_PATH = os.path.join(
    FIGURE_DIR, "representation_drift_by_layer.png"
)
STANDARDISATION_OUTPUT_PATH = os.path.join(
    RESULTS_DIR, "representation_standardisation.pt"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ==================================================
# Compression conditions (full sweep)
# ==================================================

CONDITIONS = [
    {"codec": "mp3", "bitrate": "128k"},
    {"codec": "mp3", "bitrate": "64k"},
    {"codec": "mp3", "bitrate": "32k"},
    {"codec": "mp3", "bitrate": "24k"},
    {"codec": "mp3", "bitrate": "16k"},
    {"codec": "opus", "bitrate": "64k"},
    {"codec": "opus", "bitrate": "32k"},
    {"codec": "opus", "bitrate": "16k"},
    {"codec": "opus", "bitrate": "12k"},
    {"codec": "opus", "bitrate": "8k"},
    {"codec": "opus", "bitrate": "6k"},
]

# WAV utterances used to estimate the per-layer standardisation statistics
# (the first CALIBRATION_UTTERANCES selected test-clean utterances)
CALIBRATION_DATASET = "test-clean"
CALIBRATION_UTTERANCES = 100

NUM_TRANSFORMER_LAYERS = 12
LAYER_NAMES = ["conv"] + [
    f"layer_{i}" for i in range(1, NUM_TRANSFORMER_LAYERS + 1)
]


# ==================================================
# Device and fixed Wav2Vec2 model
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = cast(Wav2Vec2Model, bundle.get_model()).to(device)
model.eval()

TARGET_SAMPLE_RATE = int(bundle.sample_rate)

print("Wav2Vec2 model loaded.")
print("Target sample rate:", TARGET_SAMPLE_RATE)


# ==================================================
# Compression helper
# ==================================================

def compress_audio(
    input_waveform: torch.Tensor,
    input_sample_rate: int,
    codec: str,
    bitrate: str,
) -> tuple[torch.Tensor, int]:

    encoders = {"mp3": "libmp3lame", "opus": "libopus"}

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, f"compressed.{codec}")

        torchaudio.save(input_path, input_waveform, input_sample_rate)

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-codec:a", encoders[codec],
            "-b:a", bitrate,
            output_path,
        ]
        subprocess.run(command, check=True)

        compressed_waveform, compressed_sr = torchaudio.load(output_path)

    return compressed_waveform, int(compressed_sr)


# ==================================================
# Extract hidden states
# ==================================================

def extract_hidden_states(
    input_waveform: torch.Tensor,
    input_sample_rate: int,
) -> dict[str, torch.Tensor]:

    if input_sample_rate != TARGET_SAMPLE_RATE:
        input_waveform = torchaudio.functional.resample(
            input_waveform, input_sample_rate, TARGET_SAMPLE_RATE
        )

    prepared = input_waveform.to(device)

    with torch.inference_mode():
        conv_features, _ = model.feature_extractor(prepared, None)
        layer_features, _ = model.extract_features(prepared)

    selected = {"conv": conv_features.detach().cpu()}

    for layer_number in range(1, NUM_TRANSFORMER_LAYERS + 1):
        selected[f"layer_{layer_number}"] = (
            layer_features[layer_number - 1].detach().cpu()
        )

    return selected


# ==================================================
# Representation similarity
# ==================================================

def representation_similarity(
    reference_representation: torch.Tensor,
    compressed_representation: torch.Tensor,
) -> float:

    min_frames = min(
        reference_representation.shape[1],
        compressed_representation.shape[1],
    )

    frame_similarity = F.cosine_similarity(
        reference_representation[:, :min_frames, :],
        compressed_representation[:, :min_frames, :],
        dim=-1,
    )

    return float(frame_similarity.mean().item())


# ==================================================
# Selected utterances
# ==================================================

def select_indices(dataset) -> list[int]:
    random.seed(RANDOM_SEED)
    return random.sample(
        range(len(dataset)),
        k=min(NUM_SAMPLES, len(dataset)),
    )


def load_dataset(dataset_name: str):
    return torchaudio.datasets.LIBRISPEECH(
        root=DATA_ROOT,
        url=dataset_name,
        download=False,
    )


# ==================================================
# Standardisation statistics (per layer, per dimension)
# ==================================================

def estimate_standardisation() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:

    dataset = load_dataset(CALIBRATION_DATASET)
    indices = select_indices(dataset)[:CALIBRATION_UTTERANCES]

    sums: dict[str, torch.Tensor] = {}
    squares: dict[str, torch.Tensor] = {}
    count = 0

    for dataset_index in tqdm(indices, desc="Calibration", unit="sample"):
        waveform, sample_rate, *_ = dataset[dataset_index]
        features = extract_hidden_states(waveform, sample_rate)

        for layer_name, feature in features.items():
            frames = feature[0].double()
            sums[layer_name] = sums.get(layer_name, 0) + frames.sum(dim=0)
            squares[layer_name] = (
                squares.get(layer_name, 0) + (frames ** 2).sum(dim=0)
            )

        count += features["conv"].shape[1]

    statistics = {}
    for layer_name in sums:
        mean = sums[layer_name] / count
        std = torch.sqrt(squares[layer_name] / count - mean ** 2)
        statistics[layer_name] = (
            mean.float(),
            torch.clamp(std, min=1e-6).float(),
        )

    return statistics


standardisation = estimate_standardisation()

# Saved so that other scripts (encodec_extension.py) use identical statistics
torch.save(standardisation, STANDARDISATION_OUTPUT_PATH)


def standardise(
    representation: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    mean, std = standardisation[layer_name]
    return (representation - mean) / std


# ==================================================
# Run analysis
# ==================================================

results: list[dict[str, object]] = []

for dataset_name in DATASET_NAMES:

    dataset = load_dataset(dataset_name)
    sample_indices = select_indices(dataset)

    print("Dataset:", dataset_name)
    print("Selected utterances:", len(sample_indices))

    for dataset_index in tqdm(sample_indices, desc=dataset_name, unit="sample"):

        (
            waveform,
            sample_rate,
            transcript,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = dataset[dataset_index]

        wav_features = extract_hidden_states(waveform, sample_rate)

        for condition in CONDITIONS:

            codec = condition["codec"]
            bitrate = condition["bitrate"]

            compressed_waveform, compressed_sr = compress_audio(
                waveform, sample_rate, codec, bitrate
            )

            compressed_features = extract_hidden_states(
                compressed_waveform, compressed_sr
            )

            row: dict[str, object] = {
                "dataset": dataset_name,
                "dataset_index": dataset_index,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
                "codec": codec,
                "bitrate": bitrate,
            }

            for layer_name in LAYER_NAMES:
                cosine_similarity = representation_similarity(
                    wav_features[layer_name],
                    compressed_features[layer_name],
                )
                row[f"drift_{layer_name}"] = 1.0 - cosine_similarity

                standardised_similarity = representation_similarity(
                    standardise(wav_features[layer_name], layer_name),
                    standardise(compressed_features[layer_name], layer_name),
                )
                row[f"sdrift_{layer_name}"] = 1.0 - standardised_similarity

            results.append(row)


# ==================================================
# Save detailed results (wide format)
# ==================================================

results_df = pd.DataFrame(results)
results_df.to_csv(DETAIL_OUTPUT_PATH, index=False, float_format="%.6g")


# ==================================================
# Summary statistics (long format)
# ==================================================

id_columns = ["dataset", "dataset_index", "codec", "bitrate"]

long_df = results_df.melt(
    id_vars=id_columns,
    value_vars=[f"drift_{layer}" for layer in LAYER_NAMES],
    var_name="layer",
    value_name="representation_drift",
)
long_df["layer"] = long_df["layer"].str.replace("drift_", "", regex=False)
long_df["cosine_similarity"] = 1.0 - long_df["representation_drift"]

standardised_long_df = results_df.melt(
    id_vars=id_columns,
    value_vars=[f"sdrift_{layer}" for layer in LAYER_NAMES],
    var_name="layer",
    value_name="standardised_drift",
)
standardised_long_df["layer"] = standardised_long_df["layer"].str.replace(
    "sdrift_", "", regex=False
)

long_df = long_df.merge(
    standardised_long_df, on=id_columns + ["layer"], how="inner"
)

summary_df = (
    long_df.groupby(["dataset", "codec", "bitrate", "layer"], sort=False)
    .agg(
        cosine_mean=("cosine_similarity", "mean"),
        cosine_median=("cosine_similarity", "median"),
        cosine_std=("cosine_similarity", "std"),
        drift_mean=("representation_drift", "mean"),
        drift_median=("representation_drift", "median"),
        drift_std=("representation_drift", "std"),
        sdrift_mean=("standardised_drift", "mean"),
        sdrift_median=("standardised_drift", "median"),
        sdrift_std=("standardised_drift", "std"),
    )
    .reset_index()
)

summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

print()
print("=" * 100)
print("REPRESENTATION SIMILARITY SUMMARY (selected layers)")
print("=" * 100)
print(
    summary_df[
        summary_df["layer"].isin(["conv", "layer_1", "layer_6", "layer_12"])
    ].to_string(index=False)
)


# ==================================================
# Plot representation drift by layer
# rows: dataset, columns: codec, shade: bitrate (light = high bitrate)
# ==================================================

layer_positions = list(range(len(LAYER_NAMES)))
layer_ticks = ["conv"] + [str(i) for i in range(1, NUM_TRANSFORMER_LAYERS + 1)]

fig, axes = plt.subplots(
    len(DATASET_NAMES), 2,
    figsize=(12, 4.5 * len(DATASET_NAMES)),
    sharex=True,
    sharey=True,
    squeeze=False,
)

for row_index, dataset_name in enumerate(DATASET_NAMES):
    for col_index, (codec, cmap) in enumerate(
        [("mp3", "Blues"), ("opus", "Oranges")]
    ):
        ax = axes[row_index][col_index]

        codec_conditions = [c for c in CONDITIONS if c["codec"] == codec]
        shades = plt.get_cmap(cmap)(
            np.linspace(0.35, 0.95, len(codec_conditions))
        )

        for condition, shade in zip(codec_conditions, shades):
            subset = summary_df[
                (summary_df["dataset"] == dataset_name)
                & (summary_df["codec"] == codec)
                & (summary_df["bitrate"] == condition["bitrate"])
            ].set_index("layer")

            ax.plot(
                layer_positions,
                subset.loc[LAYER_NAMES, "sdrift_mean"].to_numpy(),
                marker="o",
                markersize=5,
                linewidth=2,
                color=shade,
                label=condition["bitrate"],
            )

        ax.set_title(f"{codec.upper()} - {dataset_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Target bitrate", fontsize=8)

for ax in axes[-1]:
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(layer_ticks)
    ax.set_xlabel("Wav2Vec2 layer (conv = feature-encoder output)")

for ax in axes[:, 0]:
    ax.set_ylabel("Mean standardised drift")

fig.suptitle(
    "Representation drift across Wav2Vec2 layers "
    "(1 - cosine similarity of standardised hidden states)"
)
fig.tight_layout()
fig.savefig(FIGURE_OUTPUT_PATH, dpi=300)
plt.close(fig)


# ==================================================
# Final output
# ==================================================

print()
print("Saved:")
print(DETAIL_OUTPUT_PATH)
print(SUMMARY_OUTPUT_PATH)
print(FIGURE_OUTPUT_PATH)
print(STANDARDISATION_OUTPUT_PATH)
