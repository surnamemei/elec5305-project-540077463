from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from encodec.model import EncodecModel
from jiwer import wer
from torch import Tensor
from torchaudio.models import Wav2Vec2Model


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NUM_SAMPLES = 100
RANDOM_SEED = 5305

LIBRISPEECH_ROOT = "./data"
DATASET_NAME = "test-clean"

SOURCE_SAMPLE_RATE = 16000
ENCODEC_SAMPLE_RATE = 24000

TARGET_BANDWIDTHS = [24.0, 6.0, 1.5]

LAYERS = {
    1: 0,
    6: 5,
    12: 11,
}

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

DETAIL_FILE = RESULTS_DIR / "encodec_extension_results.csv"
SUMMARY_FILE = RESULTS_DIR / "encodec_extension_summary.csv"
FIGURE_FILE = FIGURES_DIR / "encodec_extension.png"


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def resample_waveform(
    waveform: Tensor,
    original_rate: int,
    target_rate: int,
) -> Tensor:
    if original_rate == target_rate:
        return waveform

    return torchaudio.functional.resample(
        waveform,
        original_rate,
        target_rate,
    )


def prepare_mono(waveform: Tensor) -> Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def resampling_control(waveform: Tensor) -> Tensor:
    """16 kHz -> 24 kHz -> 16 kHz, without compression."""
    wav_24k = resample_waveform(
        waveform,
        SOURCE_SAMPLE_RATE,
        ENCODEC_SAMPLE_RATE,
    )

    wav_16k = resample_waveform(
        wav_24k,
        ENCODEC_SAMPLE_RATE,
        SOURCE_SAMPLE_RATE,
    )

    return wav_16k


def encodec_round_trip(
    waveform: Tensor,
    model: EncodecModel,
    bandwidth: float,
    device: torch.device,
) -> Tensor:
    """
    Resample 16 -> 24 kHz, encode/decode with EnCodec,
    then resample back to 16 kHz.
    """
    wav_24k = resample_waveform(
        waveform,
        SOURCE_SAMPLE_RATE,
        ENCODEC_SAMPLE_RATE,
    )

    wav_24k = prepare_mono(wav_24k)
    wav_24k = wav_24k.unsqueeze(0).to(device)

    model.set_target_bandwidth(bandwidth)

    with torch.no_grad():
        encoded_frames = model.encode(wav_24k)
        decoded_24k = model.decode(encoded_frames)

    decoded_24k = decoded_24k.squeeze(0).cpu()

    decoded_16k = resample_waveform(
        decoded_24k,
        ENCODEC_SAMPLE_RATE,
        SOURCE_SAMPLE_RATE,
    )

    return decoded_16k


def match_length(
    reference: Tensor,
    processed: Tensor,
) -> tuple[Tensor, Tensor]:
    length = min(
        reference.shape[-1],
        processed.shape[-1],
    )

    return (
        reference[..., :length],
        processed[..., :length],
    )


def spectral_distortion(
    reference: Tensor,
    processed: Tensor,
) -> float:
    reference, processed = match_length(
        reference,
        processed,
    )

    window = torch.hann_window(400)

    ref_stft = torch.stft(
        reference.squeeze(0),
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        return_complex=True,
    )

    proc_stft = torch.stft(
        processed.squeeze(0),
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=window,
        return_complex=True,
    )

    ref_log = 20.0 * torch.log10(
        ref_stft.abs() + 1e-8
    )

    proc_log = 20.0 * torch.log10(
        proc_stft.abs() + 1e-8
    )

    distortion = torch.mean(
        (ref_log - proc_log) ** 2
    )

    return float(distortion.item())


def extract_hidden_layers(
    waveform: Tensor,
    model: Wav2Vec2Model,
    device: torch.device,
) -> dict[int, Tensor]:
    waveform = waveform.to(device)

    with torch.no_grad():
        features, _ = model.extract_features(waveform)

    selected: dict[int, Tensor] = {}

    for layer_number, layer_index in LAYERS.items():
        selected[layer_number] = (
            features[layer_index]
            .detach()
            .cpu()
        )

    return selected


def representation_drift(
    reference_features: Tensor,
    processed_features: Tensor,
) -> float:
    time_length = min(
        reference_features.shape[1],
        processed_features.shape[1],
    )

    ref = reference_features[:, :time_length, :]
    proc = processed_features[:, :time_length, :]

    similarity = F.cosine_similarity(
        ref,
        proc,
        dim=-1,
    ).mean()

    drift = 1.0 - similarity

    return float(drift.item())


def greedy_decode(
    emissions: Tensor,
    labels: tuple[str, ...],
) -> str:
    indices = torch.argmax(
        emissions,
        dim=-1,
    )[0]

    tokens: list[str] = []

    previous = -1

    for index in indices.tolist():
        if index != previous and index != 0:
            tokens.append(labels[index])

        previous = index

    return "".join(tokens).replace("|", " ").strip()


def transcribe(
    waveform: Tensor,
    model: Wav2Vec2Model,
    labels: tuple[str, ...],
    device: torch.device,
) -> str:
    waveform = waveform.to(device)

    with torch.no_grad():
        emissions, _ = model(waveform)

    return greedy_decode(
        emissions.cpu(),
        labels,
    )


def normalize_reference(text: str) -> str:
    return text.upper().strip()


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------
    # Load Wav2Vec2
    # -----------------------------------------------------

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    asr_model = cast(
        Wav2Vec2Model,
        bundle.get_model(),
    )

    asr_model.to(device)
    asr_model.eval()

    labels = bundle.get_labels()

    # -----------------------------------------------------
    # Load EnCodec
    # -----------------------------------------------------

    encodec_model = (
        EncodecModel.encodec_model_24khz()
    )

    encodec_model.to(device)
    encodec_model.eval()

    # -----------------------------------------------------
    # Load LibriSpeech
    # -----------------------------------------------------

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=LIBRISPEECH_ROOT,
        url=DATASET_NAME,
        download=False,
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:NUM_SAMPLES].tolist()

    conditions: list[tuple[str, float | None]] = [
        ("wav", None),
        ("resample_control", None),
        ("encodec_24k", 24.0),
        ("encodec_6k", 6.0),
        ("encodec_1.5k", 1.5),
    ]

    detail_rows: list[dict[str, object]] = []

    # -----------------------------------------------------
    # Run experiment
    # -----------------------------------------------------

    for sample_number, dataset_index in enumerate(
        indices,
        start=1,
    ):
        (
            waveform,
            sample_rate,
            transcript,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = dataset[dataset_index]

        waveform = prepare_mono(waveform)

        if sample_rate != SOURCE_SAMPLE_RATE:
            waveform = resample_waveform(
                waveform,
                sample_rate,
                SOURCE_SAMPLE_RATE,
            )

        reference_text = normalize_reference(
            transcript
        )

        # Reference features are calculated once.
        reference_features = extract_hidden_layers(
            waveform,
            asr_model,
            device,
        )

        for condition_name, bandwidth in conditions:
            if condition_name == "wav":
                processed = waveform.clone()

            elif condition_name == "resample_control":
                processed = resampling_control(
                    waveform
                )

            else:
                if bandwidth is None:
                    raise ValueError(
                        "EnCodec condition requires bandwidth."
                    )

                processed = encodec_round_trip(
                    waveform,
                    encodec_model,
                    bandwidth,
                    device,
                )

            reference_aligned, processed_aligned = (
                match_length(
                    waveform,
                    processed,
                )
            )

            prediction = transcribe(
                processed_aligned,
                asr_model,
                labels,
                device,
            )

            sample_wer = wer(
                reference_text,
                prediction,
            )

            if condition_name == "wav":
                dspec = 0.0

                layer_drifts = {
                    layer: 0.0
                    for layer in LAYERS
                }

            else:
                dspec = spectral_distortion(
                    reference_aligned,
                    processed_aligned,
                )

                processed_features = extract_hidden_layers(
                    processed_aligned,
                    asr_model,
                    device,
                )

                layer_drifts = {
                    layer: representation_drift(
                        reference_features[layer],
                        processed_features[layer],
                    )
                    for layer in LAYERS
                }

            detail_rows.append(
                {
                    "dataset_index": dataset_index,
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "utterance_id": utterance_id,
                    "condition": condition_name,
                    "target_bitrate_kbps": bandwidth,
                    "reference": reference_text,
                    "prediction": prediction,
                    "wer": sample_wer,
                    "spectral_distortion": dspec,
                    "layer_1_drift": layer_drifts[1],
                    "layer_6_drift": layer_drifts[6],
                    "layer_12_drift": layer_drifts[12],
                }
            )

        print(
            f"[{sample_number}/{NUM_SAMPLES}] "
            f"Processed dataset index "
            f"{dataset_index}"
        )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    detail_df = pd.DataFrame(detail_rows)

    detail_df.to_csv(
        DETAIL_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------

    summary_rows: list[dict[str, object]] = []

    for condition_name, group in detail_df.groupby("condition"):
        references = group["reference"].astype(str).tolist()
        predictions = group["prediction"].astype(str).tolist()

        corpus_wer = wer(
            references,
            predictions,
        )

        summary_rows.append(
            {
                "condition": condition_name,
                "mean_wer": corpus_wer,
                "mean_spectral_distortion": group[
                    "spectral_distortion"
                ].mean(),
                "mean_layer_1_drift": group[
                    "layer_1_drift"
                ].mean(),
                "mean_layer_6_drift": group[
                    "layer_6_drift"
                ].mean(),
                "mean_layer_12_drift": group[
                    "layer_12_drift"
                ].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    wav_wer_series = summary_df.loc[
        summary_df["condition"] == "wav",
        "mean_wer",
    ]

    if wav_wer_series.empty:
        raise RuntimeError(
            "WAV baseline missing from summary."
        )

    wav_wer = float(
        wav_wer_series.iloc[0]
    )

    summary_df["delta_wer"] = (
        summary_df["mean_wer"]
        - wav_wer
    )

    summary_df["wer_percent"] = (
        summary_df["mean_wer"]
        * 100.0
    )

    summary_df[
        "delta_wer_percentage_points"
    ] = (
        summary_df["delta_wer"]
        * 100.0
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\nEnCodec extension summary:\n")

    print(
        summary_df[
            [
                "condition",
                "wer_percent",
                "delta_wer_percentage_points",
                "mean_spectral_distortion",
                "mean_layer_1_drift",
                "mean_layer_6_drift",
                "mean_layer_12_drift",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------

    import matplotlib.pyplot as plt

    condition_order = [
        "wav",
        "resample_control",
        "encodec_24k",
        "encodec_6k",
        "encodec_1.5k",
    ]

    summary_df["condition"] = pd.Categorical(
        summary_df["condition"],
        categories=condition_order,
        ordered=True,
    )

    summary_df = summary_df.sort_values(
        "condition"
    )

    labels_for_plot = [
        "WAV",
        "16→24→16",
        "EnCodec 24k",
        "EnCodec 6k",
        "EnCodec 1.5k",
    ]

    x = range(len(summary_df))

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    axes[0].plot(
        x,
        summary_df[
            "mean_spectral_distortion"
        ],
        marker="o",
    )

    axes[0].set_ylabel(
        "Log-spectral distortion"
    )

    axes[0].set_title(
        "EnCodec Extension: "
        "Signal → Representation → Recognition"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].plot(
        x,
        summary_df[
            "mean_layer_1_drift"
        ],
        marker="o",
        label="Layer 1",
    )

    axes[1].plot(
        x,
        summary_df[
            "mean_layer_6_drift"
        ],
        marker="o",
        label="Layer 6",
    )

    axes[1].plot(
        x,
        summary_df[
            "mean_layer_12_drift"
        ],
        marker="o",
        label="Layer 12",
    )

    axes[1].set_ylabel(
        "Representation drift"
    )

    axes[1].legend()

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[2].plot(
        x,
        summary_df["wer_percent"],
        marker="o",
    )

    axes[2].set_ylabel(
        "WER (%)"
    )

    axes[2].set_xlabel(
        "Condition"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    axes[2].set_xticks(
        list(x)
    )

    axes[2].set_xticklabels(
        labels_for_plot,
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved detailed results: "
        f"{DETAIL_FILE}"
    )

    print(
        f"Saved summary: "
        f"{SUMMARY_FILE}"
    )

    print(
        f"Saved figure: "
        f"{FIGURE_FILE}"
    )


if __name__ == "__main__":
    main()