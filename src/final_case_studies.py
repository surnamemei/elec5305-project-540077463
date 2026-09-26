"""
Robust-versus-failure case study and drift-quartile analysis for the final
report (feedback point 35).

Selection rule (fixed in advance, no manual choice):

    condition: Opus 8 kbps, test-clean (the narrowband transition)
    pool:      utterances whose WAV transcript is error-free and whose
               duration is 4-10 s (readable spectrograms)

    failure case: Opus 8k transcript has >= 1 error AND standardised
                  layer-12 drift >= 75th percentile of the condition;
                  the utterance with the median layer-12 drift of this group
                  is chosen (representative, not the most extreme case)

    robust case:  Opus 8k transcript is error-free AND layer-12 drift
                  <= 25th percentile of the condition; the utterance whose
                  log-spectral distance is closest to that of the failure
                  case is chosen (matched signal distortion)

The quartile table gives the population context: for every condition, the
fraction of WAV-correct utterances that acquire a new error, in the top and
bottom quartile of layer-12 drift and of log-spectral distance.

Outputs:
    results/final_report/case_study_selection.csv
    results/final_report/table_drift_quartiles.csv
    results/final_report/fig4_case_study.png
"""

import os
import subprocess
import tempfile
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "final_report"

DATA_ROOT = "data"

CASE_DATASET = "test-clean"
CASE_CODEC = "opus"
CASE_BITRATE = "8k"

MIN_DURATION_S = 4.0
MAX_DURATION_S = 10.0

N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
DYNAMIC_RANGE_DB = 80.0

LAYER_NAMES = ["conv"] + [f"layer_{i}" for i in range(1, 13)]


def load_merged(dataset_name: str) -> pd.DataFrame:
    """Per (utterance, condition): errors, WAV errors, LSD, drift profile."""

    details = pd.read_csv(RESULTS_DIR / f"{dataset_name}_experiment_details.csv")
    details["num_words"] = details["reference"].str.split().str.len()
    details["errors"] = [
        jiwer.process_words(r, p).wer * len(r.split())
        for r, p in zip(details["reference"], details["prediction"])
    ]
    details["errors"] = details["errors"].round().astype(int)

    wav = details[details["codec"] == "wav"][
        ["dataset_index", "errors", "prediction"]
    ].rename(columns={"errors": "wav_errors", "prediction": "wav_prediction"})

    compressed = details[details["codec"] != "wav"].merge(wav, on="dataset_index")

    signal = pd.read_csv(RESULTS_DIR / "signal_distortion_results.csv")
    signal = signal[signal["dataset"] == dataset_name]

    rep = pd.read_csv(RESULTS_DIR / "representation_similarity_results.csv")
    rep = rep[rep["dataset"] == dataset_name]

    keys = ["dataset_index", "codec", "bitrate"]
    merged = compressed.merge(
        signal[keys + ["lsd_db", "retained_bandwidth_hz"]], on=keys
    ).merge(
        rep[keys + [f"sdrift_{layer}" for layer in LAYER_NAMES]], on=keys
    )
    merged["dataset"] = dataset_name
    merged["new_error"] = merged["errors"] > merged["wav_errors"]
    return merged


def quartile_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (codec, bitrate), group in merged.groupby(["codec", "bitrate"], sort=False):
        pool = group[group["wav_errors"] == 0]
        row = {
            "dataset": group["dataset"].iloc[0],
            "codec": codec,
            "bitrate": bitrate,
            "wav_correct_utterances": len(pool),
            "new_error_rate_all": pool["new_error"].mean(),
        }
        for measure, label in [("sdrift_layer_12", "l12"), ("lsd_db", "lsd")]:
            low, high = group[measure].quantile([0.25, 0.75])
            top = pool[pool[measure] >= high]
            bottom = pool[pool[measure] <= low]
            row[f"{label}_top_quartile_n"] = len(top)
            row[f"{label}_top_quartile_new_error_rate"] = top["new_error"].mean()
            row[f"{label}_bottom_quartile_n"] = len(bottom)
            row[f"{label}_bottom_quartile_new_error_rate"] = bottom["new_error"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compress(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, f"compressed.{CASE_CODEC}")
        torchaudio.save(input_path, waveform, sample_rate)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
                "-codec:a", "libopus", "-b:a", CASE_BITRATE, output_path,
            ],
            check=True,
        )
        decoded, decoded_sr = torchaudio.load(output_path)
    return torchaudio.functional.resample(decoded, decoded_sr, sample_rate)


def log_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    stft = torch.stft(
        waveform[0],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        return_complex=True,
    )
    return 20.0 * torch.log10(stft.abs() + 1e-12)


def word_differences(reference: str, hypothesis: str) -> str:
    output = jiwer.process_words(reference, hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    changes = []
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            continue
        ref_part = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx]) or "∅"
        hyp_part = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]) or "∅"
        changes.append(f"{chunk.type}: {ref_part} → {hyp_part}")
    return "; ".join(changes) if changes else "none"


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = {name: load_merged(name) for name in ["test-clean", "test-other"]}

    pd.concat(
        [quartile_table(frame) for frame in merged.values()], ignore_index=True
    ).to_csv(OUTPUT_DIR / "table_drift_quartiles.csv", index=False)

    # ----- case selection ----------------------------------------------------

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=DATA_ROOT, url=CASE_DATASET, download=False
    )

    condition = merged[CASE_DATASET][
        (merged[CASE_DATASET]["codec"] == CASE_CODEC)
        & (merged[CASE_DATASET]["bitrate"] == CASE_BITRATE)
    ].copy()

    durations = {}
    for index in condition["dataset_index"]:
        waveform, sample_rate, *_ = dataset[int(index)]
        durations[index] = waveform.shape[1] / sample_rate
    condition["duration_s"] = condition["dataset_index"].map(durations)

    q25, q75 = condition["sdrift_layer_12"].quantile([0.25, 0.75])

    pool = condition[
        (condition["wav_errors"] == 0)
        & condition["duration_s"].between(MIN_DURATION_S, MAX_DURATION_S)
    ]

    failures = pool[(pool["errors"] >= 1) & (pool["sdrift_layer_12"] >= q75)]
    failures = failures.sort_values("sdrift_layer_12").reset_index(drop=True)
    failure = failures.iloc[(len(failures) - 1) // 2]

    robust_pool = pool[(pool["errors"] == 0) & (pool["sdrift_layer_12"] <= q25)]
    robust = robust_pool.iloc[
        (robust_pool["lsd_db"] - failure["lsd_db"]).abs().argsort().iloc[0]
    ]

    selection = []
    for role, case, pool_size in [
        ("robust", robust, len(robust_pool)),
        ("failure", failure, len(failures)),
    ]:
        selection.append({
            "role": role,
            "dataset": CASE_DATASET,
            "condition": f"{CASE_CODEC} {CASE_BITRATE}",
            "dataset_index": int(case["dataset_index"]),
            "utterance": f"{case['speaker_id']}-{case['chapter_id']}-{case['utterance_id']:04d}",
            "duration_s": case["duration_s"],
            "eligible_pool_size": pool_size,
            "lsd_db": case["lsd_db"],
            "retained_bandwidth_hz": case["retained_bandwidth_hz"],
            "sdrift_conv": case["sdrift_conv"],
            "sdrift_layer_12": case["sdrift_layer_12"],
            "condition_sdrift_layer_12_q25": q25,
            "condition_sdrift_layer_12_q75": q75,
            "wav_errors": int(case["wav_errors"]),
            "compressed_errors": int(case["errors"]),
            "reference": case["reference"],
            "compressed_prediction": case["prediction"],
            "word_changes": word_differences(case["reference"], case["prediction"]),
        })

    selection_df = pd.DataFrame(selection)
    selection_df.to_csv(OUTPUT_DIR / "case_study_selection.csv", index=False)
    print(selection_df.drop(columns=["reference", "compressed_prediction"]).T)

    # ----- figure ------------------------------------------------------------

    layer_mean = condition[[f"sdrift_{l}" for l in LAYER_NAMES]].mean()
    layer_q25 = condition[[f"sdrift_{l}" for l in LAYER_NAMES]].quantile(0.25)
    layer_q75 = condition[[f"sdrift_{l}" for l in LAYER_NAMES]].quantile(0.75)

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 6.6),
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]},
    )

    for row_index, (case, title) in enumerate([
        (robust, "Robust case (no new error)"),
        (failure, "Failure case (new recognition error)"),
    ]):
        waveform, sample_rate, *_ = dataset[int(case["dataset_index"])]
        compressed = compress(waveform, sample_rate)
        length = min(waveform.shape[1], compressed.shape[1])

        reference_log = log_spectrogram(waveform[:, :length])
        compressed_log = log_spectrogram(compressed[:, :length])
        peak = reference_log.max()

        extent = [0, length / sample_rate, 0, sample_rate / 2000]

        for col_index, (spectrum, label) in enumerate([
            (reference_log, "WAV"),
            (compressed_log, "Opus 8 kbps"),
        ]):
            ax = axes[row_index][col_index]
            image = ax.imshow(
                (spectrum - peak).numpy(),
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=-DYNAMIC_RANGE_DB,
                vmax=0,
                cmap="magma",
            )
            ax.set_title(
                f"{title}\n{label}" if col_index == 0 else f"\n{label}",
                fontsize=9,
                loc="left",
            )
            ax.set_ylabel("Frequency (kHz)")
            ax.set_xlabel("Time (s)")

        cbar = fig.colorbar(image, ax=axes[row_index][1], pad=0.02)
        cbar.set_label("dB re. WAV peak", fontsize=8)

        ax = axes[row_index][2]
        positions = np.arange(len(LAYER_NAMES))
        ax.fill_between(
            positions, layer_q25.to_numpy(), layer_q75.to_numpy(),
            color="#6b6b6b", alpha=0.18, linewidth=0,
            label="Condition IQR",
        )
        ax.plot(
            positions, layer_mean.to_numpy(),
            color="#6b6b6b", linewidth=2, label="Condition mean",
        )
        ax.plot(
            positions,
            case[[f"sdrift_{l}" for l in LAYER_NAMES]].to_numpy(dtype=float),
            color="#eb6834", linewidth=2, marker="o", markersize=4,
            label="This utterance",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(["conv"] + [str(i) for i in range(1, 13)], fontsize=7)
        ax.set_ylabel("Standardised drift (1 − cos)")
        ax.set_xlabel("Wav2Vec2 layer")
        ax.set_title(
            f"LSD {case['lsd_db']:.1f} dB, "
            f"layer-12 drift {case['sdrift_layer_12']:.3f}",
            fontsize=9, loc="left",
        )
        ax.grid(True, alpha=0.3)
        if row_index == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_case_study.png", dpi=300)
    plt.close(fig)

    print(f"Saved {OUTPUT_DIR / 'fig4_case_study.png'}")


if __name__ == "__main__":
    main()
