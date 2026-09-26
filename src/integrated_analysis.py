from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

DATASET_NAMES = ["test-clean", "test-other"]

SIGNAL_FILE = RESULTS_DIR / "signal_distortion_summary.csv"
REP_FILE = RESULTS_DIR / "representation_similarity_summary.csv"
BOOTSTRAP_FILE = RESULTS_DIR / "bootstrap_results.csv"

OUTPUT_CSV = RESULTS_DIR / "integrated_analysis_summary.csv"
OUTPUT_FIGURE = FIGURES_DIR / "integrated_analysis.png"

CODEC_COLOURS = {"mp3": "#2a78d6", "opus": "#eb6834"}

LAYERS = ["conv", "layer_1", "layer_6", "layer_12"]


def condition_label(codec: str, bitrate: str) -> str:
    """Create a readable condition label."""
    if codec == "wav":
        return "WAV"
    return f"{codec.upper()} {bitrate}"


def build_table(dataset_name: str) -> pd.DataFrame:
    """Merge WER + signal distortion + representation drift for one subset."""

    wer_df = pd.read_csv(RESULTS_DIR / f"{dataset_name}_summary_results.csv")

    signal_df = pd.read_csv(SIGNAL_FILE)
    signal_df = signal_df[signal_df["dataset"] == dataset_name]

    rep_df = pd.read_csv(REP_FILE)
    rep_df = rep_df[
        (rep_df["dataset"] == dataset_name) & rep_df["layer"].isin(LAYERS)
    ]

    # Standardised drift (comparable across layers) and raw cosine drift
    rep_pivot = rep_df.pivot(
        index=["codec", "bitrate"],
        columns="layer",
        values="sdrift_mean",
    ).rename(columns={layer: f"{layer}_drift" for layer in LAYERS})

    raw_pivot = rep_df.pivot(
        index=["codec", "bitrate"],
        columns="layer",
        values="drift_mean",
    ).rename(columns={layer: f"{layer}_raw_drift" for layer in LAYERS})

    rep_pivot = rep_pivot.join(raw_pivot).reset_index()

    bootstrap_df = pd.read_csv(BOOTSTRAP_FILE)
    bootstrap_df = bootstrap_df[bootstrap_df["dataset"] == dataset_name]

    integrated_df = wer_df[
        [
            "codec",
            "bitrate",
            "overall_wer",
            "delta_wer",
            "aggregate_actual_bitrate_kbps",
        ]
    ].copy()

    integrated_df = integrated_df.merge(
        signal_df[
            [
                "codec",
                "bitrate",
                "lsd_db_mean",
                "spectral_distortion_mean",
                "retained_bandwidth_mean_hz",
                "hf_power_change_mean_db",
            ]
        ],
        on=["codec", "bitrate"],
        how="left",
    )

    integrated_df = integrated_df.merge(
        rep_pivot,
        on=["codec", "bitrate"],
        how="left",
    )

    integrated_df = integrated_df.merge(
        bootstrap_df[["codec", "bitrate", "ci_lower_pp", "ci_upper_pp"]],
        on=["codec", "bitrate"],
        how="left",
    )

    # -----------------------------------------------------
    # WAV is the reference:
    # distortion = 0, full bandwidth and representation drift = 0
    # -----------------------------------------------------

    wav_mask = integrated_df["codec"] == "wav"

    for column in [
        "lsd_db_mean",
        "spectral_distortion_mean",
        "hf_power_change_mean_db",
        "ci_lower_pp",
        "ci_upper_pp",
    ] + [f"{layer}_drift" for layer in LAYERS] + [
        f"{layer}_raw_drift" for layer in LAYERS
    ]:
        integrated_df.loc[wav_mask, column] = 0.0

    integrated_df.loc[wav_mask, "retained_bandwidth_mean_hz"] = 8000.0

    integrated_df.insert(0, "dataset", dataset_name)
    integrated_df.insert(
        1,
        "condition",
        [
            condition_label(str(c), str(b))
            for c, b in zip(integrated_df["codec"], integrated_df["bitrate"])
        ],
    )

    # Convert WER to percentage for easier interpretation
    integrated_df["wer_percent"] = integrated_df["overall_wer"] * 100.0
    integrated_df["delta_wer_percentage_points"] = (
        integrated_df["delta_wer"] * 100.0
    )

    return integrated_df.drop(columns=["overall_wer", "delta_wer"])


def plot_integrated(integrated_df: pd.DataFrame) -> None:
    """
    Feedback point 33:
    bitrate -> {signal distortion, bandwidth, representation drift, WER}
    for MP3 and Opus, one column per LibriSpeech subset.
    """

    rows = [
        ("lsd_db_mean", "Log-spectral\ndistance (dB)"),
        ("retained_bandwidth_mean_hz", "Retained\nbandwidth (Hz)"),
        ("layer_1_drift", "Layer 1 drift\n(standardised)"),
        ("layer_12_drift", "Layer 12 drift\n(standardised)"),
        ("delta_wer_percentage_points", "ΔWER vs WAV (pp)"),
    ]

    fig, axes = plt.subplots(
        len(rows), len(DATASET_NAMES),
        figsize=(11, 2.4 * len(rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for col_index, dataset_name in enumerate(DATASET_NAMES):
        subset = integrated_df[integrated_df["dataset"] == dataset_name]

        for row_index, (column, ylabel) in enumerate(rows):
            ax = axes[row_index][col_index]

            for codec, colour in CODEC_COLOURS.items():
                points = subset[subset["codec"] == codec].sort_values(
                    "aggregate_actual_bitrate_kbps"
                )
                x = points["aggregate_actual_bitrate_kbps"]
                y = points[column]

                if column == "delta_wer_percentage_points":
                    ax.fill_between(
                        x,
                        points["ci_lower_pp"],
                        points["ci_upper_pp"],
                        color=colour,
                        alpha=0.15,
                        linewidth=0,
                    )

                ax.plot(
                    x, y,
                    marker="o",
                    markersize=5,
                    linewidth=2,
                    color=colour,
                    label=codec.upper(),
                )

            if column == "delta_wer_percentage_points":
                ax.axhline(0.0, color="#6b6b6b", linewidth=1)

            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)

            if col_index == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row_index == 0:
                ax.set_title(dataset_name)

        axes[-1][col_index].set_xlabel("Measured bitrate (kbps, log scale)")
        axes[-1][col_index].set_xticks([6, 8, 12, 16, 24, 32, 64, 128])
        axes[-1][col_index].set_xticklabels(
            ["6", "8", "12", "16", "24", "32", "64", "128"]
        )

    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        "Signal → Representation → Recognition across the bitrate sweep"
    )
    fig.tight_layout()

    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:

    integrated_df = pd.concat(
        [build_table(name) for name in DATASET_NAMES],
        ignore_index=True,
    )

    integrated_df.to_csv(OUTPUT_CSV, index=False)

    print("\nIntegrated analysis summary:\n")
    print(
        integrated_df[
            [
                "dataset",
                "condition",
                "aggregate_actual_bitrate_kbps",
                "lsd_db_mean",
                "retained_bandwidth_mean_hz",
                "layer_1_drift",
                "layer_12_drift",
                "wer_percent",
                "delta_wer_percentage_points",
            ]
        ].to_string(index=False, float_format="%.3f")
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_integrated(integrated_df)

    print(f"\nSaved table: {OUTPUT_CSV}")
    print(f"Saved figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
