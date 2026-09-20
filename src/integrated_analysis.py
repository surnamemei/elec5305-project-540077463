from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

WER_FILE = RESULTS_DIR / "test-clean_summary_results.csv"
SIGNAL_FILE = RESULTS_DIR / "signal_distortion_summary.csv"
REP_FILE = RESULTS_DIR / "representation_similarity_summary.csv"

OUTPUT_CSV = RESULTS_DIR / "integrated_analysis_summary.csv"
OUTPUT_FIGURE = FIGURES_DIR / "integrated_analysis.png"


# ---------------------------------------------------------
# Representative conditions used in representation analysis
# ---------------------------------------------------------

CONDITION_ORDER = [
    ("wav", "uncompressed"),
    ("mp3", "128k"),
    ("mp3", "16k"),
    ("opus", "16k"),
    ("opus", "8k"),
    ("opus", "6k"),
]


def condition_label(codec: str, bitrate: str) -> str:
    """Create a readable condition label."""
    if codec == "wav":
        return "WAV"
    return f"{codec.upper()} {bitrate}"


def main() -> None:
    # -----------------------------------------------------
    # Load existing results
    # -----------------------------------------------------

    wer_df = pd.read_csv(WER_FILE)
    signal_df = pd.read_csv(SIGNAL_FILE)
    rep_df = pd.read_csv(REP_FILE)

    # -----------------------------------------------------
    # Prepare representation results
    # -----------------------------------------------------

    rep_pivot = rep_df.pivot(
        index=["codec", "bitrate"],
        columns="layer",
        values="drift_mean",
    ).reset_index()

    rep_pivot = rep_pivot.rename(
        columns={
            "layer_1": "layer_1_drift",
            "layer_6": "layer_6_drift",
            "layer_12": "layer_12_drift",
        }
    )

    # -----------------------------------------------------
    # Merge WER + signal distortion + representation drift
    # -----------------------------------------------------

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
                "spectral_distortion_mean",
                "compressed_bandwidth_mean_hz",
                "bandwidth_change_mean_hz",
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

    # -----------------------------------------------------
    # WAV is the reference:
    # distortion = 0 and representation drift = 0
    # -----------------------------------------------------

    wav_mask = integrated_df["codec"] == "wav"

    integrated_df.loc[wav_mask, "spectral_distortion_mean"] = 0.0
    integrated_df.loc[wav_mask, "bandwidth_change_mean_hz"] = 0.0
    integrated_df.loc[wav_mask, "layer_1_drift"] = 0.0
    integrated_df.loc[wav_mask, "layer_6_drift"] = 0.0
    integrated_df.loc[wav_mask, "layer_12_drift"] = 0.0

    # -----------------------------------------------------
    # Keep only representative conditions
    # -----------------------------------------------------

    condition_rank = {
        condition: index
        for index, condition in enumerate(CONDITION_ORDER)
    }

    integrated_df["condition_key"] = list(
        zip(integrated_df["codec"], integrated_df["bitrate"])
    )

    integrated_df = integrated_df[
        integrated_df["condition_key"].isin(CONDITION_ORDER)
    ].copy()

    integrated_df["condition_order"] = integrated_df["condition_key"].map(
        condition_rank
    )

    integrated_df = integrated_df.sort_values(
        "condition_order"
    ).reset_index(drop=True)

    integrated_df["condition"] = integrated_df.apply(
        lambda row: condition_label(
            str(row["codec"]),
            str(row["bitrate"]),
        ),
        axis=1,
    )

    # Convert WER to percentage for easier interpretation
    integrated_df["wer_percent"] = integrated_df["overall_wer"] * 100.0
    integrated_df["delta_wer_percentage_points"] = (
        integrated_df["delta_wer"] * 100.0
    )

    # -----------------------------------------------------
    # Save integrated table
    # -----------------------------------------------------

    output_columns = [
        "condition",
        "codec",
        "bitrate",
        "aggregate_actual_bitrate_kbps",
        "spectral_distortion_mean",
        "compressed_bandwidth_mean_hz",
        "bandwidth_change_mean_hz",
        "layer_1_drift",
        "layer_6_drift",
        "layer_12_drift",
        "wer_percent",
        "delta_wer_percentage_points",
    ]

    integrated_df[output_columns].to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\nIntegrated analysis summary:\n")

    print(
        integrated_df[
            [
                "condition",
                "spectral_distortion_mean",
                "layer_1_drift",
                "layer_6_drift",
                "layer_12_drift",
                "wer_percent",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # Create integrated figure
    # -----------------------------------------------------

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    x = range(len(integrated_df))
    labels = integrated_df["condition"].tolist()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        sharex=True,
    )

    # Level 1: Signal distortion
    axes[0].plot(
        x,
        integrated_df["spectral_distortion_mean"],
        marker="o",
    )

    axes[0].set_ylabel("Log-spectral distortion")
    axes[0].set_title(
        "Signal → Representation → Recognition"
    )
    axes[0].grid(True, alpha=0.3)

    # Level 2: Representation drift
    axes[1].plot(
        x,
        integrated_df["layer_1_drift"],
        marker="o",
        label="Layer 1",
    )

    axes[1].plot(
        x,
        integrated_df["layer_6_drift"],
        marker="o",
        label="Layer 6",
    )

    axes[1].plot(
        x,
        integrated_df["layer_12_drift"],
        marker="o",
        label="Layer 12",
    )

    axes[1].set_ylabel("Representation drift")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Level 3: ASR performance
    axes[2].plot(
        x,
        integrated_df["wer_percent"],
        marker="o",
    )

    axes[2].set_ylabel("WER (%)")
    axes[2].set_xlabel("Compression condition")
    axes[2].grid(True, alpha=0.3)

    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(
        labels,
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nSaved table: {OUTPUT_CSV}")
    print(f"Saved figure: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()