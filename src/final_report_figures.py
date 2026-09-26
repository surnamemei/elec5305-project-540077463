"""
Final figure set for the report.

    fig1_integrated.png     <- drawn here from results/integrated_analysis_summary.csv
                               with speaker-level CIs from table_main_results.csv
                               (same data as results/figures/integrated_analysis.png,
                               wide 2 x 5 layout for the page)
    fig2_drift_by_layer.png <- copy of results/figures/representation_drift_by_layer.png
                               (src/representation_analysis.py)
    fig3_predictors.png     <- drawn here from results/predictor_analysis/
    fig4_case_study.png     <- written by src/final_case_studies.py

Figure 3 separates the two predictor questions:
    (a, b) across conditions: corpus ΔWER against log-spectral distance and
           against standardised layer-12 drift (Pearson r, 95% bootstrap CI);
    (c)    within a condition: mean Spearman correlation between each
           Wav2Vec2 layer's drift and utterance ΔWER, with signal-level
           measures as reference lines.

Run from the repository root after predictor_analysis.py:
    python src/final_report_figures.py
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR = RESULTS_DIR / "final_report"
PREDICTOR_DIR = RESULTS_DIR / "predictor_analysis"

CODEC_COLOURS = {"wav": "#6b6b6b", "mp3": "#2a78d6", "opus": "#eb6834"}
DATASET_MARKERS = {"test-clean": "o", "test-other": "^"}
LAYER_NAMES = ["conv"] + [f"layer_{i}" for i in range(1, 13)]


def pearson_label(correlations: pd.DataFrame, dataset: str, predictor: str) -> str:
    row = correlations[
        (correlations["dataset"] == dataset)
        & (correlations["analysis"] == "condition_pearson")
        & (correlations["predictor"] == predictor)
    ].iloc[0]
    return (
        f"{dataset}: r = {row['correlation']:.2f} "
        f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
    )


def draw_fig3() -> Path:

    conditions = pd.read_csv(PREDICTOR_DIR / "condition_level_predictors.csv")
    correlations = pd.read_csv(PREDICTOR_DIR / "predictor_correlations.csv")

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4.4),
        gridspec_kw={"width_ratios": [1, 1, 1.25]},
    )

    for ax, predictor, xlabel, title in [
        (axes[0], "lsd_db", "Log-spectral distance (dB)",
         "(a) Across conditions: signal distortion"),
        (axes[1], "sdrift_layer_12", "Standardised layer-12 drift (1 − cos)",
         "(b) Across conditions: late-layer drift"),
    ]:
        for dataset, marker in DATASET_MARKERS.items():
            subset = conditions[conditions["dataset"] == dataset]
            for codec, colour in CODEC_COLOURS.items():
                points = subset[subset["codec"] == codec]
                ax.scatter(
                    points[predictor],
                    points["corpus_delta_wer_pp"],
                    marker=marker,
                    s=46,
                    color=colour,
                    edgecolor="white",
                    linewidth=1,
                    zorder=3,
                )

        text = "\n".join(
            pearson_label(correlations, dataset, predictor)
            for dataset in DATASET_MARKERS
        )
        ax.text(
            0.03, 0.97, "Pearson, 95% CI\n" + text,
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("ΔWER vs WAV (percentage points)")

    # Annotate the two conditions discussed in the text
    other = conditions[conditions["dataset"] == "test-other"]
    for bitrate in ["8k", "6k"]:
        point = other[(other["codec"] == "opus") & (other["bitrate"] == bitrate)].iloc[0]
        for ax, predictor in [(axes[0], "lsd_db"), (axes[1], "sdrift_layer_12")]:
            ax.annotate(
                f"Opus {bitrate}",
                (point[predictor], point["corpus_delta_wer_pp"]),
                textcoords="offset points", xytext=(-38, -3), fontsize=7,
                color="#444444",
            )

    # Legend: codec colours and subset markers
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=colour,
                   label=codec.upper() if codec != "opus" else "Opus")
        for codec, colour in CODEC_COLOURS.items()
    ] + [
        plt.Line2D([], [], marker=marker, linestyle="", color="#444444",
                   markerfacecolor="white", label=dataset)
        for dataset, marker in DATASET_MARKERS.items()
    ]
    axes[1].legend(handles=handles, fontsize=7.5, loc="lower right")

    # (c) within-condition correlation by layer
    ax = axes[2]
    positions = np.arange(len(LAYER_NAMES))
    for dataset, style in [("test-clean", "-"), ("test-other", "--")]:
        subset = correlations[
            (correlations["dataset"] == dataset)
            & (correlations["analysis"] == "within_condition_mean_spearman")
        ].set_index("predictor")

        values = subset.loc[[f"sdrift_{l}" for l in LAYER_NAMES]]
        ax.fill_between(
            positions, values["ci_lower"], values["ci_upper"],
            color="#2a78d6", alpha=0.12, linewidth=0,
        )
        ax.plot(
            positions, values["correlation"],
            color="#2a78d6", linestyle=style, linewidth=2, marker="o",
            markersize=4, label=f"Layer drift, {dataset}",
        )

        lsd = subset.loc["lsd_db"]
        ax.axhline(
            lsd["correlation"], color="#1baf7a", linestyle=style, linewidth=1.5,
            label=f"LSD, {dataset}",
        )

    ax.axhline(0.0, color="#6b6b6b", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(["conv"] + [str(i) for i in range(1, 13)], fontsize=8)
    ax.set_xlabel("Wav2Vec2 layer (conv = feature-encoder output)")
    ax.set_ylabel("Mean within-condition Spearman ρ")
    ax.set_title(
        "(c) Within a condition: which utterances degrade", fontsize=10, loc="left"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "fig3_predictors.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def draw_fig1() -> Path:
    """Rows: subsets; columns: signal, representation and task measures."""

    table = pd.read_csv(RESULTS_DIR / "integrated_analysis_summary.csv")

    # Use the speaker-level block-bootstrap CIs of the authoritative table
    ci = pd.read_csv(OUTPUT_DIR / "table_main_results.csv")[
        ["dataset", "codec", "bitrate", "ci_lower_pp", "ci_upper_pp"]
    ]
    table = table.drop(columns=["ci_lower_pp", "ci_upper_pp"]).merge(
        ci, on=["dataset", "codec", "bitrate"], how="left"
    )
    table[["ci_lower_pp", "ci_upper_pp"]] = table[["ci_lower_pp", "ci_upper_pp"]].fillna(0.0)

    columns = [
        ("lsd_db_mean", "LSD (dB)", "Log-spectral distance"),
        ("retained_bandwidth_mean_hz", "Retained bandwidth (kHz)", "Retained bandwidth"),
        ("layer_1_drift", "Drift (1 − cos)", "Layer-1 drift"),
        ("layer_12_drift", "Drift (1 − cos)", "Layer-12 drift"),
        ("delta_wer_percentage_points", "ΔWER (pp)", "ΔWER vs WAV"),
    ]

    fig, axes = plt.subplots(
        2, len(columns), figsize=(15, 5.6), sharex=True, sharey="col"
    )

    for row_index, dataset in enumerate(["test-clean", "test-other"]):
        subset = table[table["dataset"] == dataset]
        for col_index, (column, ylabel, title) in enumerate(columns):
            ax = axes[row_index][col_index]
            for codec in ["mp3", "opus"]:
                points = subset[subset["codec"] == codec].sort_values(
                    "aggregate_actual_bitrate_kbps"
                )
                x = points["aggregate_actual_bitrate_kbps"]
                y = points[column]
                if column == "retained_bandwidth_mean_hz":
                    y = y / 1000.0
                if column == "delta_wer_percentage_points":
                    ax.fill_between(
                        x, points["ci_lower_pp"], points["ci_upper_pp"],
                        color=CODEC_COLOURS[codec], alpha=0.15, linewidth=0,
                    )
                    ax.axhline(0.0, color="#6b6b6b", linewidth=0.8)
                ax.plot(
                    x, y, marker="o", markersize=4, linewidth=2,
                    color=CODEC_COLOURS[codec],
                    label="MP3" if codec == "mp3" else "Opus",
                )
            ax.set_xscale("log")
            ax.set_xticks([6, 8, 12, 16, 24, 32, 64, 128])
            ax.set_xticklabels(["6", "8", "12", "16", "24", "32", "64", "128"], fontsize=7)
            ax.minorticks_off()
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel, fontsize=9)
            if row_index == 0:
                ax.set_title(title, fontsize=10)
            else:
                ax.set_xlabel("Measured bitrate (kbps, log)", fontsize=9)
        axes[row_index][0].annotate(
            dataset, xy=(-0.42, 0.5), xycoords="axes fraction",
            rotation=90, va="center", ha="center", fontsize=11, fontweight="bold",
        )

    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "fig1_integrated.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Saved", draw_fig1())
    shutil.copyfile(
        FIGURES_DIR / "representation_drift_by_layer.png",
        OUTPUT_DIR / "fig2_drift_by_layer.png",
    )

    print("Saved", draw_fig3())
    print("Copied fig2_drift_by_layer.png")


if __name__ == "__main__":
    main()
