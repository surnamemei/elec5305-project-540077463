"""
Which measurement best predicts ASR degradation? (feedback points 27 and 34)

Signal-level predictors (from signal_distortion_analysis.py):
    lsd_db               log-spectral distance (dB)
    spectral_distortion  mean squared log-spectral difference (dB^2)
    bandwidth_loss_hz    8000 Hz - retained bandwidth
    hf_power_loss_db     power lost in the 4-8 kHz band (dB)

Representation-level predictors (from representation_analysis.py):
    sdrift_conv, sdrift_layer_1 ... sdrift_layer_12
    (1 - cosine similarity of standardised hidden states; raw cosine is not
    comparable across layers because of layer-dependent anisotropy)

Outcome: WER change relative to the WAV version of the same utterance.

Three analyses are run for each LibriSpeech subset:

1. Condition level: correlation between the condition-mean predictor and
   the corpus-level delta WER over the 11 codec conditions plus WAV.
2. Utterance level (pooled): Spearman correlation between the predictor and
   the utterance-level delta WER over all (utterance, condition) pairs.
3. Within condition: Spearman correlation across utterances inside a single
   codec condition, i.e. whether the predictor identifies *which*
   utterances fail at a fixed bitrate. This removes the shared effect of
   the bitrate itself.

95% confidence intervals use a block bootstrap that resamples speakers
(all utterances and conditions of a speaker stay together), following the
recommendation of Bisani and Ney (ICASSP 2004) for speaker-independent ASR
evaluation. Correlations of different predictors computed on the same
resample are paired, so their difference can be tested directly. Set
BOOTSTRAP_UNIT = "dataset_index" to resample utterances instead.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUT_DIR = RESULTS_DIR / "predictor_analysis"

SIGNAL_FILE = RESULTS_DIR / "signal_distortion_results.csv"
REP_FILE = RESULTS_DIR / "representation_similarity_results.csv"

DATASET_NAMES = ["test-clean", "test-other"]

NUM_BOOTSTRAP = 2000

# Resampling unit for the bootstrap: "speaker_id" (block bootstrap over
# speakers) or "dataset_index" (utterances)
BOOTSTRAP_UNIT = "speaker_id"
RANDOM_SEED = 5305
CONFIDENCE = 0.95

NYQUIST_HZ = 8000.0

SIGNAL_PREDICTORS = [
    "lsd_db",
    "spectral_distortion",
    "bandwidth_loss_hz",
    "hf_power_loss_db",
]

LAYER_NAMES = ["conv"] + [f"layer_{i}" for i in range(1, 13)]
REP_PREDICTORS = [f"sdrift_{layer}" for layer in LAYER_NAMES]

PREDICTORS = SIGNAL_PREDICTORS + REP_PREDICTORS

# Predictors shown in tables and compared against each other
KEY_PREDICTORS = SIGNAL_PREDICTORS + [
    "sdrift_conv",
    "sdrift_layer_1",
    "sdrift_layer_6",
    "sdrift_layer_12",
]

# Reference predictor for paired differences
REFERENCE_PREDICTOR = "lsd_db"

PREDICTOR_LABELS = {
    "lsd_db": "Log-spectral distance (dB)",
    "spectral_distortion": "D_spec (dB²)",
    "bandwidth_loss_hz": "Bandwidth loss (Hz)",
    "hf_power_loss_db": "4–8 kHz power loss (dB)",
    "sdrift_conv": "Drift, conv features",
    **{f"sdrift_layer_{i}": f"Drift, layer {i}" for i in range(1, 13)},
}

CODEC_COLOURS = {"mp3": "#2a78d6", "opus": "#eb6834"}


# ---------------------------------------------------------
# Rank correlation helpers (no SciPy dependency)
# ---------------------------------------------------------

def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, identical to scipy.stats.rankdata."""
    sorter = np.argsort(values, kind="mergesort")
    inverse = np.empty(sorter.size, dtype=np.intp)
    inverse[sorter] = np.arange(sorter.size)

    ordered = values[sorter]
    first = np.r_[True, ordered[1:] != ordered[:-1]]
    dense = first.cumsum()[inverse]
    counts = np.r_[np.nonzero(first)[0], len(first)]

    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    if denominator == 0.0:
        return float("nan")
    return float((x * y).sum() / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    alpha = (1.0 - CONFIDENCE) / 2.0
    samples = samples[~np.isnan(samples)]
    return (
        float(np.quantile(samples, alpha)),
        float(np.quantile(samples, 1.0 - alpha)),
    )


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    One row per (utterance, codec condition) with the utterance-level WER
    change and every predictor.
    """

    details = pd.read_csv(RESULTS_DIR / f"{dataset_name}_experiment_details.csv")
    details["num_words"] = details["reference"].str.split().str.len()
    details["errors"] = (details["wer"] * details["num_words"]).round()

    wav = details[details["codec"] == "wav"][
        ["dataset_index", "wer", "errors"]
    ].rename(columns={"wer": "wav_wer", "errors": "wav_errors"})

    compressed = details[details["codec"] != "wav"].merge(
        wav, on="dataset_index", how="inner"
    )
    compressed["delta_wer"] = compressed["wer"] - compressed["wav_wer"]
    compressed["delta_errors"] = compressed["errors"] - compressed["wav_errors"]

    signal = pd.read_csv(SIGNAL_FILE)
    signal = signal[signal["dataset"] == dataset_name].copy()
    signal["bandwidth_loss_hz"] = NYQUIST_HZ - signal["retained_bandwidth_hz"]
    signal["hf_power_loss_db"] = -signal["hf_power_change_db"]

    rep = pd.read_csv(REP_FILE)
    rep = rep[rep["dataset"] == dataset_name]

    keys = ["dataset_index", "codec", "bitrate"]

    merged = (
        compressed[keys + ["speaker_id", "num_words", "delta_wer", "delta_errors"]]
        .merge(signal[keys + SIGNAL_PREDICTORS], on=keys, how="inner")
        .merge(rep[keys + REP_PREDICTORS], on=keys, how="inner")
    )

    expected = compressed.shape[0]
    if merged.shape[0] != expected:
        raise ValueError(
            f"{dataset_name}: merged {merged.shape[0]} rows, expected "
            f"{expected}. Re-run the signal and representation analyses "
            "with the same NUM_SAMPLES and RANDOM_SEED as the ASR experiment."
        )

    merged["condition"] = merged["codec"] + "_" + merged["bitrate"]

    return merged


# ---------------------------------------------------------
# Correlation analyses
# ---------------------------------------------------------

def condition_level_values(
    data: pd.DataFrame,
    cluster_weights: np.ndarray,
    cluster_codes: np.ndarray,
    condition_codes: np.ndarray,
    num_conditions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Condition means of each predictor and the corpus-level delta WER,
    for a given bootstrap weighting of resampling clusters (speakers or
    utterances). The WAV condition is
    appended as a point with zero distortion and zero delta WER.
    """

    weights = cluster_weights[cluster_codes]

    weight_sum = np.bincount(condition_codes, weights, num_conditions)
    predictor_means = np.stack([
        np.bincount(condition_codes, weights * data[p].to_numpy(), num_conditions)
        / weight_sum
        for p in PREDICTORS
    ], axis=1)

    delta_errors = np.bincount(
        condition_codes,
        weights * data["delta_errors"].to_numpy(),
        num_conditions,
    )
    num_words = np.bincount(
        condition_codes,
        weights * data["num_words"].to_numpy(),
        num_conditions,
    )
    corpus_delta_wer = delta_errors / num_words

    predictor_means = np.vstack([predictor_means, np.zeros(len(PREDICTORS))])
    corpus_delta_wer = np.r_[corpus_delta_wer, 0.0]

    return predictor_means, corpus_delta_wer


def run_dataset(dataset_name: str, rng: np.random.Generator):

    data = load_dataset(dataset_name)

    cluster_codes, clusters = pd.factorize(data[BOOTSTRAP_UNIT])
    condition_codes, conditions = pd.factorize(data["condition"])
    num_clusters = len(clusters)
    num_conditions = len(conditions)

    predictor_matrix = data[PREDICTORS].to_numpy()
    outcome = data["delta_wer"].to_numpy()

    # Row indices of each resampling cluster, used to build bootstrap resamples
    rows_by_cluster = [
        np.flatnonzero(cluster_codes == k) for k in range(num_clusters)
    ]

    # ----- point estimates -----------------------------------------

    ones = np.ones(num_clusters)
    cond_x, cond_y = condition_level_values(
        data, ones, cluster_codes, condition_codes, num_conditions
    )

    cond_spearman = np.array(
        [spearman(cond_x[:, j], cond_y) for j in range(len(PREDICTORS))]
    )
    cond_pearson = np.array(
        [pearson(cond_x[:, j], cond_y) for j in range(len(PREDICTORS))]
    )

    outcome_ranks = rankdata(outcome)
    pooled = np.array([
        pearson(rankdata(predictor_matrix[:, j]), outcome_ranks)
        for j in range(len(PREDICTORS))
    ])

    within = np.full((num_conditions, len(PREDICTORS)), np.nan)
    for c in range(num_conditions):
        mask = condition_codes == c
        for j in range(len(PREDICTORS)):
            within[c, j] = spearman(predictor_matrix[mask, j], outcome[mask])

    # ----- bootstrap over utterances ----------------------------------

    boot_cond_spearman = np.full((NUM_BOOTSTRAP, len(PREDICTORS)), np.nan)
    boot_cond_pearson = np.full((NUM_BOOTSTRAP, len(PREDICTORS)), np.nan)
    boot_pooled = np.full((NUM_BOOTSTRAP, len(PREDICTORS)), np.nan)
    boot_within_mean = np.full((NUM_BOOTSTRAP, len(PREDICTORS)), np.nan)

    for b in range(NUM_BOOTSTRAP):

        sampled = rng.integers(0, num_clusters, num_clusters)

        # Condition level: resampling = integer weights per cluster
        weights = np.bincount(sampled, minlength=num_clusters).astype(float)
        bx, by = condition_level_values(
            data, weights, cluster_codes, condition_codes, num_conditions
        )
        for j in range(len(PREDICTORS)):
            boot_cond_spearman[b, j] = spearman(bx[:, j], by)
            boot_cond_pearson[b, j] = pearson(bx[:, j], by)

        # Utterance level: gather the rows of every sampled cluster
        rows = np.concatenate([rows_by_cluster[k] for k in sampled])
        x = predictor_matrix[rows]
        y = outcome[rows]
        y_ranks = rankdata(y)
        for j in range(len(PREDICTORS)):
            boot_pooled[b, j] = pearson(rankdata(x[:, j]), y_ranks)

        # Within condition, averaged over conditions
        row_conditions = condition_codes[rows]
        within_b = np.full((num_conditions, len(PREDICTORS)), np.nan)
        for c in range(num_conditions):
            mask = row_conditions == c
            yc_ranks = rankdata(y[mask])
            for j in range(len(PREDICTORS)):
                within_b[c, j] = pearson(rankdata(x[mask, j]), yc_ranks)
        boot_within_mean[b] = np.nanmean(within_b, axis=0)

    # ----- tables ------------------------------------------------------

    reference_index = PREDICTORS.index(REFERENCE_PREDICTOR)

    correlation_rows = []
    for j, predictor in enumerate(PREDICTORS):
        for analysis, estimate, samples in [
            ("condition_spearman", cond_spearman[j], boot_cond_spearman[:, j]),
            ("condition_pearson", cond_pearson[j], boot_cond_pearson[:, j]),
            ("utterance_pooled_spearman", pooled[j], boot_pooled[:, j]),
            (
                "within_condition_mean_spearman",
                np.nanmean(within[:, j]),
                boot_within_mean[:, j],
            ),
        ]:
            ci_lower, ci_upper = percentile_ci(samples)
            correlation_rows.append({
                "dataset": dataset_name,
                "analysis": analysis,
                "predictor": predictor,
                "level": "signal" if predictor in SIGNAL_PREDICTORS
                else "representation",
                "correlation": float(estimate),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            })

    difference_rows = []
    for analysis, estimates, samples in [
        ("condition_pearson", cond_pearson, boot_cond_pearson),
        ("utterance_pooled_spearman", pooled, boot_pooled),
        (
            "within_condition_mean_spearman",
            np.nanmean(within, axis=0),
            boot_within_mean,
        ),
    ]:
        for j, predictor in enumerate(PREDICTORS):
            if j == reference_index:
                continue
            diff_samples = samples[:, j] - samples[:, reference_index]
            ci_lower, ci_upper = percentile_ci(diff_samples)
            difference_rows.append({
                "dataset": dataset_name,
                "analysis": analysis,
                "predictor": predictor,
                "reference_predictor": REFERENCE_PREDICTOR,
                "difference": float(estimates[j] - estimates[reference_index]),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_excludes_zero": bool(ci_lower > 0 or ci_upper < 0),
            })

    within_rows = []
    for c, condition in enumerate(conditions):
        codec, bitrate = condition.split("_")
        for j, predictor in enumerate(PREDICTORS):
            within_rows.append({
                "dataset": dataset_name,
                "codec": codec,
                "bitrate": bitrate,
                "predictor": predictor,
                "spearman": within[c, j],
            })

    condition_rows = []
    for c, condition in enumerate(list(conditions) + ["wav_uncompressed"]):
        codec, bitrate = condition.split("_")
        row = {
            "dataset": dataset_name,
            "codec": codec,
            "bitrate": bitrate,
            "corpus_delta_wer_pp": cond_y[c] * 100.0,
        }
        for j, predictor in enumerate(PREDICTORS):
            row[predictor] = cond_x[c, j]
        condition_rows.append(row)

    return (
        pd.DataFrame(correlation_rows),
        pd.DataFrame(difference_rows),
        pd.DataFrame(within_rows),
        pd.DataFrame(condition_rows),
    )


# ---------------------------------------------------------
# Figures
# ---------------------------------------------------------

def plot_condition_scatter(condition_df: pd.DataFrame, correlations: pd.DataFrame):
    """Condition-level scatter: signal predictors and layer-12 drift vs delta WER."""

    panels = ["lsd_db", "bandwidth_loss_hz", "sdrift_layer_1", "sdrift_layer_12"]

    fig, axes = plt.subplots(
        len(DATASET_NAMES), len(panels),
        figsize=(4.2 * len(panels), 4.0 * len(DATASET_NAMES)),
        squeeze=False,
    )

    for row_index, dataset_name in enumerate(DATASET_NAMES):
        subset = condition_df[condition_df["dataset"] == dataset_name]

        for col_index, predictor in enumerate(panels):
            ax = axes[row_index][col_index]

            for codec, colour in [("wav", "#6b6b6b"), *CODEC_COLOURS.items()]:
                points = subset[subset["codec"] == codec]
                ax.scatter(
                    points[predictor],
                    points["corpus_delta_wer_pp"],
                    s=40,
                    color=colour,
                    edgecolor="white",
                    linewidth=1,
                    label=codec.upper() if codec != "wav" else "WAV",
                    zorder=3,
                )
                if codec != "wav":
                    for _, point in points.iterrows():
                        ax.annotate(
                            point["bitrate"],
                            (point[predictor], point["corpus_delta_wer_pp"]),
                            textcoords="offset points",
                            xytext=(4, 3),
                            fontsize=7,
                            color="#444444",
                        )

            r = correlations[
                (correlations["dataset"] == dataset_name)
                & (correlations["analysis"] == "condition_pearson")
                & (correlations["predictor"] == predictor)
            ].iloc[0]

            ax.set_title(
                f"{PREDICTOR_LABELS[predictor]}\n"
                f"Pearson r = {r['correlation']:.2f} "
                f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]",
                fontsize=9,
            )
            ax.set_xlabel(PREDICTOR_LABELS[predictor], fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[row_index][0].set_ylabel(f"{dataset_name}\nΔWER vs WAV (pp)")

    axes[0][0].legend(fontsize=8)
    fig.suptitle("Condition level: which measurement tracks WER degradation?")
    fig.tight_layout()

    output_path = FIGURES_DIR / "predictor_condition_scatter.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_correlation_by_layer(correlations: pd.DataFrame):
    """
    Utterance-level Spearman correlation with delta WER for every
    Wav2Vec2 layer, with the signal-level predictors as reference lines.
    """

    analyses = [
        ("utterance_pooled_spearman", "Pooled over all conditions"),
        ("within_condition_mean_spearman", "Within condition (mean over 11)"),
    ]

    fig, axes = plt.subplots(
        len(DATASET_NAMES), len(analyses),
        figsize=(12, 4.3 * len(DATASET_NAMES)),
        sharex=True,
        squeeze=False,
    )

    positions = np.arange(len(LAYER_NAMES))
    ticks = ["conv"] + [str(i) for i in range(1, 13)]

    signal_styles = {
        "lsd_db": ("#1baf7a", "-"),
        "bandwidth_loss_hz": ("#e87ba4", "--"),
    }

    for row_index, dataset_name in enumerate(DATASET_NAMES):
        for col_index, (analysis, title) in enumerate(analyses):
            ax = axes[row_index][col_index]

            subset = correlations[
                (correlations["dataset"] == dataset_name)
                & (correlations["analysis"] == analysis)
            ].set_index("predictor")

            layer_values = subset.loc[REP_PREDICTORS]

            ax.fill_between(
                positions,
                layer_values["ci_lower"],
                layer_values["ci_upper"],
                color="#2a78d6",
                alpha=0.18,
                linewidth=0,
            )
            ax.plot(
                positions,
                layer_values["correlation"],
                marker="o",
                markersize=5,
                linewidth=2,
                color="#2a78d6",
                label="Wav2Vec2 representation drift",
            )

            for predictor, (colour, style) in signal_styles.items():
                value = subset.loc[predictor]
                ax.axhspan(
                    value["ci_lower"], value["ci_upper"],
                    color=colour, alpha=0.12, linewidth=0,
                )
                ax.axhline(
                    value["correlation"],
                    color=colour,
                    linestyle=style,
                    linewidth=2,
                    label=PREDICTOR_LABELS[predictor],
                )

            ax.set_title(f"{dataset_name}: {title}", fontsize=10)
            ax.grid(True, alpha=0.3)

        axes[row_index][0].set_ylabel("Spearman ρ with utterance ΔWER")

    for ax in axes[-1]:
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Wav2Vec2 layer (conv = feature-encoder output)")

    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Utterance level: correlation of each measurement with ΔWER "
        "(shaded: 95% bootstrap CI)"
    )
    fig.tight_layout()

    output_path = FIGURES_DIR / "predictor_correlation_by_layer.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    outputs = [run_dataset(name, rng) for name in DATASET_NAMES]

    correlations = pd.concat([o[0] for o in outputs], ignore_index=True)
    differences = pd.concat([o[1] for o in outputs], ignore_index=True)
    within = pd.concat([o[2] for o in outputs], ignore_index=True)
    condition_df = pd.concat([o[3] for o in outputs], ignore_index=True)

    correlations.to_csv(OUTPUT_DIR / "predictor_correlations.csv", index=False)
    differences.to_csv(
        OUTPUT_DIR / "predictor_differences_vs_lsd.csv", index=False
    )
    within.to_csv(
        OUTPUT_DIR / "within_condition_correlations.csv", index=False
    )
    condition_df.to_csv(
        OUTPUT_DIR / "condition_level_predictors.csv", index=False
    )

    scatter_path = plot_condition_scatter(condition_df, correlations)
    layer_path = plot_correlation_by_layer(correlations)

    table = correlations[correlations["predictor"].isin(KEY_PREDICTORS)].copy()
    table["value"] = table.apply(
        lambda r: f"{r['correlation']:+.2f} [{r['ci_lower']:+.2f}, {r['ci_upper']:+.2f}]",
        axis=1,
    )
    print("\nCorrelation with ΔWER (95% bootstrap CI):\n")
    for dataset_name in DATASET_NAMES:
        print(dataset_name)
        print(
            table[table["dataset"] == dataset_name]
            .pivot(index="predictor", columns="analysis", values="value")
            .loc[KEY_PREDICTORS]
            .to_string()
        )
        print()

    print(f"Saved tables to: {OUTPUT_DIR}")
    print(f"Saved figure: {scatter_path}")
    print(f"Saved figure: {layer_path}")


if __name__ == "__main__":
    main()
