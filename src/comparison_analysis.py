import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
CLEAN_PATH = "results/test-clean_summary_results.csv"
OTHER_PATH = "results/test-other_summary_results.csv"

OUTPUT_DIR = "results/comparison_analysis"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

MATCHED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "matched_bitrate_codec_comparison.csv"
)

DIFFICULTY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "difficulty_degradation_gap.csv"
)

COMMON_BITRATES = ["64k", "32k", "16k"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_summary(path, dataset_name):
    df = pd.read_csv(path)
    df["bitrate"] = df["bitrate"].astype(str)
    df["dataset"] = dataset_name
    return df


def bitrate_to_kbps(value):
    return int(str(value).replace("k", ""))


# --------------------------------------------------
# Load existing experiment summaries
# --------------------------------------------------
clean = load_summary(CLEAN_PATH, "test-clean")
other = load_summary(OTHER_PATH, "test-other")

all_results = pd.concat(
    [clean, other],
    ignore_index=True
)

os.makedirs(FIGURE_DIR, exist_ok=True)


# --------------------------------------------------
# 1. Matched-bitrate MP3 vs Opus comparison
# --------------------------------------------------
matched = all_results[
    all_results["codec"].isin(["mp3", "opus"])
    & all_results["bitrate"].isin(COMMON_BITRATES)
].copy()

matched["bitrate_kbps"] = matched[
    "bitrate"
].map(bitrate_to_kbps)

matched["delta_wer_pp"] = (
    matched["delta_wer"] * 100
)

matched["overall_wer_percent"] = (
    matched["overall_wer"] * 100
)

matched_wide = matched.pivot_table(
    index=[
        "dataset",
        "bitrate",
        "bitrate_kbps"
    ],
    columns="codec",
    values=[
        "delta_wer_pp",
        "overall_wer_percent",
        "average_compression_ratio"
    ]
).reset_index()

matched_wide.columns = [
    "_".join(
        [
            str(part)
            for part in col
            if str(part)
        ]
    )
    if isinstance(col, tuple)
    else str(col)
    for col in matched_wide.columns
]

matched_wide[
    "opus_minus_mp3_delta_wer_pp"
] = (
    matched_wide["delta_wer_pp_opus"]
    - matched_wide["delta_wer_pp_mp3"]
)

matched_wide = matched_wide.sort_values(
    ["dataset", "bitrate_kbps"],
    ascending=[True, False]
)

matched_wide.to_csv(
    MATCHED_OUTPUT,
    index=False
)


# --------------------------------------------------
# Matched-bitrate figures
# --------------------------------------------------
for dataset_name, group in matched.groupby(
    "dataset"
):
    plt.figure(figsize=(7, 4.5))

    for codec_name, codec_group in group.groupby(
        "codec"
    ):
        codec_group = codec_group.sort_values(
            "bitrate_kbps"
        )

        plt.plot(
            codec_group["bitrate_kbps"],
            codec_group["delta_wer_pp"],
            marker="o",
            label=str(codec_name).upper()
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Bitrate (kbps)")
    plt.ylabel(
        "ΔWER relative to WAV "
        "(percentage points)"
    )
    plt.title(
        "Matched-bitrate codec comparison: "
        f"{dataset_name}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURE_DIR,
        (
            f"{dataset_name}_"
            "matched_bitrate_codec_comparison.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# --------------------------------------------------
# 2. test-other vs test-clean degradation gap
# --------------------------------------------------
compressed = all_results[
    all_results["codec"] != "wav"
].copy()

compressed["delta_wer_pp"] = (
    compressed["delta_wer"] * 100
)

compressed["bitrate_kbps"] = compressed[
    "bitrate"
].map(bitrate_to_kbps)

clean_delta = compressed[
    compressed["dataset"] == "test-clean"
][
    [
        "codec",
        "bitrate",
        "bitrate_kbps",
        "delta_wer_pp"
    ]
].rename(
    columns={
        "delta_wer_pp": "clean_delta_wer_pp"
    }
)

other_delta = compressed[
    compressed["dataset"] == "test-other"
][
    [
        "codec",
        "bitrate",
        "bitrate_kbps",
        "delta_wer_pp"
    ]
].rename(
    columns={
        "delta_wer_pp": "other_delta_wer_pp"
    }
)

difficulty_gap = clean_delta.merge(
    other_delta,
    on=[
        "codec",
        "bitrate",
        "bitrate_kbps"
    ],
    how="inner"
)

difficulty_gap[
    "other_minus_clean_delta_wer_pp"
] = (
    difficulty_gap["other_delta_wer_pp"]
    - difficulty_gap["clean_delta_wer_pp"]
)

difficulty_gap = difficulty_gap.sort_values(
    ["codec", "bitrate_kbps"],
    ascending=[True, False]
)

difficulty_gap.to_csv(
    DIFFICULTY_OUTPUT,
    index=False
)


# --------------------------------------------------
# Difficulty-gap figure
# --------------------------------------------------
plt.figure(figsize=(7, 4.5))

for codec_name, group in difficulty_gap.groupby(
    "codec"
):
    group = group.sort_values(
        "bitrate_kbps"
    )

    plt.plot(
        group["bitrate_kbps"],
        group[
            "other_minus_clean_delta_wer_pp"
        ],
        marker="o",
        label=str(codec_name).upper()
    )

plt.axhline(0, linewidth=1)

plt.xlabel("Bitrate (kbps)")
plt.ylabel(
    "ΔWER(test-other) - "
    "ΔWER(test-clean) (pp)"
)

plt.title(
    "Additional compression vulnerability "
    "of test-other"
)

plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "difficulty_degradation_gap.png"
    ),
    dpi=200
)

plt.close()


# --------------------------------------------------
# Console summary
# --------------------------------------------------
print("=" * 72)
print("COMPARISON ANALYSIS COMPLETE")
print("=" * 72)

print(f"Saved: {MATCHED_OUTPUT}")
print(f"Saved: {DIFFICULTY_OUTPUT}")
print(f"Figures: {FIGURE_DIR}")

print(
    "\nMatched-bitrate "
    "MP3 vs Opus comparison:"
)

print(
    matched_wide[
        [
            "dataset",
            "bitrate",
            "delta_wer_pp_mp3",
            "delta_wer_pp_opus",
            "opus_minus_mp3_delta_wer_pp"
        ]
    ].to_string(index=False)
)

print(
    "\nDifficulty-associated "
    "degradation gap:"
)

print(
    difficulty_gap[
        [
            "codec",
            "bitrate",
            "clean_delta_wer_pp",
            "other_delta_wer_pp",
            "other_minus_clean_delta_wer_pp"
        ]
    ].to_string(index=False)
)