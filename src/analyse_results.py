import os
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
CLEAN_PATH = "results/test-clean_summary_results.csv"
OTHER_PATH = "results/test-other_summary_results.csv"
FIGURE_DIR = "results/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# Load results
# --------------------------------------------------
clean = pd.read_csv(CLEAN_PATH)
other = pd.read_csv(OTHER_PATH)

clean["dataset"] = "test-clean"
other["dataset"] = "test-other"

# Recalculate delta WER in case it is not stored
clean_baseline = clean.loc[clean["codec"] == "wav", "overall_wer"].iloc[0]
other_baseline = other.loc[other["codec"] == "wav", "overall_wer"].iloc[0]

clean["delta_wer"] = clean["overall_wer"] - clean_baseline
other["delta_wer"] = other["overall_wer"] - other_baseline

# Convert WER to percentage
clean["wer_percent"] = clean["overall_wer"] * 100
other["wer_percent"] = other["overall_wer"] * 100

clean["delta_wer_percent"] = clean["delta_wer"] * 100
other["delta_wer_percent"] = other["delta_wer"] * 100

# Convert bitrate strings such as "128k" -> 128
def bitrate_to_number(value):
    if value == "uncompressed":
        return None
    return int(str(value).replace("k", ""))

clean["bitrate_kbps"] = clean["bitrate"].apply(bitrate_to_number)
other["bitrate_kbps"] = other["bitrate"].apply(bitrate_to_number)


# --------------------------------------------------
# Helper: extract one codec
# --------------------------------------------------
def codec_data(df, codec):
    return (
        df[df["codec"] == codec]
        .sort_values("bitrate_kbps")
        .copy()
    )


# ==================================================
# Figure 1: WER vs Bitrate
# ==================================================
plt.figure(figsize=(8, 5))

for dataset_name, df in [
    ("test-clean", clean),
    ("test-other", other)
]:
    for codec in ["mp3", "opus"]:
        data = codec_data(df, codec)

        plt.plot(
            data["bitrate_kbps"],
            data["wer_percent"],
            marker="o",
            label=f"{dataset_name} - {codec.upper()}"
        )

plt.xlabel("Bitrate (kbps)")
plt.ylabel("Word Error Rate (%)")
plt.title("ASR Word Error Rate vs Audio Bitrate")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "wer_vs_bitrate.png")
plt.savefig(path, dpi=300)
plt.close()


# ==================================================
# Figure 2: Delta WER vs Bitrate
# ==================================================
plt.figure(figsize=(8, 5))

for dataset_name, df in [
    ("test-clean", clean),
    ("test-other", other)
]:
    for codec in ["mp3", "opus"]:
        data = codec_data(df, codec)

        plt.plot(
            data["bitrate_kbps"],
            data["delta_wer_percent"],
            marker="o",
            label=f"{dataset_name} - {codec.upper()}"
        )

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Bitrate (kbps)")
plt.ylabel("WER Change from WAV Baseline (percentage points)")
plt.title("ASR Degradation Relative to Uncompressed WAV")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "delta_wer_vs_bitrate.png")
plt.savefig(path, dpi=300)
plt.close()


# ==================================================
# Figure 3: WER vs Compression Ratio
# ==================================================
plt.figure(figsize=(8, 5))

for dataset_name, df in [
    ("test-clean", clean),
    ("test-other", other)
]:
    for codec in ["mp3", "opus"]:
        data = df[df["codec"] == codec].sort_values(
            "average_compression_ratio"
        )

        plt.plot(
            data["average_compression_ratio"],
            data["wer_percent"],
            marker="o",
            label=f"{dataset_name} - {codec.upper()}"
        )

# WAV baselines
plt.scatter(
    [1],
    [clean_baseline * 100],
    marker="x",
    s=80,
    label="test-clean WAV"
)

plt.scatter(
    [1],
    [other_baseline * 100],
    marker="x",
    s=80,
    label="test-other WAV"
)

plt.xlabel("Average Compression Ratio")
plt.ylabel("Word Error Rate (%)")
plt.title("ASR Performance vs Compression Ratio")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(FIGURE_DIR, "wer_vs_compression_ratio.png")
plt.savefig(path, dpi=300)
plt.close()


# ==================================================
# Figure 4: Clean vs Other comparison
# ==================================================
comparison = clean[
    ["codec", "bitrate", "wer_percent"]
].merge(
    other[
        ["codec", "bitrate", "wer_percent"]
    ],
    on=["codec", "bitrate"],
    suffixes=("_clean", "_other")
)

labels = (
    comparison["codec"].astype(str).str.upper()
    + " "
    + comparison["bitrate"].astype(str)
).tolist()

x = range(len(comparison))

plt.figure(figsize=(11, 5))

plt.plot(
    x,
    comparison["wer_percent_clean"],
    marker="o",
    label="test-clean"
)

plt.plot(
    x,
    comparison["wer_percent_other"],
    marker="o",
    label="test-other"
)

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Word Error Rate (%)")
plt.xlabel("Compression Condition")
plt.title("ASR Robustness: test-clean vs test-other")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

path = os.path.join(
    FIGURE_DIR,
    "clean_vs_other.png"
)

plt.savefig(path, dpi=300)
plt.close()


# --------------------------------------------------
# Print useful summary
# --------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

print(
    f"test-clean WAV baseline: "
    f"{clean_baseline * 100:.2f}% WER"
)

print(
    f"test-other WAV baseline: "
    f"{other_baseline * 100:.2f}% WER"
)

print("\nTEST-CLEAN")
print(
    clean[
        [
            "codec",
            "bitrate",
            "wer_percent",
            "delta_wer_percent",
            "average_compression_ratio"
        ]
    ].to_string(index=False)
)

print("\nTEST-OTHER")
print(
    other[
        [
            "codec",
            "bitrate",
            "wer_percent",
            "delta_wer_percent",
            "average_compression_ratio"
        ]
    ].to_string(index=False)
)

print("\nFigures saved to:")
print(FIGURE_DIR)