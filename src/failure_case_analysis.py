from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
ERROR_DIR = RESULTS_DIR / "error_analysis"

INPUT_FILE = ERROR_DIR / "top_new_errors.csv"
OUTPUT_FILE = ERROR_DIR / "selected_failure_cases.csv"
OUTPUT_TEXT = ERROR_DIR / "selected_failure_cases.txt"


# We focus on severe conditions that showed clear WER degradation.
TARGET_CONDITIONS = [
    ("test-clean", "mp3", "16k"),
    ("test-clean", "opus", "8k"),
    ("test-clean", "opus", "6k"),
    ("test-other", "opus", "8k"),
    ("test-other", "opus", "6k"),
]

# Number of representative cases kept from each condition.
CASES_PER_CONDITION = 2


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "dataset",
        "codec",
        "bitrate",
        "dataset_index",
        "reference",
        "wav_prediction",
        "compressed_prediction",
        "wav_wer",
        "compressed_wer",
        "delta_sample_wer",
        "delta_substitutions",
        "delta_deletions",
        "delta_insertions",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    selected_groups: list[pd.DataFrame] = []

    for dataset, codec, bitrate in TARGET_CONDITIONS:
        subset = df[
            (df["dataset"] == dataset)
            & (df["codec"] == codec)
            & (df["bitrate"] == bitrate)
        ].copy()

        if subset.empty:
            print(
                f"Warning: no cases found for "
                f"{dataset} {codec} {bitrate}"
            )
            continue

        # Prefer examples where WAV was good but compression clearly worsened ASR.
        subset = subset.sort_values(
            by=[
                "delta_sample_wer",
                "delta_substitutions",
                "delta_deletions",
                "delta_insertions",
            ],
            ascending=False,
        )

        selected_groups.append(
            subset.head(CASES_PER_CONDITION)
        )

    if not selected_groups:
        raise RuntimeError(
            "No matching failure cases were found."
        )

    selected_df = pd.concat(
        selected_groups,
        ignore_index=True,
    )

    output_columns = [
        "dataset",
        "codec",
        "bitrate",
        "dataset_index",
        "reference",
        "wav_prediction",
        "compressed_prediction",
        "wav_wer",
        "compressed_wer",
        "delta_sample_wer",
        "delta_substitutions",
        "delta_deletions",
        "delta_insertions",
    ]

    selected_df[output_columns].to_csv(
        OUTPUT_FILE,
        index=False,
    )

    lines: list[str] = []

    lines.append("Representative Compression Failure Cases")
    lines.append("=" * 50)
    lines.append("")

    for case_number, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1,
    ):
        dataset = str(row["dataset"])
        codec = str(row["codec"]).upper()
        bitrate = str(row["bitrate"])

        lines.append(
            f"Case {case_number}: "
            f"{dataset} | {codec} {bitrate}"
        )

        lines.append(
            f"Dataset index: {row['dataset_index']}"
        )

        lines.append(
            f"Reference: {row['reference']}"
        )

        lines.append(
            f"WAV prediction: {row['wav_prediction']}"
        )

        lines.append(
            f"Compressed prediction: "
            f"{row['compressed_prediction']}"
        )

        lines.append(
            f"WAV WER: "
            f"{float(row['wav_wer']) * 100:.2f}%"
        )

        lines.append(
            f"Compressed WER: "
            f"{float(row['compressed_wer']) * 100:.2f}%"
        )

        lines.append(
            f"ΔWER: "
            f"{float(row['delta_sample_wer']) * 100:.2f} pp"
        )

        lines.append(
            "New errors: "
            f"S={int(row['delta_substitutions'])}, "
            f"D={int(row['delta_deletions'])}, "
            f"I={int(row['delta_insertions'])}"
        )

        lines.append("-" * 50)
        lines.append("")

    OUTPUT_TEXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nSelected failure cases:\n")

    print(
        selected_df[
            [
                "dataset",
                "codec",
                "bitrate",
                "dataset_index",
                "delta_sample_wer",
                "delta_substitutions",
                "delta_deletions",
                "delta_insertions",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved CSV: {OUTPUT_FILE}")
    print(f"Saved text summary: {OUTPUT_TEXT}")


if __name__ == "__main__":
    main()