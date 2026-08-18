import os
from typing import Optional

import pandas as pd


# --------------------------------------------------
# Settings
# --------------------------------------------------
DATASETS: dict[str, str] = {
    "test-clean": "results/test-clean_experiment_details.csv",
    "test-other": "results/test-other_experiment_details.csv",
}

OUTPUT_DIR = "results/error_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Word-level edit distance
# Returns:
# substitutions, deletions, insertions
# --------------------------------------------------
def word_error_counts(
    reference: str,
    hypothesis: str
) -> tuple[int, int, int]:

    ref = str(reference).split()
    hyp = str(hypothesis).split()

    n = len(ref)
    m = len(hyp)

    # dp[i][j] = minimum edit distance
    dp: list[list[int]] = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    # Operation used to reach each cell
    operation: list[list[Optional[str]]] = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    # Initial deletion path
    for i in range(1, n + 1):
        dp[i][0] = i
        operation[i][0] = "D"

    # Initial insertion path
    for j in range(1, m + 1):
        dp[0][j] = j
        operation[0][j] = "I"

    # Dynamic programming
    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                operation[i][j] = "M"

            else:
                substitution = (
                    dp[i - 1][j - 1] + 1
                )

                deletion = (
                    dp[i - 1][j] + 1
                )

                insertion = (
                    dp[i][j - 1] + 1
                )

                best = min(
                    substitution,
                    deletion,
                    insertion
                )

                dp[i][j] = best

                if best == substitution:
                    operation[i][j] = "S"

                elif best == deletion:
                    operation[i][j] = "D"

                else:
                    operation[i][j] = "I"

    # --------------------------------------------------
    # Backtrack
    # --------------------------------------------------
    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:

        op = operation[i][j]

        if op == "M":
            i -= 1
            j -= 1

        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1

        elif op == "D":
            deletions += 1
            i -= 1

        elif op == "I":
            insertions += 1
            j -= 1

        else:
            break

    return (
        substitutions,
        deletions,
        insertions
    )


# --------------------------------------------------
# Analyse one dataset
# --------------------------------------------------
def analyse_dataset(
    dataset_name: str,
    path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print("\n" + "=" * 80)
    print(f"ERROR ANALYSIS: {dataset_name}")
    print("=" * 80)

    df = pd.read_csv(path)

    df["codec"] = df["codec"].astype(str)
    df["bitrate"] = df["bitrate"].astype(str)
    df["reference"] = df["reference"].astype(str)
    df["prediction"] = df["prediction"].astype(str)

    # --------------------------------------------------
    # WAV baseline
    # --------------------------------------------------
    wav = df[
        df["codec"] == "wav"
    ][
        [
            "dataset_index",
            "speaker_id",
            "chapter_id",
            "utterance_id",
            "reference",
            "prediction",
            "wer",
        ]
    ].copy()

    wav = wav.rename(
        columns={
            "prediction": "wav_prediction",
            "wer": "wav_wer",
        }
    )

    # --------------------------------------------------
    # Compression conditions
    # --------------------------------------------------
    conditions = (
        df[
            df["codec"] != "wav"
        ][
            ["codec", "bitrate"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    all_details: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for _, condition in conditions.iterrows():

        codec = str(
            condition["codec"]
        )

        bitrate = str(
            condition["bitrate"]
        )

        compressed = df[
            (df["codec"] == codec)
            &
            (df["bitrate"] == bitrate)
        ][
            [
                "dataset_index",
                "prediction",
                "wer",
                "compression_ratio",
            ]
        ].copy()

        compressed = compressed.rename(
            columns={
                "prediction":
                    "compressed_prediction",

                "wer":
                    "compressed_wer",
            }
        )

        paired = wav.merge(
            compressed,
            on="dataset_index",
            how="inner"
        )

        paired["dataset"] = dataset_name
        paired["codec"] = codec
        paired["bitrate"] = bitrate

        paired[
            "delta_sample_wer"
        ] = (
            paired["compressed_wer"]
            -
            paired["wav_wer"]
        )

        wav_substitutions: list[int] = []
        wav_deletions: list[int] = []
        wav_insertions: list[int] = []

        compressed_substitutions: list[int] = []
        compressed_deletions: list[int] = []
        compressed_insertions: list[int] = []

        for _, row in paired.iterrows():

            reference = str(
                row["reference"]
            )

            wav_prediction = str(
                row["wav_prediction"]
            )

            compressed_prediction = str(
                row["compressed_prediction"]
            )

            # WAV baseline errors
            wav_s, wav_d, wav_i = word_error_counts(
                reference,
                wav_prediction
            )

            # Compressed-condition errors
            comp_s, comp_d, comp_i = word_error_counts(
                reference,
                compressed_prediction
            )

            wav_substitutions.append(wav_s)
            wav_deletions.append(wav_d)
            wav_insertions.append(wav_i)

            compressed_substitutions.append(comp_s)
            compressed_deletions.append(comp_d)
            compressed_insertions.append(comp_i)

        paired["wav_substitutions"] = wav_substitutions
        paired["wav_deletions"] = wav_deletions
        paired["wav_insertions"] = wav_insertions

        paired["compressed_substitutions"] = compressed_substitutions
        paired["compressed_deletions"] = compressed_deletions
        paired["compressed_insertions"] = compressed_insertions

        paired["delta_substitutions"] = (
            paired["compressed_substitutions"]
            - paired["wav_substitutions"]
        )

        paired["delta_deletions"] = (
            paired["compressed_deletions"]
            - paired["wav_deletions"]
        )

        paired["delta_insertions"] = (
            paired["compressed_insertions"]
            - paired["wav_insertions"]
        )
        # --------------------------------------------------
        # Behaviour relative to WAV baseline
        # --------------------------------------------------
        paired[
            "wav_correct"
        ] = (
            paired["wav_wer"] == 0
        )

        paired[
            "compressed_correct"
        ] = (
            paired["compressed_wer"] == 0
        )

        paired[
            "new_error"
        ] = (
            paired["wav_correct"]
            &
            (~paired["compressed_correct"])
        )

        paired[
            "recovered"
        ] = (
            (~paired["wav_correct"])
            &
            paired["compressed_correct"]
        )

        paired[
            "worsened"
        ] = (
            paired[
                "delta_sample_wer"
            ] > 0
        )

        paired[
            "improved"
        ] = (
            paired[
                "delta_sample_wer"
            ] < 0
        )

        paired[
            "unchanged"
        ] = (
            paired[
                "delta_sample_wer"
            ] == 0
        )

        all_details.append(
            paired
        )

        summary_rows.append({
            "dataset":
                dataset_name,

            "codec":
                codec,

            "bitrate":
                bitrate,

            "num_samples":
                len(paired),

            "new_errors":
                int(
                    paired[
                        "new_error"
                    ].sum()
                ),

            "recovered_errors":
                int(
                    paired[
                        "recovered"
                    ].sum()
                ),

            "worsened_samples":
                int(
                    paired[
                        "worsened"
                    ].sum()
                ),

            "improved_samples":
                int(
                    paired[
                        "improved"
                    ].sum()
                ),

            "unchanged_samples":
                int(
                    paired[
                        "unchanged"
                    ].sum()
                ),

            "wav_substitutions":
                int(
                    paired[
                        "wav_substitutions"
                    ].sum()
                ),

            "compressed_substitutions":
                int(
                    paired[
                        "compressed_substitutions"
                    ].sum()
                ),

            "delta_substitutions":
                int(
                    paired[
                        "delta_substitutions"
                    ].sum()
                ),

            "wav_deletions":
                int(
                    paired[
                        "wav_deletions"
                    ].sum()
                ),

            "compressed_deletions":
                int(
                    paired[
                        "compressed_deletions"
                    ].sum()
                ),

            "delta_deletions":
                int(
                    paired[
                        "delta_deletions"
                    ].sum()
                ),

            "wav_insertions":
                int(
                    paired[
                        "wav_insertions"
                    ].sum()
                ),

            "compressed_insertions":
                int(
                    paired[
                        "compressed_insertions"
                    ].sum()
                ),

            "delta_insertions":
                int(
                    paired[
                        "delta_insertions"
                    ].sum()
                ),
        })

        print(
            f"{codec.upper():5s} "
            f"{bitrate:6s} | "
            f"new errors="
            f"{int(paired['new_error'].sum()):3d} | "
            f"worsened="
            f"{int(paired['worsened'].sum()):3d} | "
            f"recovered="
            f"{int(paired['recovered'].sum()):3d}"
        )

    details_df = pd.concat(
        all_details,
        ignore_index=True
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    return (
        details_df,
        summary_df
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main() -> None:

    all_details: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    for dataset_name, path in DATASETS.items():

        details, summary = analyse_dataset(
            dataset_name,
            path
        )

        all_details.append(
            details
        )

        all_summary.append(
            summary
        )

    # --------------------------------------------------
    # Combine results
    # --------------------------------------------------
    details_df = pd.concat(
        all_details,
        ignore_index=True
    )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True
    )

    # --------------------------------------------------
    # Save complete per-utterance results
    # --------------------------------------------------
    details_path = os.path.join(
        OUTPUT_DIR,
        "per_utterance_errors.csv"
    )

    details_df.to_csv(
        details_path,
        index=False
    )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------
    summary_path = os.path.join(
        OUTPUT_DIR,
        "error_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------
    # Find most interesting new failures
    # WAV correct -> compressed incorrect
    # --------------------------------------------------
    top_failures = (
        details_df[
            details_df[
                "new_error"
            ]
        ]
        .sort_values(
            "delta_sample_wer",
            ascending=False
        )
    )

    top_failures_path = os.path.join(
        OUTPUT_DIR,
        "top_new_errors.csv"
    )

    top_failures.to_csv(
        top_failures_path,
        index=False
    )

    # --------------------------------------------------
    # Selected severe conditions
    # --------------------------------------------------
    selected = summary_df[
        (
            (
                summary_df["codec"]
                == "mp3"
            )
            &
            (
                summary_df["bitrate"]
                == "16k"
            )
        )
        |
        (
            (
                summary_df["codec"]
                == "opus"
            )
            &
            (
                summary_df["bitrate"]
                == "8k"
            )
        )
    ]

    print("\n" + "=" * 80)
    print("SELECTED SEVERE CONDITIONS")
    print("=" * 80)

    print(
        selected.to_string(
            index=False
        )
    )

    print("\nERROR TYPE INCREASE")

    for _, row in selected.iterrows():

        print(
            f"{str(row['dataset']):10s} "
            f"{str(row['codec']).upper():5s} "
            f"{str(row['bitrate']):5s} | "
            f"ΔS={int(row['delta_substitutions']):+4d} | "
            f"ΔD={int(row['delta_deletions']):+4d} | "
            f"ΔI={int(row['delta_insertions']):+4d}"
        )

    # --------------------------------------------------
    # Save selected severe-condition summary
    # --------------------------------------------------
    selected_path = os.path.join(
        OUTPUT_DIR,
        "selected_severe_conditions.csv"
    )

    selected.to_csv(
        selected_path,
        index=False
    )

    print("\nSaved:")
    print(details_path)
    print(summary_path)
    print(top_failures_path)
    print(selected_path)


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    main()