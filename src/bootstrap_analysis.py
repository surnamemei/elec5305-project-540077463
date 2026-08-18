import os
import random
import pandas as pd
from jiwer import wer

# --------------------------------------------------
# Settings
# --------------------------------------------------
CLEAN_DETAIL_PATH = "results/test-clean_experiment_details.csv"
OTHER_DETAIL_PATH = "results/test-other_experiment_details.csv"

OUTPUT_PATH = "results/bootstrap_results.csv"

N_BOOTSTRAP = 2000
RANDOM_SEED = 5305


# --------------------------------------------------
# Bootstrap one compressed condition vs WAV baseline
# --------------------------------------------------
def bootstrap_condition(
    df,
    codec,
    bitrate,
    n_bootstrap=N_BOOTSTRAP,
    seed=RANDOM_SEED
):
    # WAV baseline
    baseline = df[
        df["codec"] == "wav"
    ][
        ["dataset_index", "reference", "prediction"]
    ].copy()

    baseline = baseline.rename(
        columns={
            "prediction": "prediction_wav"
        }
    )

    # Compressed condition
    compressed = df[
        (df["codec"] == codec) &
        (df["bitrate"] == bitrate)
    ][
        ["dataset_index", "prediction"]
    ].copy()

    compressed = compressed.rename(
        columns={
            "prediction": "prediction_compressed"
        }
    )

    # Pair the exact same utterances
    paired = baseline.merge(
        compressed,
        on="dataset_index",
        how="inner"
    )

    if len(paired) == 0:
        raise ValueError(
            f"No paired samples found for {codec} {bitrate}"
        )

    references = paired["reference"].tolist()
    wav_predictions = paired["prediction_wav"].tolist()
    compressed_predictions = paired[
        "prediction_compressed"
    ].tolist()

    # --------------------------------------------------
    # Observed WER
    # --------------------------------------------------
    wav_wer = wer(
        references,
        wav_predictions
    )

    compressed_wer = wer(
        references,
        compressed_predictions
    )

    observed_delta = compressed_wer - wav_wer

    # --------------------------------------------------
    # Paired bootstrap
    # --------------------------------------------------
    rng = random.Random(seed)

    bootstrap_deltas = []

    n = len(paired)

    for _ in range(n_bootstrap):

        # Same sampled utterances used for both
        # WAV and compressed condition
        indices = [
            rng.randrange(n)
            for _ in range(n)
        ]

        sampled_references = [
            references[i]
            for i in indices
        ]

        sampled_wav = [
            wav_predictions[i]
            for i in indices
        ]

        sampled_compressed = [
            compressed_predictions[i]
            for i in indices
        ]

        sampled_wav_wer = wer(
            sampled_references,
            sampled_wav
        )

        sampled_compressed_wer = wer(
            sampled_references,
            sampled_compressed
        )

        bootstrap_deltas.append(
            sampled_compressed_wer
            - sampled_wav_wer
        )

    bootstrap_deltas.sort()

    lower_index = int(
        0.025 * n_bootstrap
    )

    upper_index = int(
        0.975 * n_bootstrap
    )

    ci_lower = bootstrap_deltas[
        lower_index
    ]

    ci_upper = bootstrap_deltas[
        min(
            upper_index,
            n_bootstrap - 1
        )
    ]

    return {
        "codec": codec,
        "bitrate": bitrate,
        "num_samples": len(paired),

        "wav_wer": wav_wer,
        "compressed_wer": compressed_wer,

        "delta_wer": observed_delta,

        "ci_lower": ci_lower,
        "ci_upper": ci_upper,

        "ci_excludes_zero": (
            ci_lower > 0 or ci_upper < 0
        ),
    }


# --------------------------------------------------
# Analyse one dataset
# --------------------------------------------------
def analyse_dataset(
    dataset_name,
    detail_path
):
    print("\n" + "=" * 75)
    print(
        f"BOOTSTRAP ANALYSIS: "
        f"{dataset_name}"
    )
    print("=" * 75)

    df = pd.read_csv(detail_path)

    # Make sure bitrate stays as string
    df["bitrate"] = df[
        "bitrate"
    ].astype(str)

    conditions = (
        df[
            df["codec"] != "wav"
        ][
            ["codec", "bitrate"]
        ]
        .drop_duplicates()
    )

    results = []

    for row in conditions.itertuples(
        index=False
    ):

        codec = str(row.codec)
        bitrate = str(row.bitrate)

        result = bootstrap_condition(
            df,
            codec,
            bitrate
        )

        result["dataset"] = dataset_name

        results.append(result)

        print(
            f"{codec.upper():5s} "
            f"{bitrate:6s} | "
            f"WER="
            f"{result['compressed_wer'] * 100:6.2f}% | "
            f"ΔWER="
            f"{result['delta_wer'] * 100:+6.2f} pp | "
            f"95% CI "
            f"["
            f"{result['ci_lower'] * 100:+.2f}, "
            f"{result['ci_upper'] * 100:+.2f}"
            f"] pp"
        )

    return results


# --------------------------------------------------
# Run
# --------------------------------------------------
all_results = []

all_results.extend(
    analyse_dataset(
        "test-clean",
        CLEAN_DETAIL_PATH
    )
)

all_results.extend(
    analyse_dataset(
        "test-other",
        OTHER_DETAIL_PATH
    )
)

# --------------------------------------------------
# Save
# --------------------------------------------------
results_df = pd.DataFrame(
    all_results
)

# Convert to percentage points for easier reading
results_df[
    "wav_wer_percent"
] = results_df["wav_wer"] * 100

results_df[
    "compressed_wer_percent"
] = results_df[
    "compressed_wer"
] * 100

results_df[
    "delta_wer_pp"
] = results_df[
    "delta_wer"
] * 100

results_df[
    "ci_lower_pp"
] = results_df[
    "ci_lower"
] * 100

results_df[
    "ci_upper_pp"
] = results_df[
    "ci_upper"
] * 100

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 75)
print("BOOTSTRAP COMPLETE")
print("=" * 75)

print(
    f"Saved: {OUTPUT_PATH}"
)