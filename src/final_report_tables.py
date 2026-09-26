"""
Authoritative result tables for the final report, with validation checks.

This script does not run any model. It reads the current result files,
re-checks the key numbers independently, and writes the tables that the
final report quotes:

    results/final_report/table_main_results.csv
    results/final_report/table_predictors.csv
    results/final_report/table_within_condition.csv
    results/final_report/table_layer_attenuation.csv
    results/final_report/table_encodec.csv
    results/final_report/validation_report.txt

Validation checks:
    1. corpus WER of every condition recomputed with JiWER from the
       per-utterance transcripts equals the summary CSV;
    2. every analysis (ASR, signal, representation, EnCodec) uses exactly the
       same 500 utterances per subset;
    3. total edit errors (S + D + I) recomputed with jiwer.process_words
       match results/error_analysis/error_summary.csv. The S / D / I split
       itself can differ slightly, because error_analysis.py uses its own
       alignment back-trace with different tie-breaking; JiWER's split is used
       in the report and any difference is listed as a NOTE;
    4. paired-bootstrap CIs recomputed independently (NumPy, different RNG)
       agree with results/bootstrap_results.csv to within 0.15 pp;
    5. the EnCodec WAV baseline reproduces the main test-clean WAV WER.

In addition, speaker-level block-bootstrap CIs for ΔWER are computed
(resampling LibriSpeech speakers with all their utterances, as recommended
by Bisani & Ney, ICASSP 2004). These are the primary CIs in the report
(columns ci_lower_pp / ci_upper_pp); the utterance-level CIs from
bootstrap_results.csv are kept as ci_utterance_lower_pp / ci_utterance_upper_pp.

Run from the repository root:
    python src/final_report_tables.py
"""

from pathlib import Path

import jiwer
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "final_report"

DATASET_NAMES = ["test-clean", "test-other"]

CONDITION_ORDER = [
    ("wav", "uncompressed"),
    ("mp3", "128k"), ("mp3", "64k"), ("mp3", "32k"), ("mp3", "24k"),
    ("mp3", "16k"),
    ("opus", "64k"), ("opus", "32k"), ("opus", "16k"), ("opus", "12k"),
    ("opus", "8k"), ("opus", "6k"),
]

KEY_PREDICTORS = [
    "lsd_db",
    "spectral_distortion",
    "bandwidth_loss_hz",
    "hf_power_loss_db",
    "sdrift_conv",
    "sdrift_layer_1",
    "sdrift_layer_6",
    "sdrift_layer_12",
]

BOOTSTRAP_TOLERANCE_PP = 0.15
NUM_BOOTSTRAP = 2000


class Validator:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures = 0

    def note(self, message: str) -> None:
        self.lines.append(f"[NOTE] {message}")
        print(f"[NOTE] {message}")

    def check(self, passed: bool, message: str) -> None:
        status = "PASS" if passed else "FAIL"
        if not passed:
            self.failures += 1
        self.lines.append(f"[{status}] {message}")
        print(f"[{status}] {message}")


def condition_label(codec: str, bitrate: str) -> str:
    return "WAV" if codec == "wav" else f"{codec.upper() if codec == 'mp3' else 'Opus'} {bitrate}"


def main() -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validator = Validator()

    signal = pd.read_csv(RESULTS_DIR / "signal_distortion_results.csv")
    signal_summary = pd.read_csv(RESULTS_DIR / "signal_distortion_summary.csv")
    rep = pd.read_csv(RESULTS_DIR / "representation_similarity_results.csv")
    rep_summary = pd.read_csv(RESULTS_DIR / "representation_similarity_summary.csv")
    bootstrap = pd.read_csv(RESULTS_DIR / "bootstrap_results.csv")
    errors = pd.read_csv(RESULTS_DIR / "error_analysis" / "error_summary.csv")
    correlations = pd.read_csv(
        RESULTS_DIR / "predictor_analysis" / "predictor_correlations.csv"
    )
    within = pd.read_csv(
        RESULTS_DIR / "predictor_analysis" / "within_condition_correlations.csv"
    )
    differences = pd.read_csv(
        RESULTS_DIR / "predictor_analysis" / "predictor_differences_vs_lsd.csv"
    )
    encodec = pd.read_csv(RESULTS_DIR / "encodec_extension_results.csv")
    encodec_summary = pd.read_csv(RESULTS_DIR / "encodec_extension_summary.csv")

    rng = np.random.default_rng(12345)
    speaker_rng = np.random.default_rng(5305)

    main_rows = []
    attenuation_rows = []
    utterance_sets = {}

    for dataset_name in DATASET_NAMES:

        details = pd.read_csv(RESULTS_DIR / f"{dataset_name}_experiment_details.csv")
        summary = pd.read_csv(RESULTS_DIR / f"{dataset_name}_summary_results.csv")

        # --- check 2: identical utterance sets ------------------------------
        asr_sets = {
            key: frozenset(group["dataset_index"])
            for key, group in details.groupby(["codec", "bitrate"])
        }
        reference_set = asr_sets[("wav", "uncompressed")]
        utterance_sets[dataset_name] = reference_set

        validator.check(
            len(reference_set) == 500
            and all(s == reference_set for s in asr_sets.values()),
            f"{dataset_name}: all 12 ASR conditions use the same 500 utterances",
        )

        for name, frame in [("signal", signal), ("representation", rep)]:
            subset = frame[frame["dataset"] == dataset_name]
            sets = {
                key: frozenset(group["dataset_index"])
                for key, group in subset.groupby(["codec", "bitrate"])
            }
            validator.check(
                len(sets) == 11 and all(s == reference_set for s in sets.values()),
                f"{dataset_name}: {name} analysis uses the same 500 utterances "
                f"for all 11 codec conditions",
            )

        wav_details = details[details["codec"] == "wav"].set_index("dataset_index")
        wav_output = jiwer.process_words(
            wav_details["reference"].tolist(), wav_details["prediction"].tolist()
        )
        wav_wer_summary = float(
            summary.loc[summary["codec"] == "wav", "overall_wer"].iloc[0]
        )

        for codec, bitrate in CONDITION_ORDER:

            cond = details[
                (details["codec"] == codec) & (details["bitrate"] == bitrate)
            ].set_index("dataset_index").loc[wav_details.index]

            # --- check 1: corpus WER --------------------------------------
            output = jiwer.process_words(
                cond["reference"].tolist(), cond["prediction"].tolist()
            )
            summary_row = summary[
                (summary["codec"] == codec) & (summary["bitrate"] == bitrate)
            ].iloc[0]
            validator.check(
                abs(output.wer - summary_row["overall_wer"]) < 1e-12,
                f"{dataset_name} {codec} {bitrate}: corpus WER recomputed "
                f"{output.wer * 100:.3f}% = summary "
                f"{summary_row['overall_wer'] * 100:.3f}%",
            )

            row = {
                "dataset": dataset_name,
                "codec": codec,
                "bitrate": bitrate,
                "condition": condition_label(codec, bitrate),
                "measured_bitrate_kbps": summary_row["aggregate_actual_bitrate_kbps"],
                "compression_ratio": summary_row["average_compression_ratio"],
                "wer_percent": output.wer * 100.0,
                "delta_wer_pp": (output.wer - wav_wer_summary) * 100.0,
                "substitutions": output.substitutions,
                "deletions": output.deletions,
                "insertions": output.insertions,
                "reference_words": sum(
                    len(r.split()) for r in cond["reference"]
                ),
            }

            if codec != "wav":

                # --- check 3: S / D / I -----------------------------------
                error_row = errors[
                    (errors["dataset"] == dataset_name)
                    & (errors["codec"] == codec)
                    & (errors["bitrate"] == bitrate)
                ].iloc[0]
                jiwer_split = (
                    output.substitutions, output.deletions, output.insertions
                )
                csv_split = (
                    int(error_row["compressed_substitutions"]),
                    int(error_row["compressed_deletions"]),
                    int(error_row["compressed_insertions"]),
                )
                validator.check(
                    sum(jiwer_split) == sum(csv_split),
                    f"{dataset_name} {codec} {bitrate}: total errors "
                    f"{sum(jiwer_split)} (JiWER) = {sum(csv_split)} "
                    "(error_summary.csv)",
                )
                if jiwer_split != csv_split:
                    validator.note(
                        f"{dataset_name} {codec} {bitrate}: S/D/I split "
                        f"JiWER {jiwer_split} vs error_summary.csv {csv_split} "
                        "(alignment tie-breaking; JiWER used in report)"
                    )

                # --- check 4: paired bootstrap ----------------------------
                wav_errors = np.array([
                    jiwer.process_words(r, p).wer * len(r.split())
                    for r, p in zip(wav_details["reference"], wav_details["prediction"])
                ])
                cond_errors = np.array([
                    jiwer.process_words(r, p).wer * len(r.split())
                    for r, p in zip(cond["reference"], cond["prediction"])
                ])
                words = np.array([len(r.split()) for r in cond["reference"]])
                n = len(words)
                samples = rng.integers(0, n, size=(NUM_BOOTSTRAP, n))
                deltas = (
                    cond_errors[samples].sum(1) - wav_errors[samples].sum(1)
                ) / words[samples].sum(1) * 100.0
                lower, upper = np.quantile(deltas, [0.025, 0.975])

                boot_row = bootstrap[
                    (bootstrap["dataset"] == dataset_name)
                    & (bootstrap["codec"] == codec)
                    & (bootstrap["bitrate"] == bitrate)
                ].iloc[0]
                validator.check(
                    abs(lower - boot_row["ci_lower_pp"]) < BOOTSTRAP_TOLERANCE_PP
                    and abs(upper - boot_row["ci_upper_pp"]) < BOOTSTRAP_TOLERANCE_PP,
                    f"{dataset_name} {codec} {bitrate}: bootstrap CI "
                    f"[{boot_row['ci_lower_pp']:+.2f}, {boot_row['ci_upper_pp']:+.2f}] "
                    f"reproduced as [{lower:+.2f}, {upper:+.2f}] pp",
                )

                # Speaker-level block bootstrap (primary CI)
                speakers = wav_details["speaker_id"].to_numpy()
                unique_speakers = np.unique(speakers)
                rows_by_speaker = [
                    np.flatnonzero(speakers == s) for s in unique_speakers
                ]
                delta_errors = cond_errors - wav_errors
                speaker_deltas = []
                for _ in range(NUM_BOOTSTRAP):
                    picked = speaker_rng.integers(
                        0, len(unique_speakers), len(unique_speakers)
                    )
                    rows = np.concatenate([rows_by_speaker[k] for k in picked])
                    speaker_deltas.append(
                        delta_errors[rows].sum() / words[rows].sum() * 100.0
                    )
                speaker_lower, speaker_upper = np.quantile(
                    speaker_deltas, [0.025, 0.975]
                )

                utterance_excludes = bool(boot_row["ci_excludes_zero"])
                speaker_excludes = bool(speaker_lower > 0 or speaker_upper < 0)
                validator.check(
                    speaker_excludes == utterance_excludes,
                    f"{dataset_name} {codec} {bitrate}: speaker-level CI "
                    f"[{speaker_lower:+.2f}, {speaker_upper:+.2f}] gives the same "
                    "zero-exclusion conclusion as the utterance-level CI",
                )

                row["num_speakers"] = len(unique_speakers)
                row["ci_lower_pp"] = speaker_lower
                row["ci_upper_pp"] = speaker_upper
                row["ci_excludes_zero"] = speaker_excludes
                row["ci_utterance_lower_pp"] = boot_row["ci_lower_pp"]
                row["ci_utterance_upper_pp"] = boot_row["ci_upper_pp"]
                row["delta_substitutions"] = output.substitutions - wav_output.substitutions
                row["delta_deletions"] = output.deletions - wav_output.deletions
                row["delta_insertions"] = output.insertions - wav_output.insertions

                sig = signal_summary[
                    (signal_summary["dataset"] == dataset_name)
                    & (signal_summary["codec"] == codec)
                    & (signal_summary["bitrate"] == bitrate)
                ].iloc[0]
                row["lsd_db"] = sig["lsd_db_mean"]
                row["dspec_db2"] = sig["spectral_distortion_mean"]
                row["retained_bandwidth_khz"] = sig["retained_bandwidth_mean_hz"] / 1000.0
                row["hf_power_change_db"] = sig["hf_power_change_mean_db"]
                row["mean_delay_samples"] = sig["estimated_delay_mean_samples"]

                rep_rows = rep_summary[
                    (rep_summary["dataset"] == dataset_name)
                    & (rep_summary["codec"] == codec)
                    & (rep_summary["bitrate"] == bitrate)
                ].set_index("layer")
                for layer in ["conv", "layer_1", "layer_6", "layer_12"]:
                    row[f"sdrift_{layer}"] = rep_rows.loc[layer, "sdrift_mean"]
                row["raw_drift_layer_12"] = rep_rows.loc["layer_12", "drift_mean"]

                peak_layer = rep_rows.loc[
                    [l for l in rep_rows.index], "sdrift_mean"
                ].idxmax()
                attenuation_rows.append({
                    "dataset": dataset_name,
                    "condition": row["condition"],
                    "sdrift_conv": rep_rows.loc["conv", "sdrift_mean"],
                    "sdrift_peak": rep_rows["sdrift_mean"].max(),
                    "peak_layer": peak_layer,
                    "sdrift_layer_12": rep_rows.loc["layer_12", "sdrift_mean"],
                    "conv_to_layer12_ratio": (
                        rep_rows.loc["conv", "sdrift_mean"]
                        / rep_rows.loc["layer_12", "sdrift_mean"]
                    ),
                    "peak_to_layer12_ratio": (
                        rep_rows["sdrift_mean"].max()
                        / rep_rows.loc["layer_12", "sdrift_mean"]
                    ),
                    "raw_conv_to_layer12_ratio": (
                        rep_rows.loc["conv", "drift_mean"]
                        / rep_rows.loc["layer_12", "drift_mean"]
                    ),
                    "raw_layer10_to_layer11_ratio": (
                        rep_rows.loc["layer_10", "drift_mean"]
                        / rep_rows.loc["layer_11", "drift_mean"]
                    ),
                })

            main_rows.append(row)

        validator.check(
            abs(wav_wer_summary * 100 - {"test-clean": 3.1668, "test-other": 8.2621}[dataset_name]) < 1e-3,
            f"{dataset_name}: WAV baseline WER = {wav_wer_summary * 100:.3f}%",
        )

    # --- check 5: EnCodec pairing and baseline -----------------------------
    encodec_sets = {
        key: frozenset(group["dataset_index"])
        for key, group in encodec.groupby("condition")
    }
    validator.check(
        all(s == utterance_sets["test-clean"] for s in encodec_sets.values()),
        "EnCodec extension uses the same 500 test-clean utterances "
        "for all 5 conditions",
    )

    encodec_rows = []
    main_wav = [r for r in main_rows if r["dataset"] == "test-clean" and r["codec"] == "wav"][0]
    for condition in ["wav", "resample_control", "encodec_24k", "encodec_6k", "encodec_1.5k"]:
        group = encodec[encodec["condition"] == condition]
        output = jiwer.process_words(
            group["reference"].tolist(), group["prediction"].tolist()
        )
        summary_row = encodec_summary[encodec_summary["condition"] == condition].iloc[0]
        validator.check(
            abs(output.wer - summary_row["mean_wer"]) < 1e-12,
            f"EnCodec {condition}: corpus WER recomputed {output.wer * 100:.3f}% "
            "matches encodec_extension_summary.csv",
        )
        if condition == "wav":
            validator.check(
                abs(output.wer * 100 - main_wav["wer_percent"]) < 1e-9,
                f"EnCodec WAV baseline {output.wer * 100:.3f}% equals main "
                f"test-clean WAV {main_wav['wer_percent']:.3f}%",
            )
        encodec_rows.append({
            "condition": condition,
            "wer_percent": output.wer * 100.0,
            "delta_wer_pp": summary_row["delta_wer_percentage_points"],
            "lsd_db": summary_row["mean_lsd_db"],
            "sdrift_layer_1": summary_row["mean_layer_1_drift"],
            "sdrift_layer_6": summary_row["mean_layer_6_drift"],
            "sdrift_layer_12": summary_row["mean_layer_12_drift"],
            "raw_drift_layer_12": summary_row["mean_layer_12_raw_drift"],
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
        })

    # --- predictor tables --------------------------------------------------
    predictor_table = correlations[
        correlations["predictor"].isin(KEY_PREDICTORS)
    ].copy()
    predictor_table = predictor_table.merge(
        differences[["dataset", "analysis", "predictor", "difference",
                     "ci_lower", "ci_upper", "ci_excludes_zero"]].rename(
            columns={
                "difference": "minus_lsd",
                "ci_lower": "minus_lsd_ci_lower",
                "ci_upper": "minus_lsd_ci_upper",
                "ci_excludes_zero": "minus_lsd_ci_excludes_zero",
            }
        ),
        on=["dataset", "analysis", "predictor"],
        how="left",
    )

    within_table = within[
        within["predictor"].isin(["lsd_db", "bandwidth_loss_hz", "sdrift_conv", "sdrift_layer_12"])
    ].pivot_table(
        index=["dataset", "codec", "bitrate"],
        columns="predictor",
        values="spearman",
    ).reset_index()

    # --- write --------------------------------------------------------------
    pd.DataFrame(main_rows).to_csv(OUTPUT_DIR / "table_main_results.csv", index=False)
    predictor_table.to_csv(OUTPUT_DIR / "table_predictors.csv", index=False)
    within_table.to_csv(OUTPUT_DIR / "table_within_condition.csv", index=False)
    pd.DataFrame(attenuation_rows).to_csv(
        OUTPUT_DIR / "table_layer_attenuation.csv", index=False
    )
    pd.DataFrame(encodec_rows).to_csv(OUTPUT_DIR / "table_encodec.csv", index=False)

    validator.lines.append(
        f"\n{validator.failures} failed check(s) out of {len(validator.lines)}"
    )
    (OUTPUT_DIR / "validation_report.txt").write_text("\n".join(validator.lines) + "\n")

    print(f"\n{validator.failures} failed check(s)")
    print(f"Tables written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
