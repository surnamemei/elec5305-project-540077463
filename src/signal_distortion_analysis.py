"""
Signal-level codec distortion analysis (feedback points 15, 16, 17 and 23).

For every selected utterance and codec condition the script:

1. encodes and decodes the utterance with FFmpeg;
2. aligns the decoded signal to the WAV reference (cross-correlation);
3. measures log-spectral distortion against the WAV reference;
4. measures the retained bandwidth of the codec output.

Log-spectral floor
------------------
Log-magnitude spectra are clipped to a fixed dynamic range below the
reference spectrogram peak (DYNAMIC_RANGE_DB). Without this floor, bins that
a codec sets to exactly zero (common for MP3) are mapped to 20*log10(eps),
which is a huge negative number, and those few bins dominate the average.
The floor makes the metric measure audible / physically meaningful spectral
change rather than the value of eps.

Retained bandwidth
------------------
The long-term power spectrum of the decoded signal is divided by that of the
reference. The retained bandwidth is the highest frequency at which the codec
still keeps power within BANDWIDTH_DROP_DB of the reference. This detects the
low-pass cutoff that codecs apply at low bitrates, which a cumulative-energy
roll-off measure does not (speech energy is concentrated below ~3 kHz, so a
95%-energy roll-off barely moves when 4-8 kHz content is removed).
"""

import os
import random
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


# ==================================================
# Settings
# ==================================================

DATA_ROOT = "data"
DATASET_NAMES = ["test-clean", "test-other"]

# Same utterances as run_all_experiments.py (same seed and sample size)
NUM_SAMPLES = 500
RANDOM_SEED = 5305

# STFT settings (25 ms window, 10 ms hop at 16 kHz)
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

# Only used to avoid log10(0); the dynamic-range floor below does the work
LOG_EPS = 1e-12

# Log spectra are clipped at (reference peak - DYNAMIC_RANGE_DB)
DYNAMIC_RANGE_DB = 80.0

# Retained bandwidth: highest frequency where codec power is within
# BANDWIDTH_DROP_DB of the reference power
BANDWIDTH_DROP_DB = 20.0

# Band used for the high-frequency power-loss measure
HF_BAND_HZ = (4000.0, 8000.0)

MAX_SHIFT_SAMPLES = 4000


# ==================================================
# Output paths
# ==================================================

RESULTS_DIR = "results"
DETAIL_OUTPUT_PATH = os.path.join(RESULTS_DIR, "signal_distortion_results.csv")
SUMMARY_OUTPUT_PATH = os.path.join(RESULTS_DIR, "signal_distortion_summary.csv")
FREQ_OUTPUT_DIR = os.path.join(RESULTS_DIR, "frequency_distortion")
FIGURE_OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FREQ_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)


# ==================================================
# Compression conditions
# ==================================================

CONDITIONS = [
    {"codec": "mp3", "bitrate": "128k"},
    {"codec": "mp3", "bitrate": "64k"},
    {"codec": "mp3", "bitrate": "32k"},
    {"codec": "mp3", "bitrate": "24k"},
    {"codec": "mp3", "bitrate": "16k"},
    {"codec": "opus", "bitrate": "64k"},
    {"codec": "opus", "bitrate": "32k"},
    {"codec": "opus", "bitrate": "16k"},
    {"codec": "opus", "bitrate": "12k"},
    {"codec": "opus", "bitrate": "8k"},
    {"codec": "opus", "bitrate": "6k"},
]

CODEC_COLOURS = {"mp3": "#2a78d6", "opus": "#eb6834"}


# ==================================================
# Encode + decode
# ==================================================

def compress_audio(waveform, sample_rate, codec, bitrate):
    """Encode with FFmpeg and decode back to a waveform."""

    encoders = {"mp3": "libmp3lame", "opus": "libopus"}

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, f"compressed.{codec}")

        torchaudio.save(input_path, waveform, sample_rate)

        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", input_path,
            "-codec:a", encoders[codec],
            "-b:a", bitrate,
            output_path,
        ]
        subprocess.run(command, check=True)

        compressed_waveform, compressed_sr = torchaudio.load(output_path)

    return compressed_waveform, int(compressed_sr)


# ==================================================
# Alignment (feedback point 23)
# ==================================================

def align_waveforms(reference, processed, max_shift_samples=MAX_SHIFT_SAMPLES):
    """
    Align processed audio to reference using normalised cross-correlation
    over a limited lag range. The correlation for every lag is computed at
    once with an FFT; the normalisation uses the energy of the overlapping
    segments, so the score is identical to the direct sliding computation.

    Positive delay means the processed signal is delayed relative to the
    reference.
    """

    ref = reference.mean(dim=0).cpu().double().numpy()
    proc = processed.mean(dim=0).cpu().double().numpy()

    n_ref = len(ref)
    n_proc = len(proc)
    max_shift = min(max_shift_samples, n_ref - 1, n_proc - 1)

    n_fft = 1 << int(np.ceil(np.log2(n_ref + n_proc - 1)))
    correlation = np.fft.irfft(
        np.fft.rfft(proc, n_fft) * np.conj(np.fft.rfft(ref, n_fft)),
        n_fft,
    )

    ref_energy = np.concatenate([[0.0], np.cumsum(ref ** 2)])
    proc_energy = np.concatenate([[0.0], np.cumsum(proc ** 2)])

    best_lag = 0
    best_score = -np.inf

    for lag in range(-max_shift, max_shift + 1):
        if lag >= 0:
            length = min(n_ref, n_proc - lag)
            dot = correlation[lag]
            energy = ref_energy[length] * (
                proc_energy[lag + length] - proc_energy[lag]
            )
        else:
            shift = -lag
            length = min(n_proc, n_ref - shift)
            dot = correlation[n_fft + lag]
            energy = proc_energy[length] * (
                ref_energy[shift + length] - ref_energy[shift]
            )

        if length < 100 or energy <= 0.0:
            continue

        score = dot / np.sqrt(energy)
        if score > best_score:
            best_score = score
            best_lag = lag

    ref_t = reference.mean(dim=0)
    proc_t = processed.mean(dim=0)

    if best_lag >= 0:
        aligned_processed = proc_t[best_lag:]
        aligned_reference = ref_t
    else:
        aligned_reference = ref_t[-best_lag:]
        aligned_processed = proc_t

    length = min(len(aligned_reference), len(aligned_processed))

    return (
        aligned_reference[:length].unsqueeze(0),
        aligned_processed[:length].unsqueeze(0),
        best_lag,
    )


# ==================================================
# Spectra
# ==================================================

def magnitude_spectrogram(waveform):
    window = torch.hann_window(WIN_LENGTH)
    stft = torch.stft(
        waveform.squeeze(0),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    return torch.abs(stft)


def spectral_distortion(reference_mag, processed_mag):
    """
    Returns:
        dspec:  mean squared log-spectral difference (dB^2), feedback point 15
        lsd_db: log-spectral distance, RMS over frequency then mean over
                frames (dB)
        dfrequency: mean absolute log-spectral difference per frequency
                bin (dB), feedback point 16
    """

    ref_log = 20.0 * torch.log10(reference_mag + LOG_EPS)
    proc_log = 20.0 * torch.log10(processed_mag + LOG_EPS)

    floor = ref_log.max() - DYNAMIC_RANGE_DB
    ref_log = torch.clamp(ref_log, min=floor)
    proc_log = torch.clamp(proc_log, min=floor)

    difference = proc_log - ref_log

    dspec = torch.mean(difference ** 2)
    lsd_db = torch.mean(torch.sqrt(torch.mean(difference ** 2, dim=0)))
    dfrequency = torch.mean(torch.abs(difference), dim=1)

    return float(dspec), float(lsd_db), dfrequency.numpy()


def bandwidth_measures(reference_mag, processed_mag, frequencies):
    """
    Returns:
        retained_bandwidth_hz: highest frequency where the codec keeps
            long-term power within BANDWIDTH_DROP_DB of the reference
            (feedback point 17)
        hf_power_change_db: codec / reference power in HF_BAND_HZ (dB)
    """

    ref_power = torch.mean(reference_mag ** 2, dim=1).double()
    proc_power = torch.mean(processed_mag ** 2, dim=1).double()

    ratio_db = 10.0 * torch.log10(
        (proc_power + LOG_EPS) / (ref_power + LOG_EPS)
    )

    retained = torch.where(ratio_db > -BANDWIDTH_DROP_DB)[0]
    if len(retained) == 0:
        retained_bandwidth_hz = 0.0
    else:
        retained_bandwidth_hz = float(frequencies[int(retained[-1])])

    band = (frequencies >= HF_BAND_HZ[0]) & (frequencies <= HF_BAND_HZ[1])
    band = torch.from_numpy(band)
    hf_power_change_db = 10.0 * torch.log10(
        (proc_power[band].sum() + LOG_EPS) / (ref_power[band].sum() + LOG_EPS)
    )

    return retained_bandwidth_hz, float(hf_power_change_db)


# ==================================================
# Run distortion analysis
# ==================================================

results = []
frequency_results = {}

for dataset_name in DATASET_NAMES:

    print(f"\nLoading LibriSpeech {dataset_name}...")

    dataset = torchaudio.datasets.LIBRISPEECH(
        root=DATA_ROOT,
        url=dataset_name,
        download=False,
    )

    random.seed(RANDOM_SEED)
    sample_indices = random.sample(
        range(len(dataset)),
        k=min(NUM_SAMPLES, len(dataset)),
    )

    print(f"Selected utterances: {len(sample_indices)}")

    for index in tqdm(sample_indices, desc=dataset_name, unit="sample"):

        (
            waveform,
            sample_rate,
            reference_text,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = dataset[index]

        frequencies = np.linspace(0, sample_rate / 2, N_FFT // 2 + 1)

        for condition in CONDITIONS:

            codec = condition["codec"]
            bitrate = condition["bitrate"]

            compressed_waveform, compressed_sr = compress_audio(
                waveform, sample_rate, codec, bitrate
            )

            if compressed_sr != sample_rate:
                compressed_waveform = torchaudio.functional.resample(
                    compressed_waveform, compressed_sr, sample_rate
                )

            aligned_ref, aligned_comp, delay_samples = align_waveforms(
                waveform, compressed_waveform
            )

            reference_mag = magnitude_spectrogram(aligned_ref)
            processed_mag = magnitude_spectrogram(aligned_comp)

            dspec, lsd_db, dfrequency = spectral_distortion(
                reference_mag, processed_mag
            )

            retained_bandwidth_hz, hf_power_change_db = bandwidth_measures(
                reference_mag, processed_mag, frequencies
            )

            results.append({
                "dataset": dataset_name,
                "dataset_index": index,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
                "utterance_id": utterance_id,
                "codec": codec,
                "bitrate": bitrate,
                "spectral_distortion": dspec,
                "lsd_db": lsd_db,
                "retained_bandwidth_hz": retained_bandwidth_hz,
                "hf_power_change_db": hf_power_change_db,
                "estimated_delay_samples": delay_samples,
                "estimated_delay_ms": delay_samples / sample_rate * 1000.0,
            })

            key = (dataset_name, codec, bitrate)
            if key not in frequency_results:
                frequency_results[key] = {
                    "frequencies": frequencies,
                    "distortions": [],
                }
            frequency_results[key]["distortions"].append(dfrequency)


# ==================================================
# Save detailed results and summary
# ==================================================

df = pd.DataFrame(results)
df.to_csv(DETAIL_OUTPUT_PATH, index=False)

summary = (
    df.groupby(["dataset", "codec", "bitrate"], sort=False)
    .agg(
        spectral_distortion_mean=("spectral_distortion", "mean"),
        spectral_distortion_median=("spectral_distortion", "median"),
        spectral_distortion_std=("spectral_distortion", "std"),
        lsd_db_mean=("lsd_db", "mean"),
        lsd_db_std=("lsd_db", "std"),
        retained_bandwidth_mean_hz=("retained_bandwidth_hz", "mean"),
        retained_bandwidth_median_hz=("retained_bandwidth_hz", "median"),
        hf_power_change_mean_db=("hf_power_change_db", "mean"),
        estimated_delay_mean_samples=("estimated_delay_samples", "mean"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)


# ==================================================
# Save frequency-dependent distortion curves
# ==================================================

for (dataset_name, codec, bitrate), data in frequency_results.items():

    distortions = np.stack(data["distortions"], axis=0)

    frequency_df = pd.DataFrame({
        "frequency_hz": data["frequencies"],
        "mean_distortion_db": np.mean(distortions, axis=0),
        "median_distortion_db": np.median(distortions, axis=0),
    })

    frequency_df.to_csv(
        os.path.join(
            FREQ_OUTPUT_DIR,
            f"{dataset_name}_{codec}_{bitrate}_frequency_distortion.csv",
        ),
        index=False,
    )


# ==================================================
# Plot D(f) for every condition (test-clean)
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

for ax, codec, cmap in zip(axes, ["mp3", "opus"], ["Blues", "Oranges"]):

    codec_conditions = [c for c in CONDITIONS if c["codec"] == codec]
    shades = plt.get_cmap(cmap)(
        np.linspace(0.35, 0.95, len(codec_conditions))
    )

    for condition, shade in zip(codec_conditions, shades):
        data = frequency_results[("test-clean", codec, condition["bitrate"])]
        ax.plot(
            data["frequencies"],
            np.mean(np.stack(data["distortions"]), axis=0),
            color=shade,
            linewidth=2,
            label=condition["bitrate"],
        )

    ax.set_title(codec.upper())
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Target bitrate")

axes[0].set_ylabel("Mean |log-spectral difference| (dB)")
fig.suptitle(
    "Frequency-dependent codec distortion D(f), test-clean "
    f"({DYNAMIC_RANGE_DB:.0f} dB floor)"
)
fig.tight_layout()

figure_path = os.path.join(
    FIGURE_OUTPUT_DIR, "frequency_distortion_comparison.png"
)
fig.savefig(figure_path, dpi=300)
plt.close(fig)


# ==================================================
# Print summary
# ==================================================

print("\n" + "=" * 90)
print("SIGNAL DISTORTION SUMMARY")
print("=" * 90)
print(summary.to_string(index=False))

print("\nSaved:")
print(DETAIL_OUTPUT_PATH)
print(SUMMARY_OUTPUT_PATH)
print(FREQ_OUTPUT_DIR)
print(figure_path)
