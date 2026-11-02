"""Two-utterance, live codec + fixed-ASR demonstration; no research outputs modified."""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio


def decode(emissions, labels):
    tokens = torch.argmax(emissions, dim=-1)[0].tolist()
    previous, output = None, []
    for token in tokens:
        if token != previous and token != 0:
            output.append(labels[token])
        previous = token
    return "".join(output).replace("|", " ").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                        help="directory containing LibriSpeech/test-clean")
    parser.add_argument("--output", type=Path, default=Path("demo_assets"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with Path("results/final_report/case_study_selection.csv").open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use --device cpu")
    print("Loading fixed WAV2VEC2_ASR_BASE_960H model (cached or downloaded by TorchAudio)...", flush=True)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(args.device).eval()
    labels = bundle.get_labels()
    print(f"Device: {args.device}; sample rate: {bundle.sample_rate} Hz", flush=True)
    outcomes = []
    for case in cases:
        role, uid = case["role"], case["utterance"]
        speaker, chapter, _ = uid.split("-")
        source = args.data_root / "LibriSpeech" / "test-clean" / speaker / chapter / (uid + ".flac")
        if not source.exists():
            raise FileNotFoundError(f"Missing real LibriSpeech input: {source}")
        waveform, sr = torchaudio.load(str(source))
        if sr != 16000 or waveform.shape[0] != 1:
            raise ValueError("Expected the original mono 16 kHz LibriSpeech input")
        wav = args.output / f"{role}_reference.wav"
        opus = args.output / f"{role}_8k.opus"
        decoded_wav = args.output / f"{role}_opus_decoded.wav"
        torchaudio.save(str(wav), waveform, sr, encoding="PCM_S", bits_per_sample=16)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-codec:a", "libopus", "-b:a", "8k", str(opus)], check=True)
        compressed, compressed_sr = torchaudio.load(str(opus))
        if compressed_sr != sr:
            compressed = torchaudio.functional.resample(compressed, compressed_sr, sr)
        torchaudio.save(str(decoded_wav), compressed, sr, encoding="PCM_S", bits_per_sample=16)
        predictions = []
        for signal in (waveform, compressed):
            with torch.inference_mode():
                emissions, _ = model(signal.to(args.device))
            predictions.append(decode(emissions, labels))
        result = {"role": role, "utterance": uid, "reference": case["reference"],
                  "live_wav_prediction": predictions[0], "live_opus_prediction": predictions[1],
                  "live_wav_wer": jiwer.wer(case["reference"], predictions[0]),
                  "live_opus_wer": jiwer.wer(case["reference"], predictions[1]),
                  "report_cached_opus_prediction": case["compressed_prediction"],
                  "report_cached_lsd_db": float(case["lsd_db"]),
                  "report_cached_layer12_drift": float(case["sdrift_layer_12"])}
        outcomes.append(result)
        print("\n" + role.upper() + " " + uid, flush=True)
        print("Reference:     " + result["reference"], flush=True)
        print("Live WAV ASR:  " + predictions[0], flush=True)
        print("Live Opus ASR: " + predictions[1], flush=True)
        print(f"Live WER: WAV {result['live_wav_wer']:.1%}; Opus {result['live_opus_wer']:.1%}", flush=True)
        print("Report's saved Opus ASR: " + result["report_cached_opus_prediction"], flush=True)
        print(f"Saved report measures (not computed live): LSD {result['report_cached_lsd_db']:.2f} dB; "
              f"layer-12 drift {result['report_cached_layer12_drift']:.3f}", flush=True)
        length = min(waveform.shape[1], compressed.shape[1])
        window = torch.hann_window(400)
        specs = [20 * torch.log10(torch.stft(x[0, :length], n_fft=512, hop_length=160,
                    win_length=400, window=window, return_complex=True).abs() + 1e-12)
                 for x in (waveform, compressed)]
        peak = specs[0].max().item()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, spec, title in zip(axes, specs, ["Reference WAV", "Opus 8 kbps decoded"]):
            im = ax.imshow((spec - peak).numpy(), origin="lower", aspect="auto",
                           extent=[0, length / sr, 0, sr / 2000], vmin=-80, vmax=0, cmap="magma")
            ax.set(title=title, xlabel="Time (s)", ylabel="Frequency (kHz)")
        fig.suptitle(f"{role.title()} case {uid}: live audio, 80 dB display range")
        fig.colorbar(im, ax=axes, label="dB relative to reference peak")
        fig.savefig(args.output / f"{role}_live_spectrogram.png", dpi=160)
        plt.close(fig)
    (args.output / "live_results.json").write_text(json.dumps(outcomes, indent=2) + "\n", encoding="utf-8")
    print(f"\nLive WAVs, Opus files, spectrograms and transcripts saved to {args.output.resolve()}")
    print("Report LSD and representation drift are saved values, not live calculations.")


if __name__ == "__main__":
    main()
