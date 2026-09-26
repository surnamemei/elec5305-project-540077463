"""
Which Opus coding mode and audio bandwidth did libopus choose?

Each Opus packet starts with a TOC byte whose top five bits give the
configuration (RFC 6716, Section 3.1): coding mode (SILK / Hybrid / CELT)
and audio bandwidth (NB / MB / WB / SWB / FB). This script encodes the first
NUM_UTTERANCES selected test-clean utterances at every Opus bitrate used in
the experiment, reads the Ogg packets and reports the share of packets in
each configuration.

Output:
    results/final_report/table_opus_modes.csv
"""

import collections
import os
import random
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import torchaudio


OUTPUT_PATH = Path("results") / "final_report" / "table_opus_modes.csv"

BITRATES = ["64k", "32k", "16k", "12k", "8k", "6k"]
NUM_SAMPLES = 500
RANDOM_SEED = 5305
NUM_UTTERANCES = 50


def configuration_name(config: int) -> str:
    """RFC 6716 Table 2."""
    if config < 12:
        return "SILK-" + ["NB", "MB", "WB"][config // 4]
    if config < 16:
        return "Hybrid-" + ["SWB", "FB"][(config - 12) // 2]
    return "CELT-" + ["NB", "WB", "SWB", "FB"][(config - 16) // 4]


def ogg_audio_packets(path: str) -> list[bytes]:
    """Split an Ogg Opus file into packets; drop OpusHead and OpusTags."""
    data = open(path, "rb").read()
    packets, buffer, position = [], b"", 0
    while position < len(data):
        assert data[position:position + 4] == b"OggS"
        num_segments = data[position + 26]
        lacing = data[position + 27:position + 27 + num_segments]
        body = position + 27 + num_segments
        for size in lacing:
            buffer += data[body:body + size]
            body += size
            if size < 255:
                packets.append(buffer)
                buffer = b""
        position = body
    return packets[2:]


def main() -> None:

    dataset = torchaudio.datasets.LIBRISPEECH("data", url="test-clean", download=False)
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), k=NUM_SAMPLES)[:NUM_UTTERANCES]

    rows = []
    for bitrate in BITRATES:
        counts = collections.Counter()
        for index in indices:
            waveform, sample_rate, *_ = dataset[index]
            with tempfile.TemporaryDirectory() as temp_dir:
                wav_path = os.path.join(temp_dir, "input.wav")
                opus_path = os.path.join(temp_dir, "output.opus")
                torchaudio.save(wav_path, waveform, sample_rate)
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                        "-codec:a", "libopus", "-b:a", bitrate, opus_path,
                    ],
                    check=True,
                )
                for packet in ogg_audio_packets(opus_path):
                    counts[configuration_name(packet[0] >> 3)] += 1

        total = sum(counts.values())
        for name, count in counts.most_common():
            rows.append({
                "bitrate": bitrate,
                "configuration": name,
                "packets": count,
                "share": count / total,
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_PATH, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
