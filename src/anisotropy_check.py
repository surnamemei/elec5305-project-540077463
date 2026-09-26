"""
Anisotropy check for Wav2Vec2 layers (justifies standardised drift).

For NUM_PAIRS random pairs of *different* WAV utterances from the fixed
test-clean selection, the frame-wise cosine similarity between the two
utterances is computed at every layer, with and without the per-dimension
standardisation used in representation_analysis.py. If a layer is isotropic,
unrelated utterances should have a cosine similarity near zero.

Requires results/representation_standardisation.pt
(written by representation_analysis.py).

Output:
    results/final_report/table_anisotropy.csv
"""

import random
from pathlib import Path
from typing import cast

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.models import Wav2Vec2Model


RESULTS_DIR = Path("results")
OUTPUT_PATH = RESULTS_DIR / "final_report" / "table_anisotropy.csv"

NUM_SAMPLES = 500
RANDOM_SEED = 5305
NUM_PAIRS = 50

LAYER_NAMES = ["conv"] + [f"layer_{i}" for i in range(1, 13)]


def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cast(
        Wav2Vec2Model, torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
    ).to(device).eval()

    standardisation = torch.load(RESULTS_DIR / "representation_standardisation.pt")

    dataset = torchaudio.datasets.LIBRISPEECH("data", url="test-clean", download=False)
    random.seed(RANDOM_SEED)
    indices = random.sample(range(len(dataset)), k=NUM_SAMPLES)

    # Pairs are drawn from the utterances NOT used for calibration
    # (calibration used the first 100 selected utterances)
    pair_rng = random.Random(RANDOM_SEED + 1)
    candidates = indices[100:]
    pairs = [tuple(pair_rng.sample(candidates, 2)) for _ in range(NUM_PAIRS)]

    def features(index: int) -> dict[str, torch.Tensor]:
        waveform, _, *_ = dataset[index]
        waveform = waveform.to(device)
        with torch.inference_mode():
            conv, _ = model.feature_extractor(waveform, None)
            layers, _ = model.extract_features(waveform)
        out = {"conv": conv[0].cpu()}
        for i, layer in enumerate(layers, start=1):
            out[f"layer_{i}"] = layer[0].cpu()
        return out

    rows = []
    for a, b in pairs:
        fa, fb = features(a), features(b)
        for layer in LAYER_NAMES:
            n = min(len(fa[layer]), len(fb[layer]))
            x, y = fa[layer][:n], fb[layer][:n]
            mean, std = standardisation[layer]
            rows.append({
                "layer": layer,
                "raw_cosine": F.cosine_similarity(x, y, dim=-1).mean().item(),
                "standardised_cosine": F.cosine_similarity(
                    (x - mean) / std, (y - mean) / std, dim=-1
                ).mean().item(),
            })

    table = (
        pd.DataFrame(rows)
        .groupby("layer", sort=False)
        .agg(
            raw_cosine_mean=("raw_cosine", "mean"),
            raw_cosine_std=("raw_cosine", "std"),
            standardised_cosine_mean=("standardised_cosine", "mean"),
            standardised_cosine_std=("standardised_cosine", "std"),
        )
        .reset_index()
    )
    table.insert(1, "num_pairs", NUM_PAIRS)
    table.to_csv(OUTPUT_PATH, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
