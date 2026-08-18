import torch
import torchaudio
from torchaudio.utils import _download_asset

# --------------------------------------------------
# 1. Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------------------------
# 2. Load pretrained Wav2Vec2 ASR model
# --------------------------------------------------
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
model.eval()

labels = bundle.get_labels()
print("Expected sample rate:", bundle.sample_rate)

# --------------------------------------------------
# 3. Download one test speech sample
# --------------------------------------------------
speech_file = _download_asset(
    "tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
)

waveform, sample_rate = torchaudio.load(speech_file)

print("Original sample rate:", sample_rate)
print("Waveform shape:", waveform.shape)

# --------------------------------------------------
# 4. Resample if necessary
# --------------------------------------------------
target_sample_rate = int(bundle.sample_rate)

if sample_rate != target_sample_rate:
    waveform = torchaudio.functional.resample(
        waveform,
        sample_rate,
        target_sample_rate
    )

waveform = waveform.to(device)

# --------------------------------------------------
# 5. Run Wav2Vec2
# --------------------------------------------------
with torch.inference_mode():
    emissions, _ = model(waveform)

# Highest probability label at each time frame
token_ids = torch.argmax(emissions, dim=-1)[0]

# --------------------------------------------------
# 6. Simple CTC decoder
# --------------------------------------------------
blank_id = 0

collapsed_ids = []
previous_id = None

for token_id in token_ids.tolist():
    # Remove repeated CTC predictions
    if token_id != previous_id:
        # Remove blank token
        if token_id != blank_id:
            collapsed_ids.append(token_id)

    previous_id = token_id

transcript = "".join(labels[i] for i in collapsed_ids)

# "|" means word boundary
transcript = transcript.replace("|", " ")

print("\nPredicted transcript:")
print(transcript)