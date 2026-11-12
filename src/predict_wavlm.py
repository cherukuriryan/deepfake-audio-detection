import sys
import joblib
import librosa
import numpy as np
import torch

from transformers import WavLMModel, Wav2Vec2FeatureExtractor


MODEL_NAME = "microsoft/wavlm-base-plus"
MODEL_REVISION = "refs/pr/1"
SAMPLE_RATE = 16000
MAX_SECONDS = 4
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS


print("Loading WavLM...")

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION
)

wavlm = WavLMModel.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION,
    use_safetensors=True
)

wavlm.eval()

classifier = joblib.load(
    "models/deepfake_audio_wavlm_logistic.pkl"
)

print("Models loaded.")


def extract_embedding(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]

    audio = audio.astype(np.float32)

    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = wavlm(
            input_values=inputs.input_values,
            attention_mask=inputs.get("attention_mask")
        )

    embedding = (
        outputs.last_hidden_state
        .mean(dim=1)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return embedding


def predict_audio(audio_path):

    embedding = extract_embedding(audio_path)

    prediction = classifier.predict(
        [embedding]
    )[0]

    probabilities = classifier.predict_proba(
        [embedding]
    )[0]

    confidence = np.max(probabilities) * 100

    return prediction, confidence


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python src/predict_wavlm.py <audio_file>"
        )
        sys.exit(1)

    audio_path = sys.argv[1]

    try:
        prediction, confidence = predict_audio(
            audio_path
        )

        print("\nPrediction Results")
        print("--------------------")
        print("Prediction:", prediction)
        print(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as error:
        print("Error:", error)