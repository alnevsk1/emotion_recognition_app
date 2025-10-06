from transformers import pipeline
import torchaudio

# Create pipeline
emotion_recognizer = pipeline("audio-classification", model="./fine-tuned-emotion-model")

wav_path = "path/to/audio"

# Load audio
speech, sample_rate = torchaudio.load(wav_path)

# Make audio mono
if speech.shape[0] > 1:
    speech = speech.mean(dim=0, keepdim=True)

# Resample for 16kHz
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    speech = resampler(speech)

# To numpy 1D
speech = speech.squeeze().numpy()

# Prediction
predictions = emotion_recognizer(speech)
print(predictions)