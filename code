import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Model
import threading

# ---------- Model Definition (same as training) ----------
class Classifier(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------- Load Models (runs once at startup) ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Wav2Vec2 (this may take a minute the first time)...")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec.eval()
for param in wav2vec.parameters():
    param.requires_grad = False

print("Loading trained classifier...")
classifier = Classifier().to(device)
classifier.load_state_dict(torch.load("classifier_v1.pt", map_location=device))
classifier.eval()

print("Models loaded. Ready.")

# ---------- Audio Processing ----------
MAX_LENGTH = 64000  # 4 seconds at 16kHz

def process_audio(filepath):
    import librosa
    y, sr = librosa.load(filepath, sr=16000, mono=True)
    waveform = torch.from_numpy(y).float()

    if waveform.shape[0] > MAX_LENGTH:
        waveform = waveform[:MAX_LENGTH]
    else:
        pad_len = MAX_LENGTH - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))

    return waveform.unsqueeze(0)

# ---------- Feature Extraction for Reasoning ----------
def extract_audio_features(filepath):
    import librosa
    y, sr = librosa.load(filepath, sr=16000)

    features = {}

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    features['pitch_std'] = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
    features['pitch_mean'] = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

    flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness'] = float(np.mean(flatness))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_std'] = float(np.std(centroid))

    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = float(np.mean(zcr))

    rms = librosa.feature.rms(y=y)[0]
    silence_ratio = np.sum(rms < np.percentile(rms, 20)) / len(rms)
    features['silence_ratio'] = float(silence_ratio)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['bandwidth_std'] = float(np.std(bandwidth))

    return features


def get_reasoning(prob_fake, features):
    signals = []

    if prob_fake >= 0.95:
        if features['pitch_std'] < 20:
            signals.append(f"unusually stable pitch contour (variation score: {features['pitch_std']:.1f}, typically higher in natural speech)")
        if features['spectral_flatness'] > 0.02:
            signals.append(f"elevated spectral flatness ({features['spectral_flatness']:.4f}), often associated with vocoder-synthesized audio")
        if features['zcr_mean'] < 0.05:
            signals.append(f"low zero-crossing rate ({features['zcr_mean']:.3f}), suggesting an unnaturally smooth waveform")
        if features['silence_ratio'] < 0.15:
            signals.append(f"minimal natural pause variation ({features['silence_ratio']*100:.1f}% silence), atypical of spontaneous human speech")
        if features['bandwidth_std'] < 200:
            signals.append("narrow spectral bandwidth variation, a common artifact of neural vocoders")

        if not signals:
            signals.append("subtle acoustic irregularities across pitch and spectral patterns consistent with synthetic generation")

        reasoning = "Flagged due to: " + "; ".join(signals[:3]) + "."

    else:
        natural_signals = []
        if features['pitch_std'] >= 20:
            natural_signals.append(f"natural pitch variation (score: {features['pitch_std']:.1f})")
        if features['silence_ratio'] >= 0.15:
            natural_signals.append("irregular natural pausing")
        if features['zcr_mean'] >= 0.05:
            natural_signals.append("organic waveform roughness typical of a human vocal tract")

        if not natural_signals:
            natural_signals.append("acoustic characteristics broadly consistent with natural human speech")

        reasoning = "Classified as real due to: " + "; ".join(natural_signals) + "."

    return reasoning

# ---------- Prediction ----------
@torch.no_grad()
def predict(filepath):
    waveform = process_audio(filepath).to(device)
    outputs = wav2vec(waveform)
    embedding = outputs.last_hidden_state.mean(dim=1)
    logit = classifier(embedding)
    prob_fake = torch.sigmoid(logit).item()

    features = extract_audio_features(filepath)

    return prob_fake, features

# ---------- GUI ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Voice Clone Detector")
        root.geometry("500x420")
        root.configure(bg="#1e1e2e")

        title = tk.Label(root, text="AI Voice Clone Detector", font=("Segoe UI", 18, "bold"),
                          bg="#1e1e2e", fg="white")
        title.pack(pady=20)

        self.select_btn = tk.Button(root, text="Select Audio File", font=("Segoe UI", 12),
                                     command=self.select_file, bg="#89b4fa", fg="black",
                                     padx=20, pady=10, relief="flat")
        self.select_btn.pack(pady=10)

        self.status_label = tk.Label(root, text="", font=("Segoe UI", 10),
                                      bg="#1e1e2e", fg="#a6adc8")
        self.status_label.pack(pady=5)

        self.result_label = tk.Label(root, text="", font=("Segoe UI", 20, "bold"),
                                      bg="#1e1e2e", fg="white")
        self.result_label.pack(pady=15)

        self.confidence_label = tk.Label(root, text="", font=("Segoe UI", 12),
                                          bg="#1e1e2e", fg="#a6adc8")
        self.confidence_label.pack(pady=5)

        self.reasoning_label = tk.Label(root, text="", font=("Segoe UI", 10),
                                         bg="#1e1e2e", fg="#a6adc8", wraplength=440, justify="left")
        self.reasoning_label.pack(pady=15, padx=20)

    def select_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.flac *.mp3 *.ogg")]
        )
        if not filepath:
            return

        self.status_label.config(text="Analyzing...")
        self.result_label.config(text="")
        self.confidence_label.config(text="")
        self.reasoning_label.config(text="")
        self.root.update()

        threading.Thread(target=self.run_analysis, args=(filepath,)).start()

    def run_analysis(self, filepath):
        try:
            prob_fake, features = predict(filepath)
            reasoning = get_reasoning(prob_fake, features)

            print(f"DEBUG: prob_fake = {prob_fake}")
            print(f"DEBUG: features = {features}")

            if prob_fake >= 0.95:
                label = "AI-GENERATED"
                color = "#f38ba8"
                confidence = prob_fake * 100
            else:
                label = "REAL / HUMAN"
                color = "#a6e3a1"
                confidence = (1 - prob_fake) * 100

            self.status_label.config(text="Analysis complete")
            self.result_label.config(text=label, fg=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            self.reasoning_label.config(text=f"Reasoning: {reasoning}")

        except Exception as e:
            self.status_label.config(text="")
            messagebox.showerror("Error", f"Failed to analyze audio:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
