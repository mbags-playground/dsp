import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define parameters
sample_rate = 44100  # Hz
duration = 5  # seconds
window_size = 1024  # Adjust window size as needed
overlap = 0.75  # Overlap percentage for windows

# Record audio
print("Recording audio...")
recorded_audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
sd.wait()

# Convert audio to NumPy array
audio_data = recorded_audio.flatten()

# Preprocess audio
windowed_data = []
for i in range(0, len(audio_data) - window_size + 1, int(window_size * overlap)):
    window = audio_data[i:i + window_size]
    window = (window - np.mean(window)) / np.std(window)  # Normalize
    windowed_data.append(window)

# Apply STFT
stft = librosa.stft(audio_data, n_fft=window_size, hop_length=int(window_size * overlap))
stft_magnitude = np.abs(stft)

# Design noise removal filter (example: high-pass filter)
filter_cutoff = 200  # Adjust cutoff frequency as needed
filter_bank = np.zeros_like(stft_magnitude)
filter_bank[filter_cutoff:, :] = 1.0

# Apply filter to STFT
filtered_stft = stft_magnitude * filter_bank

# Inverse STFT to obtain filtered audio
filtered_audio = librosa.istft(filtered_stft, hop_length=int(window_size * overlap))

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(121)
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio_data), ref=np.max),
                          sr=sample_rate, y_axis='mel', x_axis='time')
plt.title("Original Spectrogram")
plt.subplot(122)
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(filtered_audio), ref=np.max),
                          sr=sample_rate, y_axis='mel', x_axis='time')
plt.title("Filtered Spectrogram")
plt.tight_layout()
plt.show()

# Playback and evaluation (optional)
# ...
# Playback original and filtered audio
print("Playing original recording...")
sd.play(audio_data, samplerate=sample_rate)
sd.wait()

print("Playing filtered audio...")
filtered_audio = filtered_audio.astype(np.float32)  # Convert to supported audio data type
sd.play(filtered_audio, samplerate=sample_rate)
sd.wait()

# Evaluation: Calculate Signal-to-Noise Ratio (SNR)
snr_original = librosa.feature.perceptual_snr(audio_data, noise_audio=None)  # Assuming no isolated noise available
snr_filtered = librosa.feature.perceptual_snr(filtered_audio, noise_audio=None)
print(f"SNR improvement: {snr_filtered - snr_original:.2f} dB")

# Additional evaluation metrics (optional)
# - Perceptual Evaluation of Speech Quality (PESQ)
# - Segmental SNR

