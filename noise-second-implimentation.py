import sounddevice as sd
from sklearn.decomposition import FastICA
import librosa
import matplotlib.pyplot as plt
import numpy as np

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
print("Audio recorded!")
audio_data = recorded_audio.flatten()

# Preprocess audio for ICA
windowed_data = []
for i in range(0, len(audio_data) - window_size + 1, int(window_size * overlap)):
    window = audio_data[i:i + window_size]
    # Normalize window (zero mean, unit variance)
    window = (window - np.mean(window)) / np.std(window)
    windowed_data.append(window)

# Perform ICA decomposition
num_components = 2  # Assumes speech and noise sources
ica = FastICA(n_components=num_components)
ica_components = ica.fit_transform(windowed_data)

# Source identification (speech vs. noise)
speech_component = None
noise_component = None
for i in range(num_components):
    stft = librosa.stft(ica_components[i, :])  # Compute STFT first
    mel_spectrogram = librosa.feature.melspectrogram(S=stft)  # Pass STFT to melspectrogram   # Analyze spectral features to identify speech component
    # (Adapt criteria based on your data and specific requirements)
    if (
        np.mean(mel_spectrogram[0:50, :]) > 0.05  # Check for dominant speech frequencies
        and np.mean(mel_spectrogram[500:, :]) < 0.01  # Check for less energy in high-frequency noise range
    ):
        speech_component = i
        break
    else:
        noise_component = i

# Reconstruct speech signal
if speech_component is not None:
    reconstructed_speech = ica_components[speech_component, :]
    # Invert windowing and combine frames (assuming a Hanning window)
    restored_frames = []
    hanning_window = np.hanning(window_size)
    for window in reconstructed_speech:
        inverse_windowed_frame = window * hanning_window
        restored_frames.append(inverse_windowed_frame)
    restored_signal = np.concatenate(restored_frames)
else:
    restored_signal=audio_data;
# Play audio samples and compare
print("Playing original recording...")
sd.play(audio_data, samplerate=sample_rate)
sd.wait()

print("Playing noisy speech (isolated noise)...")
sd.play(ica_components[noise_component, :], samplerate=sample_rate)
sd.wait()

print("Playing restored speech...")
sd.play(restored_signal, samplerate=sample_rate)
sd.wait()

# Visualization and evaluation
plt.figure(figsize=(15, 5))
plt.subplot(131)
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(audio_data), ref=np.max),
                         sr=sample_rate, y_axis='mel', x_axis='time')
plt.title("Original Spectrogram")
plt.subplot(132)
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(ica_components[noise_component, :]), ref=np.max),
                         sr=sample_rate, y_axis='mel', x_axis='time')
plt.title("Noise Spectrogram")
plt.subplot(133)
librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(restored_signal), ref=np.max),
                         sr=sample_rate, y_axis='mel', x_axis='time')
plt.title("Restored Speech Spectrogram")
plt.tight_layout()
plt.show()

# Frequency Response: Plot ICA filter magnitude
ica_filter_magnitude = np.abs(ica.components_)
plt.plot(np.arange(len(ica_filter_magnitude[0])), ica_filter_magnitude[0], label='Speech', linewidth=2)
plt.plot(np.arange(len(ica_filter_magnitude[1])), ica_filter_magnitude[1], label='Noise', linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.show()

#snr_original_noise = ...
#snr_restored_noise = ...
#print(f"SNR improvement: {snr_original_noise - snr_restored_noise:.2f} dB")

