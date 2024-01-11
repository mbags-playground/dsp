import sounddevice as sd  # Import sounddevice for audio playback
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np

# Define sample rate and duration
sample_rate = 44100  # Hz
duration = 5  # seconds

# Record audio
print("Recording audio...")
recorded_audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
sd.wait()

# Stop recording and convert audio to NumPy array
print("Audio recorded!")
audio_data = recorded_audio.flatten()
plt.plot(audio_data)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Unprocessed video sound")
plt.show()
# Play the original audio
sd.play(audio_data, samplerate=sample_rate)
sd.wait()

# Define sample size for chopping
sample_size = 1024  # Adjust as needed

# Chop audio into samples
chopped_samples = []
for i in range(0, len(audio_data), sample_size):
    chopped_samples.append(audio_data[i:i + sample_size])

# Perform FFT on each sample and apply a filter (example: low-pass)
processed_samples = []
for sample in chopped_samples:
    fft_sample = np.fft.fft(sample)
    # Example filter: Apply a low-pass filter with cutoff frequency of 3000 Hz
    fft_sample[1000:] = 0  # Set frequencies above 3000 Hz to zero
    processed_sample = np.fft.ifft(fft_sample).real  # Reconstruct the signal
    processed_samples.append(processed_sample)

# Combine processed samples back into a continuous signal
processed_audio = np.concatenate(processed_samples)
plt.plot(processed_audio)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Processed audio sound")
plt.show()
# Play the processed audio

# Further analysis or processing can be done using "processed_audio"

