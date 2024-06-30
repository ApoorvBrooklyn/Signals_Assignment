import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.io import wavfile
import sounddevice as sd

# Step 1: Read the sound file
sample_rate, sound_signal = wavfile.read('recorded_audio.wav')

# Normalize the sound signal if it's in integer format
if sound_signal.dtype != np.float32:
    sound_signal = sound_signal / np.max(np.abs(sound_signal))

# Time array for plotting
time = np.linspace(0, len(sound_signal) / sample_rate, num=len(sound_signal))

# Step 2: Add noise to the sound signal
noise = np.random.normal(0, 0.1, sound_signal.shape)  # Adjust noise level as needed
noisy_signal = sound_signal + noise

# Step 3: Define the Gaussian kernel function
def gaussian_kernel(size, sigma=1):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel

# Define kernel size and sigma
kernel_size = 50
sigma = 5
kernel = gaussian_kernel(kernel_size, sigma)

# Apply convolution to the noisy signal
filtered_signal = convolve(noisy_signal, kernel, mode='same')

# Step 4: Plot all signals
plt.figure(figsize=(12, 18))

# Original signal
plt.subplot(4, 1, 1)
plt.plot(time, sound_signal, label='Original Sound Signal')
plt.title('Original Sound Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Discrete signal (stem plot)
plt.subplot(4, 1, 2)
plt.stem(time[::5000], sound_signal[::5000], linefmt='C0-', markerfmt='C0o', basefmt='C0-')  # Plot every 1000th point
plt.title('Discrete Form of Original Sound Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend(['Discrete Sound Signal'])

# Noisy signal
plt.subplot(4, 1, 3)
plt.plot(time, noisy_signal, label='Noisy Sound Signal')
plt.title('Noisy Sound Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Filtered signal
plt.subplot(4, 1, 4)
plt.plot(time, filtered_signal, label='Filtered Sound Signal', color='orange')
plt.title('Filtered Sound Signal (After Convolution)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

wavfile.write('noisy_audio.wav', sample_rate, (noisy_signal * 32767).astype(np.int16))
wavfile.write('filtered_audio.wav', sample_rate, (filtered_signal * 32767).astype(np.int16))

# Play the noisy and filtered signals
# print("Playing noisy signal...")
# sd.play(noisy_signal, sample_rate)
# sd.wait()

# print("Playing filtered signal...")
# sd.play(filtered_signal, sample_rate)
# sd.wait()

# Step 6: Calculate correlations
noise_only = noise
corr_noise_original = np.corrcoef(noise_only, sound_signal)[0, 1]
corr_filtered_original = np.corrcoef(filtered_signal, sound_signal)[0, 1]

print(f"Correlation between noise and original signal: {corr_noise_original}")
print(f"Correlation between filtered signal and original signal: {corr_filtered_original}")


def lms_filter(noisy_signal, original_signal, mu=0.01, n=256):
    w = np.zeros(n)
    filtered_output = np.zeros_like(noisy_signal)
    for i in range(n, len(noisy_signal)):
        x = noisy_signal[i-n:i]
        y = np.dot(w, x)
        e = original_signal[i] - y
        w = w + 2 * mu * e * x
        filtered_output[i] = y
    return filtered_output

mu = 0.01  # Adaptation rate
n = 256    # Filter length
adaptive_filtered_signal = lms_filter(noisy_signal, sound_signal, mu, n)


plt.figure(figsize=(12, 6))
plt.plot(time, sound_signal, label='Original Sound Signal')
plt.plot(time, adaptive_filtered_signal, label='Adaptive Filtered Signal', color='red')
plt.title('Adaptive Filtered Sound Signal (LMS)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()


wavfile.write('adaptive_filtered_audio.wav', sample_rate, (adaptive_filtered_signal * 32767).astype(np.int16))


print("Playing adaptive filtered signal...")
# sd.play(adaptive_filtered_signal, sample_rate)
# sd.wait()

# Calculate correlation for adaptive filtered signal
corr_adaptive_filtered_original = np.corrcoef(adaptive_filtered_signal, sound_signal)[0, 1]
print(f"Correlation between adaptive filtered signal and original signal: {corr_adaptive_filtered_original}")
