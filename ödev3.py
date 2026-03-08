import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# -------------------------------
# AUDIO LOAD
# -------------------------------

signal, sr = librosa.load("speech.wav", sr=None)

# normalize
signal = signal / np.max(np.abs(signal))

# -------------------------------
# FRAME PARAMETERS
# -------------------------------

frame_size = int(0.02 * sr)      # 20 ms
hop_size = int(0.01 * sr)        # 50% overlap

frames = []

for i in range(0, len(signal) - frame_size, hop_size):
    frames.append(signal[i:i+frame_size])

frames = np.array(frames)

# -------------------------------
# HAMMING WINDOW
# -------------------------------

window = np.hamming(frame_size)
frames = frames * window

# -------------------------------
# ENERGY CALCULATION
# -------------------------------

energy = np.sum(frames**2, axis=1)

# -------------------------------
# ZCR CALCULATION
# -------------------------------

zcr = []

for frame in frames:
    crossings = np.sum(np.abs(np.diff(np.sign(frame))))
    zcr.append(crossings / len(frame))

zcr = np.array(zcr)

# -------------------------------
# NOISE THRESHOLD (first 200 ms)
# -------------------------------

noise_frames = int(0.2 / 0.01)

noise_threshold = np.mean(energy[:noise_frames]) * 2

# -------------------------------
# VAD DECISION
# -------------------------------

speech_mask = energy > noise_threshold

# Hangover (speech continuity)
for i in range(3, len(speech_mask)):
    if speech_mask[i-3:i].any():
        speech_mask[i] = True

# -------------------------------
# VOICED / UNVOICED
# -------------------------------

voiced = np.zeros(len(frames))
unvoiced = np.zeros(len(frames))

for i in range(len(frames)):

    if speech_mask[i]:

        if zcr[i] < 0.1:
            voiced[i] = 1
        else:
            unvoiced[i] = 1

# -------------------------------
# SPEECH SIGNAL CREATION
# -------------------------------

speech_signal = []

for i, flag in enumerate(speech_mask):

    if flag:
        start = i * hop_size
        speech_signal.extend(signal[start:start+frame_size])

speech_signal = np.array(speech_signal)

# save output
sf.write("speech_only.wav", speech_signal, sr)

# -------------------------------
# COMPRESSION CALCULATION
# -------------------------------

original_duration = len(signal) / sr
speech_duration = len(speech_signal) / sr

compression = (original_duration - speech_duration) / original_duration * 100

print("Original Duration:", original_duration)
print("Speech Duration:", speech_duration)
print("Compression:", compression,"%")

# -------------------------------
# VISUALIZATION
# -------------------------------

time = np.arange(len(signal)) / sr

plt.figure(figsize=(12,10))

# Original Signal
plt.subplot(4,1,1)
plt.plot(time, signal)
plt.title("Original Audio Signal")

# Energy
plt.subplot(4,1,2)
plt.plot(energy)
plt.axhline(noise_threshold,color='red')
plt.title("Short Time Energy")

# ZCR
plt.subplot(4,1,3)
plt.plot(zcr)
plt.title("Zero Crossing Rate")

# Voiced / Unvoiced
plt.subplot(4,1,4)

plt.plot(voiced, label="Voiced", color="green")
plt.plot(unvoiced, label="Unvoiced", color="orange")

plt.legend()
plt.title("Voiced vs Unvoiced")

plt.tight_layout()
plt.show()