import numpy as np
from scipy.io import wavfile
import os

# -----------------------------
# PARAMETERS
# -----------------------------
fs = 44100
duration = 0.04
amplitude = 0.5

# -----------------------------
# CHARACTER SET
# -----------------------------
chars = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ ")

low_freqs  = [500, 600, 700, 800, 900]
high_freqs = [1200, 1300, 1400, 1500, 1600, 1700]

freq_map = {}
index = 0
for f_low in low_freqs:
    for f_high in high_freqs:
        if index < len(chars):
            freq_map[chars[index]] = (f_low, f_high)
            index += 1

reverse_map = {v: k for k, v in freq_map.items()}

# -----------------------------
# ENCODING
# -----------------------------
def encode_text(text):
    full_signal = np.array([])

    for ch in text.upper():
        if ch in freq_map:
            f1, f2 = freq_map[ch]
            t = np.linspace(0, duration, int(fs*duration), endpoint=False)

            tone = amplitude * (
                np.sin(2*np.pi*f1*t) +
                np.sin(2*np.pi*f2*t)
            )

            tone *= np.hamming(len(tone))
            full_signal = np.concatenate((full_signal, tone))

    return full_signal

# -----------------------------
# DECODING
# -----------------------------
def decode_signal(signal):
    samples_per_char = int(fs * duration)
    decoded_text = ""

    for i in range(0, len(signal), samples_per_char):
        chunk = signal[i:i+samples_per_char]

        if len(chunk) < samples_per_char:
            continue

        fft_vals = np.abs(np.fft.fft(chunk))
        freqs = np.fft.fftfreq(len(chunk), 1/fs)

        positive = freqs > 0
        freqs = freqs[positive]
        fft_vals = fft_vals[positive]

        peak_indices = np.argsort(fft_vals)[-2:]
        detected = sorted(freqs[peak_indices])

        detected_low = min(low_freqs, key=lambda x: abs(x-detected[0]))
        detected_high = min(high_freqs, key=lambda x: abs(x-detected[1]))

        decoded_text += reverse_map.get((detected_low, detected_high), "?")

    return decoded_text

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    text = input("Enter your text: ")
    print("Original:", text)

    signal = encode_text(text)

    if len(signal) == 0:
        print("Sinyal boş! Karakterleri kontrol et.")
        exit()

    signal_int16 = (signal * 32767).astype(np.int16)

    # 📌 DOSYAYI KOD DOSYASININ BULUNDUĞU YERE KAYDET
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "dtmf_output.wav")

    wavfile.write(output_path, fs, signal_int16)

    print("WAV kaydedildi:", output_path)

    # Test decode
    fs_read, recorded = wavfile.read(output_path)
    decoded = decode_signal(recorded.astype(float)/32767)

    print("Decoded:", decoded)