import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import wfdb

# --- Load ECG file ---
record = wfdb.rdrecord("mit-bih/101")  # MIT-BIH example
fs = record.fs
signal = record.p_signal[:, 0]

# Optional: limit to first 10 seconds
duration_sec = 2
samples = fs * duration_sec
signal = signal[:samples]

# --- Bandpass filter ---
def bandpass_filter(signal, fs, low=0.5, high=40, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

filtered = bandpass_filter(signal, fs)
t = np.linspace(0, len(filtered)/fs, len(filtered))

# --- R-peak detection ---
peaks, props = find_peaks(filtered, distance=fs*0.6, height=0.6)
rr_intervals = np.diff(peaks) / fs
heart_rate = 60 / rr_intervals if len(rr_intervals) > 0 else []

# --- Detect Arrhythmias ---
arrhythmia_flags = []

for i, rr in enumerate(rr_intervals):
    bpm = 60 / rr

    if bpm > 100:
        arrhythmia_flags.append(("Tachycardia", t[peaks[i]]))
    elif bpm < 60:
        arrhythmia_flags.append(("Bradycardia", t[peaks[i]]))
    elif i > 0:
        delta_rr = abs(rr - rr_intervals[i-1])
        if delta_rr > 0.2:
            arrhythmia_flags.append(("Irregular RR", t[peaks[i]]))

# Missed beat detection (gap > 2× median RR)
if len(rr_intervals) > 0:
    median_rr = np.median(rr_intervals)
    for i, rr in enumerate(rr_intervals):
        if rr > 2 * median_rr:
            arrhythmia_flags.append(("Possible Missed Beat", t[peaks[i]]))

# --- Plot ECG with arrhythmia markers ---
plt.figure(figsize=(15, 5))
plt.plot(t, filtered, label='Filtered ECG')
plt.plot(t[peaks], filtered[peaks], 'ro', label='R-peaks')

# Mark arrhythmia points
for label, ts in arrhythmia_flags:
    plt.axvline(ts, color='orange', linestyle='--', alpha=0.7)
    plt.text(ts, max(filtered), label, rotation=90, color='orange', fontsize=8, va='bottom')

plt.title("ECG Signal with Arrhythmia Detection")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()

# --- Print Summary ---
print(f"Detected {len(peaks)} beats.")
if len(heart_rate) > 0:
    print(f"Average heart rate: {np.mean(heart_rate):.1f} bpm")
print("Arrhythmia Events:")
for label, ts in arrhythmia_flags:
    print(f"{label} at {ts:.2f} seconds")
