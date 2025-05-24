import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# --- Load and prepare ECG ---
ecg_data = wfdb.rdrecord("mit-bih/201")
fs = ecg_data.fs
ecg_signal = ecg_data.p_signal[:, 0]
duration = 240  # seconds
samples = fs * duration
ecg_raw = ecg_signal#[:samples]

# --- Process ECG ---
signals, info = nk.ecg_process(ecg_raw, sampling_rate=fs)
r_peaks = info["ECG_R_Peaks"]

# --- Determining average RR interval ---
rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds

print(f"Average RR interval: {np.mean(rr_intervals)*1000:.2f} ms")
print(f"Standard deviation of RR intervals: {np.std(rr_intervals)*1000:.2f} ms")
print(f"Minimum RR interval: {np.min(rr_intervals)*1000:.2f} ms")
print(f"Maximum RR interval: {np.max(rr_intervals)*1000:.2f} ms") 
print(f"Average heart rate: {60 / np.mean(rr_intervals):.2f} bpm")