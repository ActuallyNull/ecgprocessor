import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import neurokit2 as nk

# --- Function: Bandpass filter the ECG signal ---
def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

# --- Load ECG record ---
record = wfdb.rdrecord("mit-bih/101")
fs = record.fs  # Sampling frequency (360 Hz)
p_signal = record.p_signal[:, 0]  # Uses MLII (Lead II)
duration = 240  # Duration in seconds to process
samples = fs * duration
ecg_raw = p_signal[:samples] # Limit to the first 5 seconds

# --- Filter ECG signal ---
ecg_filtered = bandpass_filter(ecg_raw, fs)

# --- Detect R-peaks ---
_, rpeaks = nk.ecg_peaks(ecg_filtered, sampling_rate=fs, correct_artifacts=True)

# --- Delineate ECG: detect P, Q, S, T waves ---
signals, waves = nk.ecg_delineate(ecg_filtered,
                                  rpeaks=rpeaks['ECG_R_Peaks'],
                                  sampling_rate=fs,
                                  method="dwt",
                                  show=True,
                                  show_type="peaks",)

plot = nk.events_plot([waves['ECG_T_Peaks'][:5], 
                       waves['ECG_P_Peaks'][:5],
                       waves['ECG_Q_Peaks'][:5],
                       waves['ECG_S_Peaks'][:5]], ecg_filtered[:5 * fs])

# --- Plot ECG with PQRST annotations ---
plt.figure(figsize=(15, 5))
plt.plot(ecg_filtered, label='Filtered ECG', color='black', linewidth=1)

# Mark R-peaks
r_locs = rpeaks['ECG_R_Peaks']
plt.plot(r_locs, ecg_filtered[r_locs], 'ro', label='R-peaks')

# Define wave types and colours
wave_types = {
    'ECG_P_Peaks': 'orange',
    'ECG_Q_Peaks': 'blue',
    'ECG_S_Peaks': 'purple',
    'ECG_T_Peaks': 'green'
}

# Mark each wave type safely
for wave, color in wave_types.items():
    if isinstance(waves, dict) and wave in waves and waves[wave] is not None:
        locs = np.array(waves[wave])
        locs = locs[~np.isnan(locs)].astype(int)
        plt.plot(locs, ecg_filtered[locs], 'o', color=color, label=wave)

plt.title(f'ECG Signal with Detected PQRST Waves (First {duration} Seconds)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(visible=True, which='major')
plt.grid(visible=True, which='minor')
plt.tight_layout()
plt.show()
