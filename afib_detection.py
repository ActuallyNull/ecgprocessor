import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import wfdb
from tqdm import tqdm
from tqdm.auto import trange
from scipy.signal import butter, filtfilt, welch

# Load ECG data
print("Loading ECG data...")
#record = wfdb.rdrecord("mit-bih/201")
#fs = record.fs
#ecg_signal = record.p_signal[:, 0]
fs = 360
duration = 30  # seconds
ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=100, noise=0.00)
samples = fs * duration
ecg_raw = ecg_signal#[:samples]

# Process ECG
print("Processing ECG signal...")
signals, info = nk.ecg_process(ecg_raw, sampling_rate=fs)
rpeaks = info["ECG_R_Peaks"]
cleaned = signals["ECG_Clean"]

# Delineate ECG for P-waves and PR intervals
print("Delineating ECG (P/QRS/T peaks)...")
delineate_signals, delineate_info = nk.ecg_delineate(ecg_raw, sampling_rate=fs, method="dwt", show=False)
p_waves = delineate_info["ECG_P_Peaks"]
q_peaks = delineate_info["ECG_Q_Peaks"]

# Compute RR intervals
print("Computing RR intervals...")
rr_intervals = np.diff(rpeaks) / fs
rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))

# Function to check P-wave consistency
def consistent_p_waves_detection(p_peaks, rpeaks):
    print("Checking P-wave consistency...")
    if p_peaks is not None:
        present_p_waves = np.count_nonzero(~np.isnan(p_peaks))
        ratio = present_p_waves / len(rpeaks)
        return ratio > 0.8  # 80% threshold
    return False

# Function to check PR interval consistency
def pr_interval_check(p_peaks, q_peaks):
    print("Checking PR interval consistency...")
    if p_peaks is None or q_peaks is None:
        return False
    p_peaks = np.array(p_peaks)
    q_peaks = np.array(q_peaks)
    intervals = []
    for p in tqdm(p_peaks, desc="Calculating PR intervals"):
        closest_q = q_peaks[q_peaks > p]
        if len(closest_q) > 0:
            intervals.append((closest_q[0] - p) / fs)
    if len(intervals) == 0:
        return False
    std_pr = np.std(intervals)
    return std_pr < 0.05  # 50 ms standard deviation threshold

def detect_chaotic_baseline(ecg_signal, fs):
    print("Detecting chaotic baseline...")
    b, a = butter(4, [5 / (fs / 2), 15 / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, ecg_signal)
    std_dev = np.std(filtered)
    print(f"Filtered signal std (entropy) {std_dev:.2f}")
    return std_dev > 0.1857  # Tweak threshold if needed

def detect_f_waves(ecg_signal, fs):
    print("Detecting f-waves...")
    f, Pxx = welch(ecg_signal, fs=fs, nperseg=fs*2)
    f_wave_band = (f >= 5.5) & (f <= 10)  # 330–600 bpm
    f_wave_power = np.sum(Pxx[f_wave_band])
    total_power = np.sum(Pxx)
    ratio = f_wave_power / total_power
    print("F-wave power ratio:", ratio)
    return ratio > 0.38  # Threshold may require tuning

# AFib detection logic
print("Running AFib detection logic...")
irregular_rr = rmssd > 0.08 or np.std(rr_intervals) > 0.12
no_consistent_p = not consistent_p_waves_detection(p_waves, rpeaks)
no_pr_consistency = not pr_interval_check(p_waves, q_peaks)
chaotic_baseline = detect_chaotic_baseline(cleaned, fs)
f_waves_detected = detect_f_waves(cleaned, fs)

afib_detected = irregular_rr and no_consistent_p and no_pr_consistency and chaotic_baseline

afib_score = (
    0.35 * irregular_rr +
    0.30 * no_consistent_p +
    0.20 * no_pr_consistency +
    0.10 * chaotic_baseline +
    0.05 * f_waves_detected
)

# Results
print("\n--- AFib Detection Results ---")
print(f"Definite* AFib detected: {afib_detected}")
print(f"Irregular RR detected: {irregular_rr}")
print(f"No Consistent P-waves detected: {no_consistent_p}")
print(f"No PR interval consistency detected: {no_pr_consistency}")
print(f"Chaotic baseline detected: {chaotic_baseline}")
print(f"Possible F-waves detected: {f_waves_detected}")
print(f"AFib likelihood score: {afib_score:.2f}")

if afib_score >= 0.75:
    print("High likelihood of AFib.")
elif afib_score >= 0.50:
    print("Moderate likelihood of AFib.")
else:
    print("Low likelihood of AFib.")

#nk.signal_plot(ecg_signal, title="Processed ECG Signals", figsize=(12, 8))
#plt.psd(ecg_signal, Fs=fs, NFFT=fs*2)
#plt.title("Power Spectral Density of Simulated ECG")
#plt.show()

# Threshhold Tests:
#   -60BPM: entropy=0.14, f-wave power ratio=0.34
#   -70BPM, 30sec: entropy=0.15, f-wave power ratio=0.29
#   -80BPM, 30sec: entropy=0.16, f-wave power ratio=0.34
#   -90BPM, 30sec: entropy=0.16, f-wave power ratio=0.34
#   -100BPM, 30sec: entropy=0.16, f-wave power ratio=0.35