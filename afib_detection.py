import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import wfdb
from tqdm import tqdm
from tqdm.auto import trange

# Load ECG data
print("Loading ECG data...")
record = wfdb.rdrecord("mit-bih/201")
fs = record.fs
ecg_signal = record.p_signal[:, 0]
duration = 240  # seconds
samples = fs * duration
ecg_raw = ecg_signal  # [:samples]

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

# Function to detect chaotic baseline (f-waves)
def detect_f_waves(ecg_signal, fs):
    print("Detecting chaotic baseline (f-waves)...")
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [5 / (fs / 2), 15 / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, ecg_signal)
    entropy = np.std(filtered)
    return entropy > 0.05  # threshold may need tuning

# AFib detection logic
print("Running AFib detection logic...")
irregular_rr = rmssd > 0.08 or np.std(rr_intervals) > 0.12
no_consistent_p = not consistent_p_waves_detection(p_waves, rpeaks)
no_pr_consistency = not pr_interval_check(p_waves, q_peaks)
chaotic_baseline = detect_f_waves(cleaned, fs)

afib_detected = irregular_rr and no_consistent_p and no_pr_consistency and chaotic_baseline

afib_score = (
    0.35 * irregular_rr +
    0.30 * no_consistent_p +
    0.20 * no_pr_consistency +
    0.15 * chaotic_baseline
)

# Results
print("\n--- AFib Detection Results ---")
print(f"AFib detected: {afib_detected}")
print(f"Irregular RR detected: {irregular_rr}")
print(f"No Consistent P-waves detected: {no_consistent_p}")
print(f"No PR interval consistency detected: {no_pr_consistency}")
print(f"Chaotic baseline detected: {chaotic_baseline}")
print(f"AFib likelyhood score: {afib_score:.2f}")
if afib_score >= 0.75:
    print("High likelihood of AFib.")
elif afib_score >= 0.50:
    print("Moderate likelihood of AFib.")
else:
    print("Low likelihood of AFib.")