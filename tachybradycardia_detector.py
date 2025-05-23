import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import wfdb

# Load ECG data
record = wfdb.rdrecord("mit-bih/101")
fs = record.fs
ecg_signal = record.p_signal[:, 0]
duration = 240  # seconds
samples = fs * duration
ecg_raw = ecg_signal[:samples]

# Process ECG
delineate_signals, delineate_info = nk.ecg_delineate(ecg_raw, sampling_rate=fs, method="dwt", show=False)
p_waves = delineate_info["ECG_P_Peaks"]
signals, info = nk.ecg_process(ecg_raw, sampling_rate=fs)
rpeaks = info["ECG_R_Peaks"]
cleaned = signals["ECG_Clean"]
avg_heart_rate = signals["ECG_Rate"].mean()
rr_intervals = np.diff(rpeaks) / fs  # Convert to seconds

# Measuring RMSSD
rr_dif = np.diff(rr_intervals)
rmssd = np.sqrt(np.mean(rr_dif**2))
print(f"RMSSD: {rmssd:.2f} seconds")

# Simple AFib heuristic
irregularity_threshold = 0.12
rmssd_threshold = 0.08

def consistent_p_waves_detection(p_waves, rpeaks):
    if p_waves is not None:
        
        present_p_waves = np.count_nonzero(~np.isnan(p_waves))
        ratio = present_p_waves / len(rpeaks)
        if ratio > 0.8:
            print("Likely Sinus Rhythm")
            return True
        else:
            print("Likely Non Sinus Rhythm")
            return False
# tachycardia and bradycardia basic detection

def find_consecutive_ranges(indices):
    """Group consecutive indices into ranges."""
    if not indices:
        return []
    ranges = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))
    return ranges

def tachycardia_detection(hr_data):
    tachycardia_sample = []
    for index, item in enumerate(hr_data.head(samples)):
        if item > 100:
            tachycardia_sample.append(index)
    print("Amount of tachycardia samples:", len(tachycardia_sample))

    if tachycardia_sample:
        ranges = find_consecutive_ranges(tachycardia_sample)
        for start, end in ranges:
            print(f"Tachycardia from sample {start} to {end} (duration: {(end-start+1)/fs:.2f} s)")
        print("Is Sinus Tachycardia:", consistent_p_waves_detection(p_waves=p_waves, rpeaks=rpeaks))

def bradycardia_detection(hr_data):
    bradycardia_sample = []
    for index, item in enumerate(hr_data.head(samples)):
        if item < 60:
            bradycardia_sample.append(index)
    print("Amount of bradycardia samples:", len(bradycardia_sample))

    if bradycardia_sample:
        ranges = find_consecutive_ranges(bradycardia_sample)
        for start, end in ranges:
            print(f"Bradycardia from sample {start} to {end} (duration: {(end-start+1)/fs:.2f} s)")
        print("Is Sinus Bradycardia:", consistent_p_waves_detection(p_waves=p_waves, rpeaks=rpeaks))

tachycardia_detection(signals["ECG_Rate"])
bradycardia_detection(signals["ECG_Rate"])

if rmssd > rmssd_threshold:                                     # AFib detection based on RMSSD
    print("Potential AFib detected (high RMSSD)")               #
if np.mean(rr_intervals) > irregularity_threshold:              # AFib detection based on irregular RR intervals
    print("Potential AFib detected (irregular RR intervals)")   #

print(f"Average Heart Rate over {duration} seconds: {avg_heart_rate:.2f} bpm")