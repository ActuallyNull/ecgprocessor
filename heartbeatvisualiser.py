import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import wfdb

# Load ECG data
record = wfdb.rdrecord("mit-bih/106")
fs = record.fs
ecg_signal = record.p_signal[:, 0]
duration = 240  # seconds
samples = fs * duration
ecg_raw = ecg_signal[:samples]

# Process ECG
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

# --------- PLOT 1: Scrollable ECG with paper grid ---------

window_duration = 5  # seconds to display at a time
window_samples = fs * window_duration
fig, ax = plt.subplots(figsize=(12, 4))
plt.subplots_adjust(bottom=0.25)

def draw_segment(start):
    ax.clear()
    end = start + window_samples
    time = np.linspace(start/fs, end/fs, window_samples)
    ax.plot(time, cleaned[start:end], label='ECG')
    peaks = rpeaks[(rpeaks >= start) & (rpeaks < end)]
    ax.scatter(peaks/fs, cleaned[peaks], color='red', label='R-peaks', zorder=5)
    
    small_step = 0.04
    big_step = 0.2
    for x in np.arange(start/fs, end/fs, small_step):
        ax.axvline(x=x, color='lightgrey', linewidth=0.5)
    for x in np.arange(start/fs, end/fs, big_step):
        ax.axvline(x=x, color='grey', linewidth=1)
    for y in np.arange(-1, 2, 0.1):
        ax.axhline(y=y, color='lightgrey', linewidth=0.5)
    for y in np.arange(-1, 2, 0.5):
        ax.axhline(y=y, color='grey', linewidth=1)

    ax.set_title("ECG with R-peaks and Grid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(start/fs, end/fs)
    ax.legend()
    fig.canvas.draw_idle()

def update(val):
    draw_segment(int(val * fs))

slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(slider_ax, 'Scroll', 0, duration - window_duration, valinit=0, valstep=1)
slider.on_changed(update)

draw_segment(0)

# --------- PLOT 2: Stacked heartbeat cycles ---------
nk.ecg_segment(cleaned, rpeaks=rpeaks, sampling_rate=fs, show=True)

# tachycardia and bradycardia basic detection

def tachycardia_detection(hr_data):
    tachycardia_sample = []
    for index, item in enumerate(hr_data.head(samples)):
        if item > 100: 
            print(f"Tachycardia detected at sample: {index}, heart rate: {item:.2f}")
            tachycardia_sample.append(index)
    print("Amount of tachycardia samples:", len(tachycardia_sample))
            

def bradycardia_detection(hr_data):
    bradycardia_sample = []
    for index, item in enumerate(hr_data.head(samples)):
        if item < 60:
            print(f"Bradycardia detected at sample: {index}, heart rate: {item:.2f}")
            bradycardia_sample.append(index)
    print("Amount of bradycardia samples:", len(bradycardia_sample))

if avg_heart_rate > 100:
    print("Tachycardia detected")
    tachycardia_detection(signals["ECG_Rate"])
elif avg_heart_rate < 60:
    print("Bradycardia detected")
    bradycardia_detection(signals["ECG_Rate"])
else:
    print("Normal heart rate detected")

if rmssd > rmssd_threshold:
    print("Potential AFib detected (high RMSSD)")
if np.mean(rr_intervals) > irregularity_threshold:
    print("Potential AFib detected (irregular RR intervals)")

print(f"Average Heart Rate over {duration} seconds: {avg_heart_rate:.2f} bpm")

plt.show()
