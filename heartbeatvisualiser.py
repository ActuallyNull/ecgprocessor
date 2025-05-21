import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# Load ECG data
record = wfdb.rdrecord("mit-bih/106")
fs = record.fs
ecg_signal = record.p_signal[:, 0]
duration = 5  # seconds
samples = fs * duration
ecg_raw = ecg_signal[:samples]

# Process ECG
signals, info = nk.ecg_process(ecg_raw, sampling_rate=fs)
rpeaks = info["ECG_R_Peaks"]
cleaned = signals["ECG_Clean"]
avg_heart_rate = signals["ECG_Rate"].mean()

# --------- PLOT 1: 5-second ECG with paper grid ---------
plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, duration, samples), cleaned, label='ECG')
plt.scatter(rpeaks/fs, cleaned[rpeaks], color='red', label='R-peaks', zorder=5)

# ECG paper grid (0.04s small squares, 0.2s large)
small_step = 0.04
big_step = 0.2
for x in np.arange(0, duration, small_step):
    plt.axvline(x=x, color='lightgrey', linewidth=0.5)
for x in np.arange(0, duration, big_step):
    plt.axvline(x=x, color='grey', linewidth=1)
for y in np.arange(-1, 2, 0.1):  # Small squares vertically (adjust y limits as needed)
    plt.axhline(y=y, color='lightgrey', linewidth=0.5)
for y in np.arange(-1, 2, 0.5):
    plt.axhline(y=y, color='grey', linewidth=1)

plt.title("ECG with R-peaks and Grid")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, duration)
plt.legend()
plt.tight_layout()

# --------- PLOT 2: Stacked heartbeat cycles ---------
nk.ecg_segment(cleaned, rpeaks=rpeaks, sampling_rate=fs, show=True)


#print(signals[["ECG_Rate"]].head(10))  # Display first 10 heart rates

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

print(f"Average Heart Rate over {duration} seconds: {avg_heart_rate:.2f} bpm")
plt.show()