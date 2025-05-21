import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# --- Load and prepare ECG ---
ecg_data = wfdb.rdrecord("mit-bih/106")
fs = ecg_data.fs
ecg_signal = ecg_data.p_signal[:, 0]
duration = 240  # seconds
samples = fs * duration
ecg_raw = ecg_signal[:samples]

# --- Process ECG ---
signals, info = nk.ecg_process(ecg_raw, sampling_rate=fs)
rpeaks = info["ECG_R_Peaks"]
heart_rate = signals["ECG_Rate"]
avg_heart_rate = heart_rate.mean()

# --- Detection functions ---
def tachycardia_detection(hr_data):
    return [index for index, item in enumerate(hr_data) if item > 100]

def bradycardia_detection(hr_data):
    return [index for index, item in enumerate(hr_data) if item < 60]

# --- Apply detections ---
tachy_samples = tachycardia_detection(heart_rate)
brady_samples = bradycardia_detection(heart_rate)

# --- Print summary ---
if avg_heart_rate > 100:
    print("Tachycardia detected")
elif avg_heart_rate < 60:
    print("Bradycardia detected")
else:
    print("Normal heart rate detected")

print(f"Average Heart Rate over {duration} seconds: {avg_heart_rate:.2f} bpm")
print(f"Amount of Tachycardia samples: {len(tachy_samples)}, estimated time in Tachycardia: {len(tachy_samples)/fs:.2f} seconds")
print(f"Amount of Bradycardia samples: {len(brady_samples)}, estimated time in Bradycardia: {len(brady_samples)/fs:.2f} seconds")

# --- Plotting ---
time = np.linspace(0, len(heart_rate) / fs, len(heart_rate))

plt.figure(figsize=(12, 6))
plt.plot(time, heart_rate, label='Heart Rate', color='black')

# Overlay tachycardia in red
plt.scatter([time[i] for i in tachy_samples], [heart_rate[i] for i in tachy_samples],
            color='red', label='Tachycardia (>100 bpm)', s=15)

# Overlay bradycardia in blue
plt.scatter([time[i] for i in brady_samples], [heart_rate[i] for i in brady_samples],
            color='blue', label='Bradycardia (<60 bpm)', s=15)

plt.xlabel("Time (s)")
plt.ylabel("Heart Rate (bpm)")
plt.title("Heart Rate with Tachycardia and Bradycardia Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()