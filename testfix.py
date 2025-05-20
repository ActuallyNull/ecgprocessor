import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import neurokit2 as nk
import pandas as pd

def bandpass_filter(signal, fs, lowcut=0.5, highcut=40, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

record = wfdb.rdrecord("mit-bih/101") 
fs = record.fs # sampling frequency, 360hz -> 360 samples per second
p_signal = record.p_signal[:, 0] # p_signal is a 2D array, we take the first column
sig_name = record.sig_name[0] # signal name, mlii -> modified lead II, takes right arm and left leg electrodes
# usually used in arrhythmia detection
record_name = record.record_name 
record_comments = record.comments # comments in the record
duration = 10 #seconds
x = find_peaks(p_signal, distance=fs*1)
time_axis = np.linspace(0, duration, len(p_signal[0:fs*duration])) # time axis for 2 seconds
plt.plot(time_axis, bandpass_filter(p_signal[0:fs*duration], fs))

plt.title('ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
samples = fs * duration
print(len(p_signal[:samples]))
print(sig_name)
print(record_name)
print(record_comments)
print(x[0])

ecg_signal = p_signal[:samples]

_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs, correct_artifacts=True)
# plot the r peaks
rpeak_times = rpeaks['ECG_R_Peaks'] / fs  # Convert sample indices to time in seconds

corrected_rpeaks = []
window = 10  # samples around detected peak

for peak in rpeaks['ECG_R_Peaks']:
    start = max(0, peak - window)
    end = min(len(ecg_signal), peak + window)
    true_peak = start + np.argmax(ecg_signal[start:end])
    corrected_rpeaks.append(true_peak)

# Now plot with corrected peaks
corrected_rpeaks = np.array(corrected_rpeaks)
plt.plot(corrected_rpeaks / fs, ecg_signal[corrected_rpeaks], 'rx')



plt.show()