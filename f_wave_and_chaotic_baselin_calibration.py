import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, welch
from tqdm import tqdm

def detect_chaotic_baseline(ecg_signal, fs):
    b, a = butter(4, [5 / (fs / 2), 15 / (fs / 2)], btype='band')
    filtered = filtfilt(b, a, ecg_signal)
    return np.std(filtered)

def detect_f_waves(ecg_signal, fs):
    f, Pxx = welch(ecg_signal, fs=fs, nperseg=fs * 2)
    f_band = (f >= 5.5) & (f <= 10)
    return np.sum(Pxx[f_band]) / np.sum(Pxx)

def calibrate_thresholds(samples=3600, fs=360):
    std_devs = []
    ratios = []
    for _ in tqdm(range(samples), desc="Calibrating thresholds"):
        ecg = nk.ecg_simulate(duration=30, sampling_rate=fs, heart_rate=70)
        std_devs.append(detect_chaotic_baseline(ecg, fs))
        ratios.append(detect_f_waves(ecg, fs))
    std_thresh = np.mean(std_devs) + 2 * np.std(std_devs)
    ratio_thresh = np.mean(ratios) + 2 * np.std(ratios)
    return std_thresh, ratio_thresh

std_thresh, ratio_thresh = calibrate_thresholds()
print("Suggested chaotic baseline threshold:", round(std_thresh, 4))
print("Suggested f-wave power ratio threshold:", round(ratio_thresh, 4))

# W/ 3600 samples
# Suggested chaotic baselin threshold: 0.1857
# Suggested f-wave power ratio threshold: 0.3378