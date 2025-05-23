import wfdb
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

# --- Load and prepare ECG ---
ecg_data = wfdb.rdrecord("mit-bih/106")
fs = ecg_data.fs
ecg_signal = ecg_data.p_signal[:, 0]
duration = 240  # seconds
samples = fs * duration
signal_name = ecg_data.sig_name

print(signal_name)