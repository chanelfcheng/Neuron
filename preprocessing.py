import mne
import numpy as np
import pandas as pd

def load_file_data(filename):
    df = pd.read_csv(filename, delimiter=",")
    df = df.drop(df.index[0])   # Drop the first row
    df = df.drop(columns=['Sample Index'])   # Drop the first col
    df = df.drop(columns=df.columns[8:])  # Drop cols 8 and after

    # Channel names for OpenBCI Mark IV EEG headset (8 channels)
    ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    raw = mne.io.RawArray(df.values.transpose(), mne.create_info(ch_names, 250, ch_types="eeg"))
    raw._data*=1e-6 # Convert from microvolts to volts

    # Set montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    return raw

def load_np_data(np_array):
    np_array = np_array[:8, :]
    print(np_array.shape)
    ch_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    raw = mne.io.RawArray(np_array, mne.create_info(ch_names, 250, ch_types="eeg"))

    return raw

def filter_eeg_data(raw):
    # Apply notch filter to remove 60Hz noise
    filtered = raw.copy().notch_filter(60, filter_length="auto", phase="zero")

    # Apply bandpass filter to remove noise outside of 1-50Hz
    filtered = filtered.copy().filter(1, 50, fir_design="firwin")

    return filtered

def compute_psd(filtered):
    # Compute PSD
    psd = filtered.compute_psd(method='multitaper', fmin=1, fmax=50)

    return psd

def compute_bands(psd):
    # Get delta, theta, alpha, beta, and gamma bands
    delta = psd.get_data(picks=[0, 1, 2, 3, 4, 5, 6, 7], fmin=1, fmax=4).mean()
    theta = psd.get_data(picks=[0, 1, 2, 3, 4, 5, 6, 7], fmin=4, fmax=8).mean()
    alpha = psd.get_data(picks=[0, 1, 2, 3, 4, 5, 6, 7], fmin=8, fmax=13).mean()
    beta = psd.get_data(picks=[0, 1, 2, 3, 4, 5, 6, 7], fmin=13, fmax=30).mean()
    gamma = psd.get_data(picks=[0, 1, 2, 3, 4, 5, 6, 7], fmin=30, fmax=50).mean()

    return delta, theta, alpha, beta, gamma

def remove_artifacts(filtered, pov=0.9):
    # Apply ICA to remove artifacts
    ica = mne.preprocessing.ICA(n_components=pov)#, random_state=97)
    ica.fit(filtered);
    ica.exclude = [];
    # ica.plot_sources(filtered)
    # ica.plot_components()
    # ica.plot_overlay(filtered, exclude=ica.exclude)
    # ica.plot_properties(filtered, picks=ica.exclude)  # plot_properties causes an error
    ica.apply(filtered);

    return filtered

if __name__ == "__main__":
    raw_focus = load_file_data("OpenBCISession_steve_focus1/OpenBCI-RAW-2023-04-13_17-26-20.csv")