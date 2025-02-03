import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import config as cfg
from gwpy.timeseries import TimeSeries

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply bandpass filter.

    Parameters
    ----------
    data : np.array
        Data to be filtered.
    lowcut : float
        Upper cutoff frequency.
    highcut : float
        Lower cutoff frequency.
    fs : int
        The sampling frequency.
    order : int
        The order of the filter.

    Returns
    -------
    data : np.array
        Filtered data.
    """
    nyquist_rate = 0.5 * fs
    low, high = lowcut / nyquist_rate, highcut / nyquist_rate
    numerator, denominator = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(numerator, denominator, data)

def whiten_signal(data):
    """
    Whiten the signal by dividing by the square root of the PSD.

    Parameters
    ----------
    data : np.array
        Data to be whitened.

    Returns
    -------
    data : np.array
        Whitened data.
    """
    fft_data = np.fft.rfft(data)
    norm_fft = fft_data / np.sqrt(np.abs(fft_data) ** 2 + 1e-10)
    return np.fft.irfft(norm_fft)

def notch_filter(data, fs, freq=60, quality_factor=30):
    """
    Remove specific frequencies.

    Parameters
    ----------
    data : np.array
        Data to be processed.
    fs : int
        The sampling frequency.
    freq : int
        Frequency to be deleted.
    quality_factor : int
        Parameter for the irrnotch function.

    Returns
    -------
    data : np.array
        Filtered data.
    """
    nyquist_rate = 0.5 * fs
    w0 = freq / nyquist_rate
    numerator, denominator = iirnotch(w0, quality_factor)
    return filtfilt(numerator, denominator, data)

def analyze_strain_data(data, fs, lowcut=50, highcut=300, fftlength=4):
    ts = TimeSeries(data, sample_rate=fs)
    ts = ts.bandpass(lowcut, highcut)
    ts = ts.whiten(fftlength=fftlength)
    return ts.value

def process_dataframe(df):
    """
    Apply all signal processing functions to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with 'time' and 'strain' columns.

    Returns
    -------
    df : pd.DataFrame
        Data frame with added columns corresponding to processed data.
    """
    df["strain"] = df["strain"].interpolate(method="linear", limit_direction="both")
    df["processed"] = bandpass_filter(df["strain"].values, 35, 350, fs=cfg.FS, order=4)
    df["processed"] = notch_filter(df["processed"].values, cfg.FS, 60, 30)
    df["processed"] = whiten_signal(df["processed"].values)
    df["processed_differently"] = analyze_strain_data(df["strain"].values, cfg.FS)

    return df