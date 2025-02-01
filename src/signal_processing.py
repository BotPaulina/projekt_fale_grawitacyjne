import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import config as cfg

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

def compute_psd(data, fs):
    """
    Compute the power spectral density.

    Parameters
    ----------
    data : to be determined
        Description.
    fs : int
        The sampling frequency.

    Returns
    -------
    data : np.array
        Description.
    """
    pass

def whiten_signal(data, fs):
    """
    Whiten the signal by dividing by the square root of the PSD.

    Parameters
    ----------
    data : np.array
        Data to be whitened.
    fs : int
        The sampling frequency.

    Returns
    -------
    data : np.array
        Whitened data.
    """
    pass


def process_dataframe(df):
    """
    Apply all signal processing functions to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with 'time' and 'strain' columns.
    fs : int
        The sampling frequency.

    Returns
    -------
    df : pd.DataFrame
        Data frame with added column corresponding to processed data.
    """
    df["strain"] = df["strain"].interpolate(method="linear", limit_direction="both")
    df["processed"] = bandpass_filter(df["strain"].values, 35, 350, fs=cfg.FS, order=4)

    return df