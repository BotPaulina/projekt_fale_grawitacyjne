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
        The input data array.
    lowcut : float
        Lower cutoff frequency.
    highcut : float
        Upper cutoff frequency.
    fs : int
        Sampling frequency of the data.
    order : int, optional
        Order of the filter. Default is 2.

    Returns
    -------
    data : np.array
        Filtered data.

    Raises
    ------
    ValueError
        If `lowcut` or `highcut` is not within the valid range (0, 0.5*fs).
        If `order` is less than 1.
        If `data` is not a np.array.
    TypeError
        If any input data are of incorrect types.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a np.array.")

    if not isinstance(lowcut, (int, float)) or not isinstance(highcut, (int, float)):
        raise TypeError("Lowcut and highcut frequencies must be numbers.")

    if not isinstance(fs, int) or not isinstance(order, int):
        raise TypeError("Sampling frequency and order must be integers.")

    if fs <= 0:
        raise ValueError("Sampling frequency `fs` must be greater than zero.")

    if lowcut < 0 or highcut > 0.5 * fs:
        raise ValueError("Cutoff frequencies `lowcut` and `highcut` must be within the valid range (0, 0.5*fs).")

    if order <= 0:
        raise ValueError("Filter order `order` must be at least 1.")

    nyquist_rate = 0.5 * fs
    low, high = lowcut / nyquist_rate, highcut / nyquist_rate

    try:
        numerator, denominator = butter(order, [low, high], btype='band', analog=False)
        return filtfilt(numerator, denominator, data)
    except Exception as e:
        raise RuntimeError(f"Failed to apply bandpass filter: {e}") from e

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

    Raises
    ------
    ValueError
        If `data` is empty.
    TypeError
        If `data` is not a np.array.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a np.array.")

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    try:
        fft_data = np.fft.rfft(data)
        norm_fft = fft_data / np.sqrt(np.abs(fft_data) ** 2 + 1e-10)
        return np.fft.irfft(norm_fft)
    except Exception as e:
        raise RuntimeError(f"Failed to whiten signal: {e}") from e

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
        Frequency to be deleted. Default is 60 Hz.
    quality_factor : int
        Parameter for the `iirnotch` function. Default is 30.

    Returns
    -------
    data : np.array
        Filtered data.

    Raises
    ------
    ValueError
        If `fs` is not a positive integer.
        If `freq` is not within the valid range (0, 0.5*fs).
        If `quality_factor` is less than or equal to zero.
        If `data` is not a np.array.
    TypeError
        If any input data are of incorrect types.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a np.array.")

    if not isinstance(fs, int) or fs <= 0:
        raise ValueError("Sampling frequency `fs` must be a positive integer.")

    if not isinstance(freq, (int, float)) or freq <= 0 or freq >= 0.5 * fs:
        raise ValueError("Frequency `freq` must be within the valid range (0, 0.5*fs).")

    if not isinstance(quality_factor, int) or quality_factor <= 0:
        raise ValueError("Quality factor must be a positive integer.")

    try:
        nyquist_rate = 0.5 * fs
        w0 = freq / nyquist_rate
        numerator, denominator = iirnotch(w0, quality_factor)
        return filtfilt(numerator, denominator, data)
    except Exception as e:
        raise RuntimeError(f"Failed to notch filter data: {e}") from e

def analyze_strain_data(data, fs, lowcut=50, highcut=300, fftlength=4):
    """
    Analyze strain data using gwpy package.

    Parameters
    ----------
    data : np.array
        Data to be analyzed.
    fs : int
        The sampling frequency.
    lowcut : float
        Lower cutoff frequency. Default is 50 Hz.
    highcut : float
        Lower cutoff frequency. Ddefault is 300 Hz.
    fftlength : int or float
        Length of FFT for whitening. Default is 4 seconds.

    Returns
    -------
    ts.value : np.array
        Filtered data.

    Raises
    ------
    ValueError
        If `fs` is not a positive integer.
        If `lowcut` or `highcut` is not within the valid range (0, 0.5*fs).
        If `fftlength` is less than or equal to zero.
        If `data` is not a np.array.
    TypeError
        If any input data are of incorrect types.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a np.array.")

    if not isinstance(fs, int) or fs <= 0:
        raise ValueError("Sampling frequency `fs` must be a positive integer.")

    if not isinstance(lowcut, (int, float)) or not isinstance(highcut, (int, float)):
        raise TypeError("Lowcut and highcut frequencies must be numbers.")

    if lowcut < 0 or highcut > 0.5 * fs:
        raise ValueError("Cutoff frequencies `lowcut` and `highcut` must be within the valid range (0, 0.5*fs).")

    if not isinstance(fftlength, (int, float)) or fftlength <= 0:
        raise ValueError("FFT length `fftlength` must be a positive number.")

    try:
        ts = TimeSeries(data, sample_rate=fs)
        ts = ts.bandpass(lowcut, highcut)
        ts = ts.whiten(fftlength=fftlength)
        return ts.value
    except Exception as e:
        raise RuntimeError(f"Failed to analyze strain data: {e}") from e

def process_dataframe(df):
    """
    Apply all signal processing functions to the data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with 'time' and 'strain' columns.

    Returns
    -------
    df : pd.DataFrame
        Data frame with added columns corresponding to processed data.

    Raises
    ------
    KeyError
        If 'time' or 'strain' column is missing in the data frame.
    TypeError
        If `df` is of incorrect types.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas data frame.")

    if 'time' not in df.columns or 'strain' not in df.columns:
        raise KeyError("Data frame must contain 'time' and 'strain' columns.")

    try:
        df["strain"] = df["strain"].interpolate(method="linear", limit_direction="both")
        df["processed"] = bandpass_filter(df["strain"].values, 35, 350, fs=cfg.FS, order=4)
        df["processed"] = notch_filter(df["processed"].values, cfg.FS, 60, 30)
        df["processed"] = whiten_signal(df["processed"].values)
        df["processed_differently"] = analyze_strain_data(df["strain"].values, cfg.FS)

        return df
    except Exception as e:
        raise RuntimeError(f"Failed to process data frame: {e}") from e