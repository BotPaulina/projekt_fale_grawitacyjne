from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
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
    np.array
        Filtered data.
    """
    nyquist_rate = 0.5 * fs
    lowcut = lowcut / nyquist_rate
    highcut = highcut / nyquist_rate
    numerator, denominator = butter(order, [lowcut, highcut], fs=fs, btype='band')
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
    return type
        Description.
    """
    pass

def whiten_signal(data, fs):
    """
    Whiten the signal by dividing by the square root of the PSD.

    Parameters
    ----------
    data : to be determined
        Description.
    fs : int
        The sampling frequency.

    Returns
    -------
    return type
        Description.
    """
    pass


def process_dataframe(df, fs):
    """
    Apply all signal processing functions to the dataframe.

    Parameters:
    - df: Pandas DataFrame with 'time' and 'strain' columns
    - fs: Sampling frequency (Hz)

    Returns:
    - Processed DataFrame
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
    df["processed"] = bandpass_filter(df["strain"].values, fs)

    return df