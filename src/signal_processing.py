from scipy.signal import butter, filtfilt, iirnotch

def bandpass_filter(data, lowcut, highcut, fs):
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

    Returns
    -------
    np.array
        Filtered data.
    """
    nyquist_rate = 0.5 * fs
    lowcut = lowcut / nyquist_rate
    highcut = highcut / nyquist_rate
    numerator, denominator = butter(order, [lowcut, highcut], btype='band')
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