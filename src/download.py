import os
import requests
import numpy as np
import pandas as pd
import h5py
import config as cfg

def download_ligo_data(url, data_path):
    """Download LIGO data if not already present.

    Parameters
    ----------
    url : str
        URL address of the data set.
    data_path : str
        Path to the hdf5 file to be downloaded.

    Raises
    ------
    TypeError
        If `url` is not a string or `data_path` are of incorrect types.
    """
    if not isinstance(url, str):
        raise TypeError("URL must be a string.")

    if not isinstance(data_path, str):
        raise TypeError("Data path must be a string.")

    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))

    if not os.path.exists(data_path):
        print(f"Downloading data from {url}")
        try:
            response = requests.get(url, stream=True)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error downloading data: {e}")
        # Downloading the file in parts to not load whole data into memory at once
        with open(data_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Download complete: {data_path}")
    else:
        print(f"File already exists: {data_path}")


def load_ligo_data(data_path, dataset_name="strain/Strain"):
    """Load LIGO strain data from an HDF5 file.

    Parameters
    ----------
    data_path : str
        Path to the hdf5 file.
    dataset_name : str
        Name of the dataset to be loaded.

    Returns
    -------
    np.array
        Array that will be later transformed into a data frame.

    Raises
    ------
    FileNotFoundError
        If the specified `data_path` does not exist.
    KeyError
        If the specified `dataset_name` is not found within the HDF5 file.
    TypeError
        If `data_path` or `dataset_name` are not of type string.
    """
    if not isinstance(data_path, str):
        raise TypeError("Data path must be a string.")

    if not isinstance(dataset_name, str):
        raise TypeError("Dataset name must be a string.")

    try:
        with h5py.File(data_path, 'r') as f:
            strain = f[dataset_name][:]
    except FileNotFoundError:
        raise RuntimeError(f"File not found at {data_path}")
    except KeyError:
        raise RuntimeError(f"Dataset '{dataset_name}' not found in file")

    return np.array(strain)

def create_dataframe(strain, fs):
    """Create a pd.DataFrame with time and strain.

    Parameters
    ----------
    strain : np.array
        Array of values corresponding to strain at times t.
    fs : int
        Sampling frequency of the LIGO detector.

    Returns
    -------
    pd.DataFrame
        Data frame ready to be used in further analysis.

    Raises
    ------
    ValueError
        If `strain` and `time` have different lengths.
    TypeError
        If any of the input parameters are not of the expected type.
    """
    if not isinstance(strain, np.ndarray):
        raise TypeError("Strain data must be a np.array.")

    if not isinstance(fs, int) or fs <= 0:
        raise ValueError("Sampling frequency must be a positive integer.")

    time = np.arange(0, len(strain) / fs, 1 / fs)

    if len(time) != len(strain):
        raise ValueError("The length of the time array does not match the length of the strain array")

    df = pd.DataFrame({"time": time, "strain": strain})
    return df

def save_dataframe(df, csv_path):
    """Save the data frame to a CSV file - done mostly for grading.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to be saved.
    csv_path : str
        Path for the csv file to be saved.

    Raises
    ------
    TypeError
        If input parameters are not of the expected type.
    ValueError
        If `csv_path` already exists and cannot be overwritten.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Data frame must be a pd.DataFrame.")

    if not isinstance(csv_path, str):
        raise TypeError("CSV path must be a string.")

    if not csv_path.endswith('.csv'):
        raise ValueError("The provided CSV path does not have a .csv extension")

    if os.path.exists(csv_path):
        print(f"File already exists: {csv_path}")
        return

    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    try:
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
    except Exception as e:
        raise RuntimeError(f"Error saving data frame: {e}")

def get_data():
    try:
        download_ligo_data(cfg.URL, cfg.DATA_PATH)
        strain_data = load_ligo_data(cfg.DATA_PATH, "strain/Strain")
        ligo_df = create_dataframe(strain_data, cfg.FS)
        save_dataframe(ligo_df, cfg.CSV_PATH)
        return ligo_df
    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")

if __name__ == "__main__":
    df = get_data()
    if df is not None:
        print("Data loaded successfully.")
    else:
        print("Loading data failed.")